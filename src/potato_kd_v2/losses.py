"""Losses for the improved multi-objective distillation.

Every term keeps the paper's semantics, but the feature term now compares
spatial maps instead of pooled vectors, and the response term can use the
decoupled (TCKD + NCKD) formulation that is known to behave better than plain
KL when the teacher and student disagree about easy classes.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def ce_loss(
    logits: Tensor,
    labels: Tensor,
    weight: Optional[Tensor] = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    return F.cross_entropy(
        logits, labels, weight=weight, label_smoothing=label_smoothing
    )


def soft_ce(logits: Tensor, soft_targets: Tensor) -> Tensor:
    """Cross-entropy against a soft (e.g. mixup) target."""
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def kd_kl_per_sample(student_logits: Tensor, teacher_logits: Tensor, temperature: float) -> Tensor:
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1) * temperature ** 2


def dkd_per_sample(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    alpha: float = 0.5,
    beta: float = 4.0,
    temperature: float = 4.0,
) -> Tensor:
    """Decoupled KD (Zhao et al., CVPR 2022).

    TCKD transfers the target/non-target *confidence* structure, NCKD transfers
    the relation among the non-target classes, which is exactly the inter-class
    information the paper claims response KD provides.
    """
    t = temperature
    s_logit, t_logit = student_logits / t, teacher_logits.detach() / t
    b = s_logit.shape[0]
    idx = torch.arange(b, device=s_logit.device)
    mask = torch.zeros_like(s_logit, dtype=torch.bool)
    mask[idx, labels] = True

    t_bin = torch.stack([t_logit[mask], t_logit[~mask].view(b, -1).sum(1)], dim=1)
    s_bin = torch.stack([s_logit[mask], s_logit[~mask].view(b, -1).sum(1)], dim=1)
    tckd = F.kl_div(F.log_softmax(s_bin / 1.0, 1), F.softmax(t_bin / 1.0, 1),
                    reduction="none").sum(1) * 1.0

    neg_t = t_logit[~mask].view(b, -1)
    neg_s = s_logit[~mask].view(b, -1)
    nckd = F.kl_div(F.log_softmax(neg_s, 1), F.softmax(neg_t, 1),
                    reduction="none").sum(1) * t ** 2
    return alpha * tckd + beta * nckd


def normalize_maps(x: Tensor) -> Tensor:
    return F.normalize(x.flatten(2), dim=1)


def feature_mse(student_map: Tensor, teacher_map: Tensor) -> Tensor:
    """L2-normalised spatial MSE (scale free, channel-wise cosine matching)."""
    s = normalize_maps(student_map)
    t = normalize_maps(teacher_map.detach())
    return F.mse_loss(s, t)


def attention_transfer(student_map: Tensor, teacher_map: Tensor) -> Tensor:
    """Attention transfer on channel-wise spatial energy maps (Zagoruyko 2017)."""
    s = F.normalize(student_map.pow(2).mean(1, keepdim=True).flatten(1), dim=1)
    t = F.normalize(teacher_map.detach().pow(2).mean(1, keepdim=True).flatten(1), dim=1)
    return F.mse_loss(s, t) * student_map.shape[1]


def teacher_loss(
    teacher_output: Dict[str, Tensor],
    labels: Tensor,
    class_weights: Optional[Tensor] = None,
    label_smoothing: float = 0.1,
    aux_weight: float = 1.0,
) -> Tensor:
    main = ce_loss(teacher_output["main_logits"], labels, class_weights, label_smoothing)
    aux = ce_loss(teacher_output["aux_logits"], labels, class_weights, label_smoothing)
    return main + aux_weight * aux


def student_loss(
    student_output: Dict[str, Tensor],
    teacher_output: Dict[str, Tensor],
    labels: Tensor,
    lambdas: Tuple[float, float, float, float] = (1.0, 0.5, 1.0, 0.3),
    tau: float = 4.0,
    tau_prime: float = 4.0,
    class_weights: Optional[Tensor] = None,
    label_smoothing: float = 0.1,
    kd_mode: str = "dkd",
    feature_mode: str = "map",
    aux_weight: float = 1.0,
    soft_targets: Optional[Tensor] = None,
    normalize_kd: bool = True,
) -> Dict[str, Tensor]:
    """lambdas = (hard, feature, response, self)."""
    l_hard, l_feat, l_resp, l_self = lambdas

    if soft_targets is not None:
        hard = soft_ce(student_output["main_logits"], soft_targets) + \
               soft_ce(student_output["aux_logits"], soft_targets)
    else:
        hard = ce_loss(student_output["main_logits"], labels, class_weights, label_smoothing) + \
               aux_weight * ce_loss(student_output["aux_logits"], labels, class_weights, label_smoothing)

    if feature_mode == "map":
        feat = (
            feature_mse(student_output["stage3_feature"], teacher_output["layer3_projected"])
            + feature_mse(student_output["main_feature"], teacher_output["layer4_projected"])
            + 0.5 * attention_transfer(student_output["stage3_feature"], teacher_output["layer3_projected"])
            + 0.5 * attention_transfer(student_output["main_feature"], teacher_output["layer4_projected"])
        )
    elif feature_mode == "vector":
        # baseline behaviour, kept for ablation
        feat = (student_output["main_vector"] - teacher_output["main_vector"].detach()).norm(dim=1).mean()
        feat = feat + (student_output["aux_vector"] - teacher_output["aux_vector"].detach()).norm(dim=1).mean()
    elif feature_mode == "none":
        feat = torch.zeros((), device=labels.device)
    else:
        raise ValueError(feature_mode)

    if kd_mode == "dkd":
        resp = (
            dkd_per_sample(student_output["main_logits"], teacher_output["main_logits"], labels,
                           alpha=1.0, beta=8.0, temperature=tau)
            + dkd_per_sample(student_output["aux_logits"], teacher_output["aux_logits"], labels,
                             alpha=1.0, beta=8.0, temperature=tau)
        ).mean()
    elif kd_mode == "kl":
        resp = (
            kd_kl_per_sample(student_output["main_logits"], teacher_output["main_logits"], tau)
            + kd_kl_per_sample(student_output["aux_logits"], teacher_output["aux_logits"], tau)
        ).mean()
    elif kd_mode == "none":
        resp = torch.zeros((), device=labels.device)
    else:
        raise ValueError(kd_mode)

    self_kd = kd_kl_per_sample(
        student_output["main_logits"], student_output["aux_logits"].detach(), tau_prime
    ).mean()

    # The three KD terms live on very different numeric scales: a map MSE can
    # be ~30 while CE is ~2. Without normalisation the lambdas are meaningless
    # and feature KD silently dominates the objective. Dividing by the detached
    # batch mean puts every term on a comparable scale (~1), so lambda really
    # expresses relative importance. This was the single biggest cause of the
    # student degrading during warm-start refinement.
    if normalize_kd:
        feat = feat / feat.detach().clamp_min(1e-6)
        resp = resp / resp.detach().clamp_min(1e-6)

    total = l_hard * hard + l_feat * feat + l_resp * resp + l_self * self_kd
    return {
        "total": total,
        "hard": hard.detach(),
        "feature": feat.detach(),
        "response": resp.detach(),
        "self": self_kd.detach(),
    }


class ModelEMA:
    """Exponential moving average of model weights (works for any nn.Module)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        # Warm up the decay so the EMA tracks the live model during the first
        # epochs instead of staying pinned to the random initialisation.
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for ema_p, p in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                ema_p.copy_(p)
