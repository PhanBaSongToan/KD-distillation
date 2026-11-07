"""Multi-teacher / multi-scale response distillation at 256 px.

The single-teacher refinement saturates at ~86.5. The heterogeneous teacher
ensemble (t224@320 + t320@352) reaches 92.86 on test, so we test whether a
stronger *soft target* lifts the 1.05 M student.

Each teacher sees the batch resized to its own best input resolution; the
student always trains at --image-size. Only response KD is used (feature maps
do not align across scales), plus hard CE and self-KD, as in the single-teacher
refinement.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data import make_loaders
from losses import ModelEMA, ce_loss, kd_kl_per_sample, student_loss
from metrics import metrics_from_logits, predict_logits
from models import DistillStudent, HybridStudent, TeacherResNet18, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(opt, total, warmup_frac=0.05):
    warmup = max(1, int(total * warmup_frac))

    def fn(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher", nargs="+", required=True, help="ckpt:size pairs")
    ap.add_argument("--init-student", default=None,
                    help="optional warm start; omit to train from scratch")
    ap.add_argument("--output", required=True)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--lambda-response", type=float, default=0.5)
    ap.add_argument("--lambda-self", type=float, default=0.2)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--growth", type=int, default=16,
                    help="student growth rate; 16 = paper (1.05M), 24 = ~2.2M")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", default="field")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes = make_loaders(
        args.data, args.batch_size, 0, args.image_size, aug=args.aug, seed=args.seed)
    num_classes = len(classes)

    teachers = []
    for spec in args.teacher:
        path, size = spec.rsplit(":", 1)
        ck = torch.load(path, map_location=device)
        m = TeacherResNet18(num_classes, pretrained=False).to(device)
        m.load_state_dict(ck.get("ema") or ck["teacher"])
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        teachers.append((m, int(size)))
        print(f"teacher {Path(path).parent.name} @ {size}")

    student = DistillStudent(num_classes, growth_rate=args.growth).to(device)
    if args.init_student:
        ck = torch.load(args.init_student, map_location=device)
        sd = ck.get("student") or ck.get("ema")
        if any(k.startswith("backbone.") for k in sd):
            sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        student.backbone.load_state_dict(sd, strict=False)
        print("warm start from", args.init_student)
    else:
        print("training student from scratch")
    print("student params:", count_parameters(student.backbone) / 1e6, "M")

    ema = ModelEMA(student, decay=0.999)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = build_scheduler(opt, args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda")

    best, best_ep, history = -1.0, 0, []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        student.train()
        run = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                probs = None
                for m, size in teachers:
                    x = images if size == images.shape[-1] else F.interpolate(
                        images, size=(size, size), mode="bilinear", align_corners=False)
                    p = torch.softmax(m(x)["main_logits"].float(), dim=1)
                    probs = p if probs is None else probs + p
                probs = probs / len(teachers)
                t_logits = torch.log(probs.clamp_min(1e-12))

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                s_out = student(images)
                hard = ce_loss(s_out["main_logits"], labels, None, args.label_smoothing) + \
                    ce_loss(s_out["aux_logits"], labels, None, args.label_smoothing)
                resp = (
                    kd_kl_per_sample(s_out["main_logits"], t_logits, args.tau)
                    + kd_kl_per_sample(s_out["aux_logits"], t_logits, args.tau)
                ).mean()
                self_kd = kd_kl_per_sample(s_out["main_logits"],
                                           s_out["aux_logits"].detach(), args.tau).mean()
                loss = hard + args.lambda_response * resp + args.lambda_self * self_kd
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            ema.update(student)
            run += loss.item()
        sched.step()

        logits, labels_v = predict_logits(ema.module, val_loader, device)
        vm = metrics_from_logits(logits, labels_v, classes)
        score = 0.5 * (vm["accuracy"] + vm["macro_f1"])
        row = {"epoch": epoch, "loss": run / len(train_loader),
               "val_acc": vm["accuracy"], "val_f1": vm["macro_f1"],
               "sec": round(time.time() - start, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best:
            best, best_ep = score, epoch
            torch.save({"ema": ema.module.state_dict(), "classes": classes,
                        "epoch": epoch}, out / "best.pt")

    final = HybridStudent(num_classes, growth_rate=args.growth).to(device)
    sd = torch.load(out / "best.pt", map_location=device)["ema"]
    final.load_state_dict({k[len("backbone."):]: v for k, v in sd.items()
                           if k.startswith("backbone.")}, strict=False)
    final.eval()
    report = {"best_epoch": best_ep, "args": vars(args)}
    for name, loader in (("test", test_loader),):
        lg, lb = predict_logits(final, loader, device)
        report[name] = metrics_from_logits(lg, lb, classes)
        print(name, "acc=%.4f f1=%.4f" % (report[name]["accuracy"], report[name]["macro_f1"]))
    (out / "test_metrics.json").write_text(json.dumps(report, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
