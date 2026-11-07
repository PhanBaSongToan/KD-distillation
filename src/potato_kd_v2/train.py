"""Train the ResNet18 teacher or the distilled student on the potato dataset.

Two-phase design (the main difference from the paper's online baseline):

  1. `--mode teacher` trains ResNet18 to convergence with a modern recipe
     (cosine schedule, label smoothing, EMA, strong field augmentation).
     This raises the distillation ceiling: in the baseline the teacher was
     itself under-trained and updated jointly with the student.
  2. `--mode student` freezes that teacher and distils it offline into the
     paper's hybrid student, with map-level feature KD, decoupled response KD
     and EMA. On-the-fly online teaching is kept available via `--online`.

Examples
--------
python train.py --mode teacher --data ../potato_kd_baseline/data/potato_split_448 \
    --output runs/teacher_r18_224 --epochs 60 --image-size 224 --batch-size 32 --lr 3e-4

python train.py --mode student --data ../potato_kd_baseline/data/potato_split_448 \
    --teacher runs/teacher_r18_224/best.pt --output runs/student_224 --epochs 80
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data import TTADataset, make_loaders
from losses import ModelEMA, kd_kl_per_sample, soft_ce, student_loss, teacher_loss
from metrics import (metrics_from_logits, predict_logits, tune_logit_bias)
from models import DistillStudent, HybridStudent, TeacherResNet18, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, total_steps: int, warmup_frac: float, kind: str = "cosine"):
    warmup = max(1, int(total_steps * warmup_frac))

    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        if kind == "cosine":
            p = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def mixup_batch(images, labels, num_classes, alpha, device):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=device)
    mixed = lam * images + (1 - lam) * images[perm]
    target = torch.zeros(images.size(0), num_classes, device=device)
    target.scatter_(1, labels[:, None], lam)
    target.scatter_add_(1, labels[perm][:, None], torch.full_like(labels[:, None], 1 - lam, dtype=torch.float))
    return mixed, target


def make_class_weights(dataset, device):
    counts = np.bincount(np.array(dataset.targets), minlength=len(dataset.classes)).astype(np.float64)
    w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def run_epoch(student, teacher, loader, optimizers, scaler, device, args, epoch, num_classes,
              class_weights, ema_students, ema_teacher):
    teacher_mode = teacher is not None and student is None
    online = teacher is not None and student is not None and "teacher" in optimizers
    if teacher is not None and (teacher_mode or online):
        teacher.train()
    if student is not None:
        student.train()
    totals = {"loss": 0.0}
    n_batches = len(loader)
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        soft = None
        if args.mixup > 0 and np.random.rand() < 0.5:
            images, soft = mixup_batch(images, labels, num_classes, args.mixup, device)

        if teacher_mode:
            opt = optimizers["teacher"]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                out = teacher(images)
                if soft is not None:
                    loss = 2.0 * soft_ce(out["main_logits"], soft)
                else:
                    loss = teacher_loss(out, labels, class_weights, args.label_smoothing,
                                        args.aux_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if ema_teacher is not None:
                ema_teacher.update(teacher)
            totals["loss"] += loss.item()
            continue

        if teacher is not None and online:  # paper setting: update teacher first
            opt = optimizers["teacher"]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                out_t = teacher(images)
                if soft is not None:
                    loss_t = 2.0 * soft_ce(out_t["main_logits"], soft)
                else:
                    loss_t = teacher_loss(out_t, labels, class_weights, args.label_smoothing,
                                          args.aux_weight)
            scaler.scale(loss_t).backward()
            scaler.step(opt)
            scaler.update()
            if ema_teacher is not None:
                ema_teacher.update(teacher)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.amp):
            t_out = teacher(images)
            t_out = {k: v.float() for k, v in t_out.items()}
        t_out_detached = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in t_out.items()}
        t_out_detached["layer3_projected"] = student.proj3(t_out_detached["layer3_feature"])
        t_out_detached["layer4_projected"] = student.proj4(t_out_detached["layer4_feature"])

        opt = optimizers["student"]
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            s_out = student(images)
            ramp = getattr(args, "_ramp", 1.0)
            losses = student_loss(
                s_out, t_out_detached, labels,
                lambdas=(args.lambda_hard, args.lambda_feature * ramp,
                         args.lambda_response * ramp, args.lambda_self * ramp),
                tau=args.tau, tau_prime=args.tau_prime,
                class_weights=class_weights,
                label_smoothing=args.label_smoothing,
                kd_mode=args.kd,
                feature_mode=args.feature,
                aux_weight=args.aux_weight,
                soft_targets=soft,
                normalize_kd=args.normalize_kd,
            )
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        if ema_students is not None:
            ema_students.update(student)
        totals["loss"] += losses["total"].item()

    return totals["loss"] / max(n_batches, 1)


def evaluate_split(model, loader, device, classes, tta=False):
    logits, labels = predict_logits(model, loader, device, tta=tta)
    return logits, labels, metrics_from_logits(logits, labels, classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "student"], default="teacher")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--teacher", default=None, help="teacher checkpoint for offline KD")
    ap.add_argument("--online", action="store_true", help="train teacher jointly (paper setting)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--aug", default="strong", choices=["paper", "field", "strong", "fast"])
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--class-weights", action="store_true", help="inverse-frequency CE")
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--lambda-hard", type=float, default=1.0)
    ap.add_argument("--lambda-feature", type=float, default=0.5)
    ap.add_argument("--lambda-response", type=float, default=1.0)
    ap.add_argument("--lambda-self", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--tau-prime", type=float, default=4.0)
    ap.add_argument("--kd", default="dkd", choices=["dkd", "kl", "none"])
    ap.add_argument("--feature", default="map", choices=["map", "vector", "none"])
    ap.add_argument("--no-normalize-kd", dest="normalize_kd", action="store_false",
                    help="reproduce the paper loss scale: no per-term normalisation")
    ap.set_defaults(normalize_kd=True)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--growth", type=int, default=16,
                    help="student growth rate; 16 = paper (1.05M), 24 = ~2.2M")
    ap.add_argument("--track", default="mean", choices=["acc", "macro_f1", "mean"])
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--kd-ramp", type=int, default=0,
                    help="linearly ramp feature/response/self KD weights over N epochs")
    ap.add_argument("--init-student", default=None,
                    help="warm-start the student backbone from a checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--train-on-val", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.lr is None:
        args.lr = 3e-4 if args.mode == "teacher" else 1e-3

    train_loader, val_loader, test_loader, classes = make_loaders(
        args.data, args.batch_size, args.num_workers, args.image_size,
        aug=args.aug, train_on_val=args.train_on_val, seed=args.seed)
    num_classes = len(classes)

    train_set = train_loader.dataset
    if args.class_weights:
        if hasattr(train_set, "targets"):
            targets = np.array(train_set.targets)
        else:
            targets = np.array([s[1] for s in train_set.samples])
        counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
        cw = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)),
                          dtype=torch.float32, device=device)
        print("class weights:", [round(float(x), 3) for x in cw.cpu()])
    else:
        cw = None

    teacher = None
    student = None
    ema_teacher = None
    ema_students = None

    if args.mode == "teacher":
        teacher = TeacherResNet18(num_classes, pretrained=True, dropout=args.dropout).to(device)
        params = teacher.parameters()
        print("teacher params:", count_parameters(teacher) / 1e6, "M")
        ema_teacher = ModelEMA(teacher, decay=args.ema)
        student = None
    else:
        if args.online:
            teacher = TeacherResNet18(num_classes, pretrained=True, dropout=args.dropout).to(device)
            ema_teacher = ModelEMA(teacher, decay=args.ema)
        else:
            if not args.teacher:
                raise SystemExit("--teacher checkpoint required for offline distillation")
            ckpt = torch.load(args.teacher, map_location=device)
            state = ckpt.get("ema") or ckpt.get("teacher")
            teacher = TeacherResNet18(num_classes, pretrained=False).to(device)
            teacher.load_state_dict(state)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
        student = DistillStudent(num_classes, dropout=args.dropout,
                                 growth_rate=args.growth).to(device)
        if args.init_student:
            ck = torch.load(args.init_student, map_location=device)
            sd = ck.get("student_ema_full") or ck.get("student") or ck.get("ema")
            if any(k.startswith("backbone.") for k in sd):
                sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
            else:
                sd = {k: v for k, v in sd.items() if not k.startswith("proj3.")}
            missing, unexpected = student.backbone.load_state_dict(sd, strict=False)
            print(f"warm start from {args.init_student}: missing={len(missing)} unexpected={len(unexpected)}")
        print("student backbone params:", count_parameters(student.backbone) / 1e6, "M")
        ema_students = ModelEMA(student, decay=args.ema)

    optimizers = {}
    if teacher is not None and (args.mode == "teacher" or args.online):
        optimizers["teacher"] = torch.optim.AdamW(teacher.parameters(), lr=args.lr,
                                                  weight_decay=args.weight_decay)
    if student is not None:
        optimizers["student"] = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                                  weight_decay=args.weight_decay)
    for opt in optimizers.values():
        scheduler = build_scheduler(opt, args.epochs * len(train_loader), args.warmup)
        opt._sched = scheduler  # type: ignore[attr-defined]

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    model_for_eval = ema_teacher.module if args.mode == "teacher" else ema_students.module

    best_score, best_epoch, history = -1.0, 0, []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        args._ramp = 1.0 if args.kd_ramp <= 0 else min(1.0, epoch / args.kd_ramp)
        loss = run_epoch(student, teacher, train_loader, optimizers, scaler, device,
                         args, epoch, num_classes, cw, ema_students, ema_teacher)
        for opt in optimizers.values():
            opt._sched.step()  # type: ignore[attr-defined]

        logits, labels, val_m = evaluate_split(model_for_eval, val_loader, device, classes)
        score = {"acc": val_m["accuracy"], "macro_f1": val_m["macro_f1"],
                 "mean": 0.5 * (val_m["accuracy"] + val_m["macro_f1"])}[args.track]
        row = {"epoch": epoch, "loss": loss, "val_accuracy": val_m["accuracy"],
               "val_macro_f1": val_m["macro_f1"],
               "val_balanced_acc": val_m["balanced_accuracy"], "seconds": round(time.time() - start, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)

        if score > best_score:
            best_score, best_epoch = score, epoch
            ckpt = {"epoch": epoch, "classes": classes, "args": vars(args)}
            if args.mode == "teacher":
                ckpt["teacher"] = teacher.state_dict()
                ckpt["ema"] = ema_teacher.module.state_dict()
            else:
                ckpt["student"] = student.state_dict()
                ckpt["ema"] = ema_students.module.state_dict()
                at = (out_dir / "all.pt")
                torch.save({"teacher": teacher.state_dict() if teacher is not None else
                            torch.load(args.teacher, map_location="cpu").get("ema"),
                            "stud_ema": ema_students.module.state_dict(),
                            "stud_raw": student.state_dict(),
                            "classes": classes,
                            "student_ema_full": ema_students.module.state_dict()}, at)
            torch.save(ckpt, out_dir / "best.pt")
        elif args.patience and epoch - best_epoch >= args.patience:
            print(f"early stop at {epoch}")
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Final evaluation with EMA weights, plain + TTA, plus validation-only
    # logit-bias calibration.
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    if args.mode == "teacher":
        final = TeacherResNet18(num_classes, pretrained=False)
        final.load_state_dict(ckpt["ema"])
    else:
        final = HybridStudent(num_classes, growth_rate=args.growth)
        sd = {k.replace("backbone.", ""): v for k, v in ckpt["ema"].items()
              if k.startswith("backbone.")}
        final.load_state_dict(sd)
    final = final.to(device).eval()

    report = {"args": vars(args), "best_epoch": best_epoch,
              "params_millions": count_parameters(final) / 1e6}

    val_logits, val_labels = predict_logits(final, val_loader, device, tta=False)
    report["val"] = metrics_from_logits(val_logits, val_labels, classes)

    tta_set = TTADataset(Path(args.data) / "test", args.image_size, views="rich")
    from torch.utils.data import DataLoader
    tta_loader = DataLoader(tta_set, batch_size=max(1, args.batch_size // 2), shuffle=False,
                            num_workers=0, pin_memory=True)
    for name, loader, use_tta in (
        ("test", test_loader, False),
        ("test_tta", tta_loader, True),
    ):
        logits, labels = predict_logits(final, loader, device, tta=use_tta)
        report[name] = metrics_from_logits(logits, labels, classes)
        bias = tune_logit_bias(val_logits, val_labels, classes)
        report[name + "_calibrated"] = metrics_from_logits(logits, labels, classes, bias)
        np.save(out_dir / f"{name}_logits.npy", logits)
        np.save(out_dir / f"{name}_labels.npy", labels)
        report[name + "_bias"] = bias.tolist()

    (out_dir / "test_metrics.json").write_text(json.dumps(report, indent=2))
    for k in ("test", "test_tta", "test_calibrated", "test_tta_calibrated"):
        if k in report:
            print(k, "acc=%.4f f1=%.4f" % (report[k]["accuracy"], report[k]["macro_f1"]))


if __name__ == "__main__":
    main()
