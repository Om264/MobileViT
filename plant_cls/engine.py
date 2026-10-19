"""Training engine: single-GPU / multi-GPU (DDP) training loop with AMP, EMA, accumulation, resume."""
from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import ImageDataset, build_splits, build_transforms
from .metrics import evaluate, plot_history, save_reports
from .model import EMA, build_model, load_ckpt
from .utils import autocast_ctx, log, make_scaler, prep, seed_everything, set_rank, setup_dist


def train_one_epoch(model, base, ema, loader, sampler, optimizer, scheduler, scaler, criterion,
                    mixup_fn, device, args, epoch, amp_on, amp_dtype, ddp):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    n, total, count = len(loader), 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(loader):
        x = prep(x, device, args.channels_last)
        y = y.to(device, non_blocking=True)
        if mixup_fn is not None:
            x, y = mixup_fn(x, y)
        boundary = (i + 1) % args.accum_steps == 0 or (i + 1) == n
        sync = contextlib.nullcontext() if (not ddp or boundary) else model.no_sync()
        with sync:
            with autocast_ctx(device, amp_on, amp_dtype):
                out = model(x)
            loss = criterion(out.float(), y)
            scaler.scale(loss / args.accum_steps).backward()
        if boundary:
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(base.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(base)
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / max(count, 1)


def run_training(args):
    ddp, rank, world, local_rank, device = setup_dist()
    set_rank(rank)
    seed_everything(args.seed + rank)

    use_mix = args.mixup > 0 or args.cutmix > 0
    if use_mix and args.batch_size % 2:
        raise SystemExit("--batch_size must be even when using mixup/cutmix")

    exp = args.exp_name or args.model
    out_dir = Path(args.output_dir) / exp
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()

    # ---- data splits (deterministic, identical on every rank) ----
    train_s, val_s, test_s, classes = build_splits(args)
    num_classes = len(classes)
    if not val_s:
        raise SystemExit("Validation set is empty; adjust --val_split or provide a val/ folder")
    if rank == 0:
        (out_dir / "class_names.json").write_text(json.dumps(classes, indent=2))
        (out_dir / "splits.json").write_text(json.dumps({"train": train_s, "val": val_s, "test": test_s}))
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log(f"classes={num_classes} train={len(train_s)} val={len(val_s)} test={len(test_s)} "
        f"device={device} world_size={world}")

    # ---- model (rank 0 first so pretrained weights are downloaded only once) ----
    if ddp and rank != 0:
        dist.barrier()
    base = build_model(args.model, num_classes, args.pretrained, args.drop_rate, args.drop_path)
    if ddp and rank == 0:
        dist.barrier()
    cfg = timm.data.resolve_data_config({}, model=base)
    img_size = args.img_size or cfg["input_size"][-1]
    mean, std = cfg["mean"], cfg["std"]
    log(f"model={args.model} params={sum(p.numel() for p in base.parameters()) / 1e6:.2f}M "
        f"img_size={img_size} pretrained={args.pretrained}")

    # ---- loaders ----
    tf_train, tf_eval = build_transforms(img_size, mean, std, args.aug)
    ds_train = ImageDataset(train_s, tf_train)
    pin = device.type == "cuda"
    sampler = DistributedSampler(ds_train, world, rank, shuffle=True, seed=args.seed, drop_last=True) if ddp else None
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                              num_workers=args.workers, pin_memory=pin, drop_last=True,
                              persistent_workers=args.workers > 0)
    if len(loader_train) == 0:
        raise SystemExit("Training set smaller than one global batch; lower --batch_size")

    def eval_loader(samples):
        return DataLoader(ImageDataset(samples, tf_eval), batch_size=args.batch_size * 2, shuffle=False,
                          num_workers=args.workers, pin_memory=pin)

    loader_val = eval_loader(val_s) if rank == 0 else None

    # ---- model placement ----
    base.to(device)
    if args.channels_last:
        base.to(memory_format=torch.channels_last)
    ema = EMA(base, args.ema_decay) if args.ema_decay > 0 else None
    model = DDP(base, device_ids=[local_rank] if device.type == "cuda" else None) if ddp else base

    # ---- loss ----
    if use_mix:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, prob=args.mix_prob,
                         label_smoothing=args.label_smoothing, num_classes=num_classes)
        criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn, weight = None, None
        if args.class_weights:
            counts = np.bincount([y for _, y in train_s], minlength=num_classes)
            weight = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)),
                                  dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    # ---- optimiser / scheduler (per-step warmup + cosine decay) ----
    lr = args.lr * (args.batch_size * world * args.accum_steps / 256) if args.scale_lr else args.lr
    optimizer = create_optimizer_v2(base, opt=args.opt, lr=lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loader_train) / args.accum_steps)
    total_steps, warm_steps = steps_per_epoch * args.epochs, steps_per_epoch * args.warmup_epochs

    def lr_lambda(step):
        if step < warm_steps:
            return (step + 1) / max(1, warm_steps)
        prog = (step - warm_steps) / max(1, total_steps - warm_steps)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    amp_on = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = make_scaler(amp_on and amp_dtype == torch.float16)

    # ---- resume ----
    start_epoch, best_f1, best_epoch, history = 0, -1.0, -1, []
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        base.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema"):
            ema.module.load_state_dict(ck["ema"])
            ema.step = ck.get("ema_step", 0)
        start_epoch, best_f1, best_epoch, history = ck["epoch"] + 1, ck["best_f1"], ck["best_epoch"], ck["history"]
        log(f"resumed from {args.resume} at epoch {start_epoch}")

    eval_model = ema.module if ema is not None else base
    meta = dict(model_name=args.model, class_names=classes, img_size=img_size, mean=list(mean), std=list(std))

    # ---- main loop ----
    bad_epochs = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, base, ema, loader_train, sampler, optimizer, scheduler, scaler,
                                  criterion, mixup_fn, device, args, epoch, amp_on, amp_dtype, ddp)
        stop = False
        if rank == 0:
            res = evaluate(eval_model, loader_val, device, amp_on, amp_dtype, args.channels_last)
            row = dict(epoch=epoch + 1, train_loss=tr_loss, val_loss=res["loss"], val_acc=res["acc"],
                       val_macro_f1=res["macro_f1"], lr=optimizer.param_groups[0]["lr"], time=time.time() - t0)
            history.append(row)
            improved = res["macro_f1"] > best_f1
            if improved:
                best_f1, best_epoch, bad_epochs = res["macro_f1"], epoch + 1, 0
                torch.save({"state_dict": eval_model.state_dict(), "epoch": epoch + 1, "best_f1": best_f1, **meta},
                           out_dir / "best.pt")
            else:
                bad_epochs += 1
            torch.save({"model": base.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "ema": ema.module.state_dict() if ema else None, "ema_step": ema.step if ema else 0,
                        "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch, "history": history, **meta},
                       out_dir / "last.pt")
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
            log(f"[{epoch + 1:3d}/{args.epochs}] train_loss={tr_loss:.4f} val_loss={res['loss']:.4f} "
                f"val_acc={res['acc']:.4f} val_f1={res['macro_f1']:.4f} {'*' if improved else ' '} "
                f"({row['time']:.0f}s)")
            stop = args.patience > 0 and bad_epochs >= args.patience
            if stop:
                log(f"early stopping: no val macro-F1 improvement for {args.patience} epochs")
        if ddp:  # also a sync point so the other ranks wait for rank-0 evaluation
            flag = torch.tensor([int(stop)], device=device)
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
        if stop:
            break

    # ---- final evaluation with the best checkpoint (rank 0) ----
    if rank == 0:
        if history:
            plot_history(history, out_dir / "training_curves.png")
        best = load_ckpt(out_dir / "best.pt")
        eval_model.load_state_dict(best["state_dict"])
        results = {"best_epoch": best_epoch, "best_val_macro_f1": best_f1}
        for tag, samples in (("val", val_s), ("test", test_s)):
            if not samples:
                continue
            res = evaluate(eval_model, eval_loader(samples), device, amp_on, amp_dtype, args.channels_last)
            save_reports(res["y_true"], res["y_pred"], classes, out_dir, tag)
            results[tag] = {k: res[k] for k in ("loss", "acc", "macro_f1", "weighted_f1")}
            log(f"{tag.upper()}: acc={res['acc']:.4f} macro_f1={res['macro_f1']:.4f} "
                f"weighted_f1={res['weighted_f1']:.4f}")
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        log(f"done. outputs in {out_dir}")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
