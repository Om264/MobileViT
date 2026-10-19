"""Command-line argument definitions shared by the entry-point scripts."""
from __future__ import annotations

import argparse
import os

NUM_WORKERS = min(8, os.cpu_count() or 2)


def get_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a MobileViT plant-disease classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("data")
    g.add_argument("--data_dir", required=True, help="dataset root (see README for the two supported layouts)")
    g.add_argument("--val_split", type=float, default=0.15, help="used when data has no train/val folders")
    g.add_argument("--test_split", type=float, default=0.15, help="0 disables the test split")
    g.add_argument("--group_regex", default=None,
                   help="regex with ONE capture group giving a group id from the file stem "
                        "(e.g. '^(.+)_\\d+$') so images of the same leaf/plant never straddle splits")

    g = p.add_argument_group("model")
    g.add_argument("--model", default="mobilevit_s", help="any timm model name, e.g. mobilevit_xxs, mobilevitv2_100")
    g.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--img_size", type=int, default=None, help="default: the model's native size (256)")
    g.add_argument("--drop_rate", type=float, default=0.1)
    g.add_argument("--drop_path", type=float, default=0.05)

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--batch_size", type=int, default=32, help="per GPU")
    g.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--scale_lr", action="store_true", help="scale lr by global_batch/256")
    g.add_argument("--opt", default="adamw", help="any timm optimizer name (adamw, sgd, lamb, ...)")
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--warmup_epochs", type=int, default=2)
    g.add_argument("--min_lr_ratio", type=float, default=0.01)
    g.add_argument("--label_smoothing", type=float, default=0.1)
    g.add_argument("--clip_grad", type=float, default=5.0)
    g.add_argument("--ema_decay", type=float, default=0.0, help="0 disables EMA (try 0.999 for big datasets)")

    g = p.add_argument_group("augmentation / imbalance")
    g.add_argument("--aug", choices=["basic", "strong"], default="basic")
    g.add_argument("--mixup", type=float, default=0.0)
    g.add_argument("--cutmix", type=float, default=0.0)
    g.add_argument("--mix_prob", type=float, default=1.0)
    g.add_argument("--class_weights", action="store_true", help="inverse-frequency loss weights (no mixup/cutmix)")

    g = p.add_argument_group("system")
    g.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="fp16")
    g.add_argument("--channels_last", action="store_true")
    g.add_argument("--workers", type=int, default=NUM_WORKERS)
    g.add_argument("--patience", type=int, default=10, help="early stopping on val macro-F1; 0 disables")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output_dir", default="runs")
    g.add_argument("--exp_name", default=None, help="run folder name (default: model name)")
    g.add_argument("--resume", default=None, help="path to last.pt")
    return p


def get_eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on an image-folder dataset")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True, help="folder with <class>/images")
    p.add_argument("--output_dir", default="eval_out")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    return p


def get_predict_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict on an image or a folder of images")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="image file or folder")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_csv", default=None)
    return p


def get_export_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a checkpoint to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default=None, help="default: next to the checkpoint, .onnx")
    p.add_argument("--opset", type=int, default=17)
    return p
