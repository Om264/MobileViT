"""Evaluate a checkpoint on a folder dataset, predict on images, export to ONNX."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import IMG_EXT, ImageDataset, eval_transform, scan_folder
from .metrics import evaluate, save_reports
from .model import load_ckpt, model_from_ckpt
from .utils import pick_device


def run_evaluate(args):
    device = pick_device()
    ck = load_ckpt(args.checkpoint)
    classes = ck["class_names"]
    model = model_from_ckpt(ck, device)
    root = Path(args.data_dir)
    unknown = [d.name for d in root.iterdir() if d.is_dir() and d.name not in classes]
    if unknown:
        print(f"[warn] folders not in the checkpoint's class list are ignored: {unknown}")
    samples, _ = scan_folder(root, classes)
    if not samples:
        raise SystemExit("No images found")
    tf = eval_transform(ck["img_size"], ck["mean"], ck["std"])
    loader = DataLoader(ImageDataset(samples, tf), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    res = evaluate(model, loader, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = save_reports(res["y_true"], res["y_pred"], classes, out_dir, "eval")
    print(txt)
    print(f"acc={res['acc']:.4f} macro_f1={res['macro_f1']:.4f} weighted_f1={res['weighted_f1']:.4f}")
    print(f"reports saved to {out_dir}")


@torch.no_grad()
def run_predict(args):
    device = pick_device()
    ck = load_ckpt(args.checkpoint)
    classes = ck["class_names"]
    model = model_from_ckpt(ck, device)
    tf = eval_transform(ck["img_size"], ck["mean"], ck["std"])
    inp = Path(args.input)
    paths = [inp] if inp.is_file() else sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not paths:
        raise SystemExit("No images found")
    k = min(args.topk, len(classes))
    rows = []
    for s in range(0, len(paths), args.batch_size):
        chunk = paths[s:s + args.batch_size]
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        probs = model(batch).softmax(1).cpu()
        top_p, top_i = probs.topk(k, dim=1)
        for p, tp, ti in zip(chunk, top_p, top_i):
            preds = [(classes[i], float(v)) for v, i in zip(tp, ti)]
            rows.append((str(p), preds))
            print(f"{p}: " + ", ".join(f"{c} ({v:.1%})" for c, v in preds))
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path"] + [f"top{i + 1}_{x}" for i in range(k) for x in ("class", "prob")])
            for p, preds in rows:
                w.writerow([p] + [x for c, v in preds for x in (c, f"{v:.6f}")])
        print(f"saved {args.output_csv}")


def run_export(args):
    ck = load_ckpt(args.checkpoint)
    model = model_from_ckpt(ck, torch.device("cpu"))
    s = ck["img_size"]
    out = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".onnx")
    dummy = torch.randn(1, 3, s, s)
    kw = dict(input_names=["image"], output_names=["logits"], opset_version=args.opset,
              dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kw)
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(model, dummy, str(out), **kw)
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(
        {"class_names": ck["class_names"], "img_size": s, "mean": ck["mean"], "std": ck["std"],
         "preprocess": "resize to (img_size, img_size), scale to [0,1], normalise with mean/std"}, indent=2))
    print(f"exported {out} (+ {meta_path.name} with class names and preprocessing)")
