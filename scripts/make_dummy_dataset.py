#!/usr/bin/env python3
"""Create a small synthetic image-folder dataset for smoke-testing the pipeline (no real data needed).

Usage:
    python scripts/make_dummy_dataset.py --out data/dummy --classes 4 --per_class 60
    python train.py --data_dir data/dummy --no-pretrained --epochs 2 --batch_size 8 --model mobilevit_xxs

Files are named plant<class><group>_<k>.jpg, so you can also try the leakage-safe split:
    --group_regex '^(plant\\d+)_\\d+$'
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/dummy")
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--per_class", type=int, default=60)
    ap.add_argument("--images_per_plant", type=int, default=3, help="images sharing one group id")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    s = args.size
    yy, xx = np.ogrid[:s, :s]
    for c in range(args.classes):
        cdir = out / f"class_{c:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        base = rng.integers(40, 215, size=3)
        for i in range(args.per_class):
            group, k = divmod(i, args.images_per_plant)
            img = np.clip(base + rng.normal(0, 25, (s, s, 3)), 0, 255).astype(np.uint8)
            cy, cx = rng.integers(s // 4, 3 * s // 4, size=2)
            r = s // (4 + c)
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            img[mask] = np.array([(c * 60) % 255, (255 - c * 50) % 255, (c * 90) % 255], dtype=np.uint8)
            Image.fromarray(img).save(cdir / f"plant{c}{group:03d}_{k}.jpg")
    total = args.classes * args.per_class
    print(f"wrote {total} images in {args.classes} classes to {out}")


if __name__ == "__main__":
    main()
