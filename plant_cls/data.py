"""Dataset discovery, train/val/test splitting, torch Dataset and image transforms."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T

from .utils import log

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scan_folder(root, classes=None):
    """Scan root/<class>/**/*.img. Returns (samples=[(path, label)], classes)."""
    root = Path(root)
    if classes is None:
        classes = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    samples = []
    for idx, c in enumerate(classes):
        cdir = root / c
        if not cdir.is_dir():
            continue
        for p in sorted(cdir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                samples.append((str(p), idx))
    return samples, classes


def _split(samples, frac, seed, group_regex=None):
    """Hold out `frac` of samples. Returns (kept, held_out)."""
    idx = np.arange(len(samples))
    labels = np.array([s[1] for s in samples])
    if group_regex:
        # Group-aware split: all images sharing a group id (first capture group of the regex
        # applied to the file stem) stay on the same side -> avoids same-leaf leakage.
        rx = re.compile(group_regex)
        groups = []
        for p, _ in samples:
            stem = Path(p).stem
            m = rx.search(stem)
            groups.append(m.group(1) if m and m.groups() else stem)
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed).split(idx, labels, groups))
    else:
        try:
            a, b = train_test_split(idx, test_size=frac, random_state=seed, stratify=labels)
        except ValueError:
            log("[warn] stratified split impossible (a class has too few images); using random split")
            a, b = train_test_split(idx, test_size=frac, random_state=seed)
    return [samples[i] for i in a], [samples[i] for i in b]


def build_splits(args):
    """Returns (train, val, test, classes); each split is a list of (path, label).

    Layout A (pre-split):  data/train/<class>, data/val|valid|validation/<class>, optional data/test/<class>
    Layout B (single root): data/<class>/*.jpg -> split with --val_split / --test_split
    """
    root = Path(args.data_dir)
    sub = {n: root / n for n in ("train", "val", "valid", "validation", "test") if (root / n).is_dir()}
    val_dir = next((sub[n] for n in ("val", "valid", "validation") if n in sub), None)
    if "train" in sub and val_dir is not None:
        train, classes = scan_folder(sub["train"])
        val, _ = scan_folder(val_dir, classes)
        test = scan_folder(sub["test"], classes)[0] if "test" in sub else []
        return train, val, test, classes
    samples, classes = scan_folder(root)
    if not samples:
        raise SystemExit(f"No images found under {root}")
    if args.test_split > 0:
        rest, test = _split(samples, args.test_split, args.seed, args.group_regex)
    else:
        rest, test = samples, []
    val_frac = args.val_split / (1.0 - args.test_split)
    train, val = _split(rest, val_frac, args.seed + 1, args.group_regex)
    return train, val, test, classes


class ImageDataset(Dataset):
    """Simple (path, label) dataset that always returns RGB images."""

    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.transform(img), y


def eval_transform(img_size, mean, std):
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean, std)])


def build_transforms(img_size, mean, std, aug="basic"):
    """Returns (train_transform, eval_transform)."""
    train_ops = [
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05),
    ]
    if aug == "strong":
        train_ops.append(T.TrivialAugmentWide())
    train_ops += [T.ToTensor(), T.Normalize(mean, std), T.RandomErasing(p=0.25)]
    return T.Compose(train_ops), eval_transform(img_size, mean, std)
