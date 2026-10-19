"""Tests for dataset discovery and splitting. Run with: pytest -q"""
import re
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from plant_cls.data import build_splits, scan_folder  # noqa: E402


def _make_classes(root: Path, classes=("a", "b", "c"), groups=10, per_group=3):
    for c in classes:
        d = root / c
        d.mkdir(parents=True, exist_ok=True)
        for g in range(groups):
            for k in range(per_group):
                Image.new("RGB", (8, 8)).save(d / f"plant{c}{g}_{k}.jpg")


def _args(root, **kw):
    base = dict(data_dir=str(root), val_split=0.2, test_split=0.2, seed=0, group_regex=None)
    base.update(kw)
    return Namespace(**base)


def test_scan_folder_finds_all_images(tmp_path):
    _make_classes(tmp_path)
    samples, classes = scan_folder(tmp_path)
    assert classes == ["a", "b", "c"]
    assert len(samples) == 90


def test_auto_split_is_disjoint_and_complete(tmp_path):
    _make_classes(tmp_path)
    train, val, test, classes = build_splits(_args(tmp_path))
    paths = [p for p, _ in train + val + test]
    assert len(paths) == len(set(paths)) == 90
    assert len(test) > 0 and len(val) > 0
    assert {y for _, y in train} == {0, 1, 2}


def test_group_split_has_no_leakage(tmp_path):
    _make_classes(tmp_path)
    rx = re.compile(r"^(.+)_\d+$")
    train, val, test, _ = build_splits(_args(tmp_path, group_regex=r"^(.+)_\d+$"))

    def groups(split):
        return {rx.search(Path(p).stem).group(1) for p, _ in split}

    assert not groups(train) & groups(val)
    assert not groups(train) & groups(test)
    assert not groups(val) & groups(test)


def test_presplit_layout_is_detected(tmp_path):
    for split in ("train", "val", "test"):
        _make_classes(tmp_path / split, groups=2)
    train, val, test, classes = build_splits(_args(tmp_path))
    assert classes == ["a", "b", "c"]
    assert len(train) == len(val) == len(test) == 18
