"""Small environment helpers: seeding, device selection, distributed setup, AMP helpers."""
from __future__ import annotations

import contextlib
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

_RANK = 0  # process rank, set by set_rank(); used so only the main process prints


def set_rank(rank: int) -> None:
    global _RANK
    _RANK = rank


def log(*a) -> None:
    """Print from the main process only."""
    if _RANK == 0:
        print(*a, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_dist():
    """Returns (ddp, rank, world_size, local_rank, device). Works with or without torchrun."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(minutes=60))
        device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
        return True, dist.get_rank(), dist.get_world_size(), local_rank, device
    return False, 0, 1, 0, pick_device()


def autocast_ctx(device, enabled, dtype):
    if enabled:
        return torch.autocast(device.type, dtype=dtype, enabled=True)
    return contextlib.nullcontext()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def prep(x, device, channels_last: bool):
    x = x.to(device, non_blocking=True)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return x
