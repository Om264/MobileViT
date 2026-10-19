"""Model construction (any timm model, MobileViT by default), EMA and checkpoint helpers."""
from __future__ import annotations

import copy

import timm
import torch


def build_model(name, num_classes, pretrained, drop_rate=0.0, drop_path=0.0):
    """Create any timm model, e.g. mobilevit_xxs / mobilevit_xs / mobilevit_s / mobilevitv2_100."""
    return timm.create_model(
        name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate, drop_path_rate=drop_path
    )


class EMA:
    """Exponential moving average of model weights (independent of the timm version)."""

    def __init__(self, model, decay):
        self.decay, self.step = decay, 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def load_ckpt(path):
    """Load a checkpoint written by training (contains weights + class names + preprocessing)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def model_from_ckpt(ck, device):
    m = build_model(ck["model_name"], len(ck["class_names"]), pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()
