"""Runnable companion code for Book 2 training and regularization snippets."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def cutout(image, mask_size):
    """Apply cutout to either CHW tensors or HWC arrays."""
    if image.ndim != 3:
        raise ValueError(f"expected 3D image tensor/array, got shape {image.shape}")

    channels_first = image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3)
    if channels_first:
        _, h, w = image.shape
    else:
        h, w = image.shape[:2]

    mask_size = min(mask_size, h, w)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip([cy - mask_size // 2, cy + mask_size // 2], 0, h)
    x1, x2 = np.clip([cx - mask_size // 2, cx + mask_size // 2], 0, w)
    image_aug = image.clone() if torch.is_tensor(image) else image.copy()
    if channels_first:
        image_aug[:, y1:y2, x1:x2] = 0
    else:
        image_aug[y1:y2, x1:x2, ...] = 0
    return image_aug


def mixup(x, y, alpha=0.2, num_classes=None):
    """Apply mixup and always return soft labels."""
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    if y.dim() == 1:
        if num_classes is None:
            raise ValueError(
                "num_classes must be provided for integer labels. "
                "Inferring from a batch is fragile."
            )
        y = torch.eye(num_classes, device=y.device)[y]

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def cosine_schedule(epoch, total_epochs, initial_lr, min_lr=1e-6):
    """Cosine annealing schedule that reaches min_lr on the final epoch."""
    return min_lr + 0.5 * (initial_lr - min_lr) * (
        1 + np.cos(np.pi * epoch / max(1, total_epochs - 1))
    )


def step_decay_schedule(epoch, initial_lr, drop_factor=0.1, drop_epochs=(30, 60, 80)):
    """Piecewise step-decay schedule."""
    lr = initial_lr
    for drop_epoch in drop_epochs:
        if epoch >= drop_epoch:
            lr *= drop_factor
    return lr


def warmup_schedule(epoch, warmup_epochs, initial_lr):
    """Zero-indexed linear warmup for the first few epochs."""
    if warmup_epochs <= 0:
        return initial_lr
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    return initial_lr


def warmup_cosine_schedule(epoch, total_epochs, warmup_epochs, initial_lr, min_lr=1e-6):
    """Zero-indexed linear warmup followed by cosine annealing."""
    if total_epochs <= 1:
        return initial_lr
    if epoch < warmup_epochs:
        return warmup_schedule(epoch, warmup_epochs, initial_lr)
    if total_epochs <= warmup_epochs:
        return initial_lr
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_adamw_with_warmup(model, lr=1e-4, weight_decay=0.01,
                          warmup_steps=1000, total_steps=100000,
                          warmup_init_lr_factor=1e-2):
    """AdamW with linear warmup followed by cosine decay."""
    peak_lr = lr
    optimizer = AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    def lr_lambda(step):
        if total_steps <= 1:
            return 1.0

        if warmup_steps <= 0:
            progress = min(step / max(1, total_steps - 1), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        if step < warmup_steps:
            warmup_progress = step / max(1, warmup_steps)
            return warmup_init_lr_factor + (1.0 - warmup_init_lr_factor) * warmup_progress

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    if warmup_steps > 0:
        initial_lr = peak_lr * warmup_init_lr_factor
        for param_group in optimizer.param_groups:
            param_group["lr"] = initial_lr
    return optimizer, scheduler


__all__ = [
    "cosine_schedule",
    "cutout",
    "get_adamw_with_warmup",
    "mixup",
    "step_decay_schedule",
    "warmup_cosine_schedule",
    "warmup_schedule",
]
