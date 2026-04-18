"""Validation script for Book 2 training and regularization companion code."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn


_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.training_examples import (  # noqa: E402
    cosine_schedule,
    cutout,
    get_adamw_with_warmup,
    mixup,
    warmup_cosine_schedule,
    warmup_schedule,
)


def test_cutout_supports_chw_and_hwc() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    chw = torch.ones(3, 8, 8)
    hwc = np.ones((8, 8, 3), dtype=np.float32)

    chw_aug = cutout(chw, 4)
    hwc_aug = cutout(hwc, 4)

    assert chw_aug.shape == chw.shape
    assert hwc_aug.shape == hwc.shape
    assert (chw_aug == 0).any()
    assert (hwc_aug == 0).any()


def test_cutout_preserves_requested_odd_mask_size() -> None:
    torch.manual_seed(6)
    np.random.seed(6)

    chw = torch.ones(3, 9, 9)
    chw_aug = cutout(chw, 5)

    zero_mask = (chw_aug[0] == 0)
    zero_rows, zero_cols = torch.where(zero_mask)
    assert len(zero_rows) == 25
    assert zero_rows.min().item() == 1
    assert zero_rows.max().item() == 5
    assert zero_cols.min().item() == 2
    assert zero_cols.max().item() == 6


def test_mixup_returns_soft_labels() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    x = torch.arange(24, dtype=torch.float32).view(3, 2, 4)
    y = torch.tensor([0, 1, 2], dtype=torch.long)

    mixed_x, mixed_y = mixup(x, y, alpha=0.4, num_classes=3)

    assert mixed_x.shape == x.shape
    assert mixed_y.shape == (3, 3)
    torch.testing.assert_close(mixed_y.sum(dim=1), torch.ones(3))
    assert torch.all((mixed_y >= 0) & (mixed_y <= 1))


def test_cosine_schedule_hits_min_lr() -> None:
    initial_lr = 1e-2
    min_lr = 1e-5
    total_epochs = 12

    assert cosine_schedule(0, total_epochs, initial_lr, min_lr) == initial_lr
    assert np.isclose(
        cosine_schedule(total_epochs - 1, total_epochs, initial_lr, min_lr),
        min_lr,
    )


def test_warmup_schedule_reaches_target_lr() -> None:
    initial_lr = 3e-4
    warmup_epochs = 5
    assert np.isclose(warmup_schedule(0, 0, initial_lr), initial_lr)
    assert np.isclose(warmup_schedule(7, 0, initial_lr), initial_lr)
    assert np.isclose(warmup_schedule(0, warmup_epochs, initial_lr), initial_lr / 5)
    assert np.isclose(warmup_schedule(4, warmup_epochs, initial_lr), initial_lr)
    assert np.isclose(warmup_schedule(7, warmup_epochs, initial_lr), initial_lr)


def test_warmup_cosine_schedule_matches_zero_indexed_formula() -> None:
    initial_lr = 1e-2
    min_lr = 1e-5
    total_epochs = 10
    warmup_epochs = 3

    assert np.isclose(
        warmup_cosine_schedule(0, total_epochs, warmup_epochs, initial_lr, min_lr),
        initial_lr / warmup_epochs,
    )
    assert np.isclose(
        warmup_cosine_schedule(2, total_epochs, warmup_epochs, initial_lr, min_lr),
        initial_lr,
    )
    assert np.isclose(
        warmup_cosine_schedule(3, total_epochs, warmup_epochs, initial_lr, min_lr),
        initial_lr,
    )
    assert np.isclose(
        warmup_cosine_schedule(total_epochs - 1, total_epochs, warmup_epochs, initial_lr, min_lr),
        min_lr,
    )


def test_adamw_with_warmup_scheduler_reaches_floor() -> None:
    model = nn.Linear(4, 2)
    optimizer, scheduler = get_adamw_with_warmup(
        model,
        lr=1e-3,
        warmup_steps=3,
        total_steps=8,
        warmup_init_lr_factor=1e-2,
    )

    initial_lr = optimizer.param_groups[0]["lr"]
    assert np.isclose(initial_lr, 1e-5)

    lrs = []
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert np.isclose(lrs[0], 3.4e-4)
    assert np.isclose(lrs[1], 6.7e-4)
    assert np.isclose(lrs[2], 1e-3)
    assert max(lrs) <= 1e-3 + 1e-12
    assert np.isclose(lrs[-1], 0.0, atol=1e-12)


def main() -> None:
    print("Running Book 2 training-example validation...")
    test_cutout_supports_chw_and_hwc()
    print("  - cutout works for CHW tensors and HWC arrays")
    test_cutout_preserves_requested_odd_mask_size()
    print("  - cutout preserves the requested odd mask size when the mask fits")
    test_mixup_returns_soft_labels()
    print("  - mixup returns mixed inputs with valid soft labels")
    test_cosine_schedule_hits_min_lr()
    print("  - cosine schedule reaches min_lr on the final epoch")
    test_warmup_schedule_reaches_target_lr()
    print("  - warmup schedule handles zero warmup and reaches the target LR cleanly")
    test_warmup_cosine_schedule_matches_zero_indexed_formula()
    print("  - warmup cosine schedule matches the documented zero-indexed convention")
    test_adamw_with_warmup_scheduler_reaches_floor()
    print("  - AdamW warmup scheduler decays to its final floor")
    print("\nAll Book 2 training-example validations passed.")


if __name__ == "__main__":
    main()
