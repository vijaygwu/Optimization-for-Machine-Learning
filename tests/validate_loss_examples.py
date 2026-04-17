"""Validation script for the Chapter 14 loss-function companion code."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add the repository root to Python path for imports to work from any directory.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.loss_examples import (  # noqa: E402
    PairedSuperResolutionDataset,
    PerceptualLoss,
    SuperResolutionNet,
    cross_entropy_manual,
    select_device,
)


def _create_image_pairs(root: Path, num_pairs: int = 4) -> tuple[Path, Path]:
    lr_dir = root / "LR"
    hr_dir = root / "HR"
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for idx in range(num_pairs):
        hr = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
        lr = hr.reshape(8, 2, 8, 2, 3).mean(axis=(1, 3)).astype(np.uint8)

        Image.fromarray(lr).save(lr_dir / f"{idx:04d}.png")
        Image.fromarray(hr).save(hr_dir / f"{idx:04d}.png")

    return lr_dir, hr_dir


def test_cross_entropy_manual_matches_pytorch_hard_labels() -> None:
    logits = torch.tensor([[1.2, -0.5, 0.3], [0.1, 0.8, -1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    actual = cross_entropy_manual(logits, targets)
    expected = F.cross_entropy(logits, targets)

    torch.testing.assert_close(actual, expected)


def test_cross_entropy_manual_matches_pytorch_soft_labels() -> None:
    logits = torch.tensor([[1.2, -0.5, 0.3], [0.1, 0.8, -1.0]], dtype=torch.float32)
    targets = torch.tensor([[0.7, 0.2, 0.1], [0.05, 0.9, 0.05]], dtype=torch.float32)

    actual = cross_entropy_manual(logits, targets)
    expected = F.cross_entropy(logits, targets)

    torch.testing.assert_close(actual, expected)


def test_paired_super_resolution_dataset_reads_matching_pairs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lr_dir, hr_dir = _create_image_pairs(Path(tmp), num_pairs=3)
        dataset = PairedSuperResolutionDataset(lr_dir, hr_dir)

        assert len(dataset) == 3
        lr_image, hr_image = dataset[0]
        assert tuple(lr_image.shape) == (3, 8, 8)
        assert tuple(hr_image.shape) == (3, 16, 16)


def test_perceptual_training_step_updates_generator() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lr_dir, hr_dir = _create_image_pairs(Path(tmp), num_pairs=4)
        dataset = PairedSuperResolutionDataset(lr_dir, hr_dir)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        device = select_device()
        generator = SuperResolutionNet().to(device)
        feature_extractor = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        perceptual_loss = PerceptualLoss(
            layers=("relu_small_1", "relu_small_2"),
            weights=[1.0, 0.5],
            feature_extractor=feature_extractor,
            layer_indices={"relu_small_1": 1, "relu_small_2": 3},
        ).to(device)
        l1_loss = nn.L1Loss()
        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

        before = [param.detach().clone() for param in generator.parameters()]

        lr_images, hr_images = next(iter(dataloader))
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        sr_images = generator(lr_images)
        loss_pixel = l1_loss(sr_images, hr_images)
        loss_perceptual = perceptual_loss(sr_images, hr_images)
        total_loss = loss_pixel + 0.1 * loss_perceptual

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        deltas = [
            (after.detach() - before_param).abs().sum().item()
            for after, before_param in zip(generator.parameters(), before)
        ]

        assert total_loss.item() > 0.0
        assert max(deltas) > 0.0


def test_perceptual_loss_matches_manual_layer_average() -> None:
    feature_extractor = nn.Sequential(
        nn.Identity(),
        nn.Identity(),
    )
    perceptual_loss = PerceptualLoss(
        layers=("identity",),
        weights=[0.5],
        feature_extractor=feature_extractor,
        layer_indices={"identity": 1},
    )

    pred = torch.tensor(
        [[[[1.0, 3.0]], [[2.0, 4.0]], [[5.0, 7.0]]]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[[[0.0, 1.0]], [[2.0, 1.0]], [[1.0, 2.0]]]],
        dtype=torch.float32,
    )

    actual = perceptual_loss(pred, target)
    pred_norm = perceptual_loss.normalize(pred)
    target_norm = perceptual_loss.normalize(target)
    manual = 0.5 * (((pred_norm - target_norm) ** 2).mean() / pred.size(1))
    torch.testing.assert_close(actual, manual)


def main() -> None:
    print("Running Chapter 14 loss-example validation...")
    test_cross_entropy_manual_matches_pytorch_hard_labels()
    print("  - hard-label cross-entropy matches PyTorch")
    test_cross_entropy_manual_matches_pytorch_soft_labels()
    print("  - soft-label cross-entropy matches PyTorch")
    test_paired_super_resolution_dataset_reads_matching_pairs()
    print("  - paired super-resolution dataset loads matching LR/HR files")
    test_perceptual_training_step_updates_generator()
    print("  - perceptual training step updates generator weights")
    test_perceptual_loss_matches_manual_layer_average()
    print("  - perceptual layer scaling matches the manual mean-and-channel formula")
    print("\nAll Chapter 14 validations passed.")


if __name__ == "__main__":
    main()
