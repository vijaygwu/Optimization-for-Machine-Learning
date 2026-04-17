"""Runnable companion code for Chapter 14 loss-function examples.

This module mirrors the Chapter 14 code listings in the book.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import torchvision.models as tv_models
except ImportError:  # pragma: no cover - optional dependency at import time
    tv_models = None


def select_device() -> torch.device:
    """Pick the best available training device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cross_entropy_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Categorical cross-entropy that supports hard labels and soft labels.

    Args:
        logits: Tensor of shape ``(N, C)`` containing raw, unnormalized logits.
        targets: Tensor of shape ``(N,)`` with integer class indices, or
            tensor of shape ``(N, C)`` with one-hot / soft labels.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape (N, C); got {tuple(logits.shape)}"
        )

    log_probs = F.log_softmax(logits, dim=-1)

    if targets.ndim == 2:
        if targets.shape != logits.shape:
            raise ValueError(
                f"soft-label targets must match logits shape; "
                f"got {tuple(targets.shape)} vs {tuple(logits.shape)}"
            )
        targets = targets.to(device=logits.device, dtype=logits.dtype)
        return -(targets * log_probs).sum(dim=-1).mean()

    if targets.ndim != 1:
        raise ValueError(
            "targets must have shape (N,) for hard labels "
            "or (N, C) for soft labels"
        )

    targets = targets.to(device=logits.device, dtype=torch.long)
    return F.nll_loss(log_probs, targets)


class SuperResolutionNet(nn.Module):
    """Minimal super-resolution model used in the Chapter 14 training example."""

    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(
                scale_factor=upscale_factor,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class PairedSuperResolutionDataset(Dataset):
    """
    Dataset that pairs low-resolution and high-resolution images by filename.
    """

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(self, lr_dir: str | Path, hr_dir: str | Path):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.pairs: list[tuple[Path, Path]] = []

        for hr_path in sorted(self.hr_dir.iterdir()):
            if hr_path.suffix.lower() not in self.IMAGE_SUFFIXES:
                continue
            lr_path = self.lr_dir / hr_path.name
            if lr_path.exists():
                self.pairs.append((lr_path, hr_path))

        if not self.pairs:
            raise ValueError("No matching LR/HR image pairs found")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        lr_path, hr_path = self.pairs[idx]
        with Image.open(lr_path) as lr_image:
            lr_tensor = _pil_to_tensor(lr_image.convert("RGB"))
        with Image.open(hr_path) as hr_image:
            hr_tensor = _pil_to_tensor(hr_image.convert("RGB"))
        return lr_tensor, hr_tensor


def _default_vgg19_features() -> nn.Sequential:
    if tv_models is None:
        raise ImportError(
            "torchvision is required to build the default "
            "VGG19 feature extractor"
        )

    weights_enum = getattr(tv_models, "VGG19_Weights", None)
    checkpoint_name = "vgg19-dcbb9e9d.pth"
    if weights_enum is not None:
        checkpoint_name = weights_enum.DEFAULT.url.rsplit("/", 1)[-1]

    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        raise RuntimeError(
            "Default PerceptualLoss expects locally cached VGG19 ImageNet "
            f"weights at {checkpoint_path}. Pre-download the checkpoint or "
            "pass a custom feature_extractor for an offline fallback."
        )

    model = tv_models.vgg19(weights=None)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.features


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using a frozen feature extractor.

    By default this uses locally cached pretrained VGG19 features. Tests and
    lightweight offline examples can inject a smaller ``feature_extractor``
    and corresponding ``layer_indices`` mapping instead.
    """

    DEFAULT_LAYER_INDICES = {
        "relu1_2": 3,
        "relu2_2": 8,
        "relu3_3": 15,
        "relu3_4": 17,
        "relu4_3": 24,
        "relu4_4": 26,
        "relu5_3": 33,
        "relu5_4": 35,
    }

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        layers: Sequence[str] = ("relu2_2", "relu4_4"),
        weights: Optional[Sequence[float]] = None,
        normalize_features: bool = True,
        feature_extractor: Optional[nn.Sequential] = None,
        layer_indices: Optional[Mapping[str, int]] = None,
    ):
        super().__init__()

        self.layer_indices = dict(layer_indices or self.DEFAULT_LAYER_INDICES)
        self.layers = tuple(layers)
        self.weights = (
            list(weights) if weights is not None else [1.0] * len(self.layers)
        )
        self.normalize_features = normalize_features

        if len(self.weights) != len(self.layers):
            raise ValueError("weights must match the number of layers")

        for layer in self.layers:
            if layer not in self.layer_indices:
                raise ValueError(
                    f"Unknown layer '{layer}'. "
                    f"Valid options: {sorted(self.layer_indices)}"
                )

        if feature_extractor is None:
            feature_extractor = _default_vgg19_features()
        if not isinstance(feature_extractor, nn.Sequential):
            raise TypeError(
                "feature_extractor must be an nn.Sequential instance"
            )

        max_idx = max(self.layer_indices[layer] for layer in self.layers)
        children = list(feature_extractor.children())
        if max_idx >= len(children):
            raise ValueError(
                f"feature_extractor has {len(children)} layers, "
                f"but layer index {max_idx} was requested"
            )

        self.feature_extractor = nn.Sequential(*children[: max_idx + 1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.register_buffer("mean", self.MEAN.clone())
        self.register_buffer("std", self.STD.clone())

    def _validate_input(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 4:
            raise ValueError(f"expected 4D input (N, C, H, W), got {tensor.ndim}D")
        if tensor.size(1) != 3:
            raise ValueError(f"expected 3 RGB channels, got {tensor.size(1)}")

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def extract_features(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        tensor = self.normalize(tensor)
        features: dict[str, torch.Tensor] = {}

        for idx, layer in enumerate(self.feature_extractor):
            tensor = layer(tensor)
            for name, layer_idx in self.layer_indices.items():
                if idx == layer_idx and name in self.layers:
                    features[name] = tensor

        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._validate_input(pred)
        self._validate_input(target)

        # Note: caller must ensure this module is on the correct device
        # before calling forward(). Use .to(device) at initialization time,
        # not here in the hot path.

        pred_features = self.extract_features(pred)
        with torch.no_grad():
            target_features = self.extract_features(target)

        loss = pred.new_tensor(0.0)
        for layer, weight in zip(self.layers, self.weights):
            pred_layer = pred_features[layer]
            target_layer = target_features[layer]
            # ``reduction='mean'`` already divides by N * C * H * W.
            layer_loss = F.mse_loss(pred_layer, target_layer)
            if self.normalize_features:
                # Optionally downweight wider layers by their channel count.
                layer_loss = layer_loss / pred_layer.size(1)
            loss = loss + weight * layer_loss

        return loss


__all__ = [
    "PairedSuperResolutionDataset",
    "PerceptualLoss",
    "SuperResolutionNet",
    "cross_entropy_manual",
    "select_device",
]
