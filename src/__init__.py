"""Companion code exports for Optimization for Machine Learning.

This module uses lazy imports for torch-dependent components to avoid
ImportError when torch is not installed (e.g., when using only the
numpy-based optimizers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import helpers for torch-dependent modules
def __getattr__(name: str):
    """Lazy import torch-dependent symbols on first access."""
    _torch_symbols = {
        "PairedSuperResolutionDataset",
        "PerceptualLoss",
        "SuperResolutionNet",
        "cross_entropy_manual",
        "select_device",
    }
    _training_symbols = {
        "cosine_schedule",
        "cutout",
        "get_adamw_with_warmup",
        "step_decay_schedule",
        "warmup_schedule",
    }

    if name in _torch_symbols:
        from . import loss_examples
        return getattr(loss_examples, name)
    elif name in _training_symbols:
        from . import training_examples
        return getattr(training_examples, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# For static type checkers - these imports are never executed at runtime
if TYPE_CHECKING:
    from .loss_examples import (
        PairedSuperResolutionDataset,
        PerceptualLoss,
        SuperResolutionNet,
        cross_entropy_manual,
        select_device,
    )
    from .training_examples import (
        cosine_schedule,
        cutout,
        get_adamw_with_warmup,
        step_decay_schedule,
        warmup_schedule,
    )

__all__ = [
    "PairedSuperResolutionDataset",
    "PerceptualLoss",
    "SuperResolutionNet",
    "cosine_schedule",
    "cross_entropy_manual",
    "cutout",
    "get_adamw_with_warmup",
    "select_device",
    "step_decay_schedule",
    "warmup_schedule",
]
