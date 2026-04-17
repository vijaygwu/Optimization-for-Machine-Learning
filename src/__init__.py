"""Companion code exports for the Book 2 optimization examples.

This module uses lazy imports to avoid pulling in torch-dependent code
when only lightweight utilities are needed. Optimizer-only users can
import without torch installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy imports for ALL external-dependency modules
_LAZY_MODULES = {
    # torch-dependent (loss_examples)
    "PairedSuperResolutionDataset": "loss_examples",
    "PerceptualLoss": "loss_examples",
    "SuperResolutionNet": "loss_examples",
    "cross_entropy_manual": "loss_examples",
    "select_device": "loss_examples",
    # torch-dependent (training_examples)
    "cosine_schedule": "training_examples",
    "cutout": "training_examples",
    "get_adamw_with_warmup": "training_examples",
    "mixup": "training_examples",
    "step_decay_schedule": "training_examples",
    "warmup_cosine_schedule": "training_examples",
    "warmup_schedule": "training_examples",
}

def __getattr__(name: str):
    """Lazy import for torch-dependent symbols."""
    if name in _LAZY_MODULES:
        module_name = _LAZY_MODULES[name]
        import importlib
        module = importlib.import_module(f".{module_name}", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
        mixup,
        step_decay_schedule,
        warmup_cosine_schedule,
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
    "mixup",
    "select_device",
    "step_decay_schedule",
    "warmup_cosine_schedule",
    "warmup_schedule",
]
