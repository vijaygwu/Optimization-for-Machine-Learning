"""Offline smoke tests for the mixed runtime stack used across Book 2."""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import torch
import torchvision
from scipy.special import logsumexp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def test_runtime_dependencies_import() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("torch", "torchvision", "sklearn", "scipy", "numpy"):
        module = importlib.import_module(module_name)
        versions[module_name] = getattr(module, "__version__", "unknown")
    return versions


def test_torchvision_forward_pass() -> None:
    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None)
    model.eval()

    inputs = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 1000)
    assert torch.isfinite(outputs).all()


def test_capstone_preprocessing_pipeline_offline() -> None:
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0

    # Exercise the string-label path that the capstone handles for OpenML data.
    y = LabelEncoder().fit_transform(digits.target.astype(str))

    num_classes = len(np.unique(y))
    y_onehot = np.zeros((len(y), num_classes), dtype=np.float32)
    y_onehot[np.arange(len(y)), y] = 1.0

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_onehot, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1111,
        random_state=42,
        stratify=np.argmax(y_temp, axis=1),
    )

    total = len(X)
    assert len(X_train) + len(X_val) + len(X_test) == total
    assert abs(len(X_train) / total - 0.8) < 0.02
    assert abs(len(X_val) / total - 0.1) < 0.02
    assert abs(len(X_test) / total - 0.1) < 0.02
    assert y_train.shape[1] == num_classes
    assert np.array_equal(np.unique(np.argmax(y_train, axis=1)), np.arange(num_classes))


def test_logit_space_loss_matches_scipy_reference() -> None:
    logits = np.array([[2.0, 0.5, -1.0], [-0.3, 1.2, 0.1]], dtype=np.float64)

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    manual_log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    scipy_log_probs = logits - logsumexp(logits, axis=1, keepdims=True)

    np.testing.assert_allclose(manual_log_probs, scipy_log_probs, atol=1e-12, rtol=1e-12)


def main() -> None:
    print("Running Book 2 runtime smoke tests...")
    versions = test_runtime_dependencies_import()
    print(
        "  - runtime imports succeeded for "
        + ", ".join(f"{name}={version}" for name, version in versions.items())
    )
    test_torchvision_forward_pass()
    print("  - torchvision ResNet forward pass works in the pinned runtime")
    test_capstone_preprocessing_pipeline_offline()
    print("  - capstone-style sklearn preprocessing and stratified splits work offline")
    test_logit_space_loss_matches_scipy_reference()
    print("  - scipy logsumexp matches the capstone logit-space loss math")
    print("\nAll Book 2 runtime smoke tests passed.")


if __name__ == "__main__":
    main()
