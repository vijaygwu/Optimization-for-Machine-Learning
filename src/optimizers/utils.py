"""
Utility Functions for Optimizers

This module provides utility functions commonly used with optimization
algorithms, including gradient clipping, parameter initialization,
and learning rate utilities.

Author: Optimization for Machine Learning Book
License: MIT
"""

from typing import List, Optional, Tuple, Union, Callable
import numpy as np


def clip_grad_norm_(
    grads: List[np.ndarray],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> float:
    """
    Clip the gradient norm of an iterable of gradients.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        grads: List of gradient arrays.
        max_norm: Maximum allowed norm of gradients.
        norm_type: Type of the norm (default: 2.0 for L2 norm).
            Use float('inf') for max norm.
        error_if_nonfinite: If True, raise error if gradients contain
            inf or nan values.

    Returns:
        Total norm of the gradients (before clipping).

    Example:
        >>> grads = [np.random.randn(10, 5), np.random.randn(5, 2)]
        >>> total_norm = clip_grad_norm_(grads, max_norm=1.0)
        >>> print(f"Gradient norm: {total_norm:.4f}")
    """
    if len(grads) == 0:
        return 0.0

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Filter out None gradients
    valid_grads = [g for g in grads if g is not None]

    if len(valid_grads) == 0:
        return 0.0

    # Compute total norm
    if norm_type == float('inf'):
        # Max norm
        total_norm = max(np.abs(g).max() for g in valid_grads)
    else:
        # Lp norm
        total_norm = 0.0
        for g in valid_grads:
            total_norm += np.sum(np.abs(g) ** norm_type)
        total_norm = total_norm ** (1.0 / norm_type)

    # Check for non-finite values
    if error_if_nonfinite and (np.isnan(total_norm) or np.isinf(total_norm)):
        raise RuntimeError(
            f"Total norm of gradients is {total_norm}. "
            "Gradients contain inf or nan values."
        )

    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for g in grads:
            if g is not None:
                g *= clip_coef

    return float(total_norm)


def clip_grad_value_(grads: List[np.ndarray], clip_value: float) -> None:
    """
    Clip gradient values to a specified range.

    Gradients are modified in-place to be in [-clip_value, clip_value].

    Args:
        grads: List of gradient arrays.
        clip_value: Maximum absolute value for gradient elements.

    Example:
        >>> grads = [np.random.randn(10, 5) * 10]
        >>> clip_grad_value_(grads, clip_value=1.0)
        >>> assert np.all(np.abs(grads[0]) <= 1.0)
    """
    clip_value = float(clip_value)

    for g in grads:
        if g is not None:
            np.clip(g, -clip_value, clip_value, out=g)


def compute_grad_norm(
    grads: List[np.ndarray],
    norm_type: float = 2.0
) -> float:
    """
    Compute the total gradient norm without clipping.

    Args:
        grads: List of gradient arrays.
        norm_type: Type of norm to compute (default: 2.0).

    Returns:
        Total norm of all gradients.
    """
    valid_grads = [g for g in grads if g is not None]

    if len(valid_grads) == 0:
        return 0.0

    if norm_type == float('inf'):
        return max(np.abs(g).max() for g in valid_grads)
    else:
        total_norm = sum(np.sum(np.abs(g) ** norm_type) for g in valid_grads)
        return total_norm ** (1.0 / norm_type)


def warmup_lr(
    step: int,
    warmup_steps: int,
    base_lr: float,
    warmup_init_lr: float = 0.0
) -> float:
    """
    Compute learning rate with linear warmup.

    Args:
        step: Current training step (0-indexed).
        warmup_steps: Number of warmup steps.
        base_lr: Target learning rate after warmup.
        warmup_init_lr: Initial learning rate at step 0.

    Returns:
        Learning rate for the current step.

    Example:
        >>> for step in range(1000):
        ...     lr = warmup_lr(step, warmup_steps=100, base_lr=0.001)
        ...     optimizer.set_lr(lr)
    """
    if warmup_steps <= 0:
        return base_lr

    if step < warmup_steps:
        # Linear warmup
        return warmup_init_lr + (base_lr - warmup_init_lr) * step / warmup_steps
    else:
        return base_lr


def cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
    warmup_steps: int = 0
) -> float:
    """
    Compute learning rate with cosine annealing and optional warmup.

    Args:
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.
        base_lr: Maximum learning rate (after warmup).
        min_lr: Minimum learning rate.
        warmup_steps: Number of warmup steps.

    Returns:
        Learning rate for the current step.
    """
    if step < warmup_steps:
        return warmup_lr(step, warmup_steps, base_lr)

    # Cosine annealing
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)  # Cap at 1.0

    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


def polynomial_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    end_lr: float = 0.0,
    power: float = 1.0,
    warmup_steps: int = 0
) -> float:
    """
    Compute learning rate with polynomial decay.

    Args:
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.
        base_lr: Initial learning rate (after warmup).
        end_lr: Final learning rate.
        power: Power of polynomial decay (1.0 = linear).
        warmup_steps: Number of warmup steps.

    Returns:
        Learning rate for the current step.
    """
    if step < warmup_steps:
        return warmup_lr(step, warmup_steps, base_lr)

    decay_steps = total_steps - warmup_steps
    step_offset = step - warmup_steps

    decay_factor = (1 - step_offset / decay_steps) ** power
    decay_factor = max(0.0, decay_factor)

    return end_lr + (base_lr - end_lr) * decay_factor


def exponential_lr(
    step: int,
    base_lr: float,
    decay_rate: float,
    decay_steps: int = 1,
    staircase: bool = False
) -> float:
    """
    Compute learning rate with exponential decay.

    lr = base_lr * decay_rate ^ (step / decay_steps)

    Args:
        step: Current training step.
        base_lr: Initial learning rate.
        decay_rate: Decay multiplier (e.g., 0.96).
        decay_steps: How often to apply decay (default: 1).
        staircase: If True, decay in discrete steps.

    Returns:
        Learning rate for the current step.
    """
    if staircase:
        exponent = step // decay_steps
    else:
        exponent = step / decay_steps

    return base_lr * (decay_rate ** exponent)


def get_lr_scheduler(
    schedule_type: str,
    base_lr: float,
    total_steps: int,
    **kwargs
) -> Callable[[int], float]:
    """
    Create a learning rate scheduler function.

    Args:
        schedule_type: One of 'constant', 'linear', 'cosine', 'polynomial',
            'exponential'.
        base_lr: Base learning rate.
        total_steps: Total training steps.
        **kwargs: Additional arguments for specific schedulers.

    Returns:
        Function that takes step number and returns learning rate.

    Example:
        >>> scheduler = get_lr_scheduler('cosine', base_lr=0.001,
        ...                              total_steps=10000, warmup_steps=500)
        >>> for step in range(10000):
        ...     lr = scheduler(step)
    """
    warmup_steps = kwargs.get('warmup_steps', 0)

    if schedule_type == 'constant':
        def scheduler(step: int) -> float:
            if step < warmup_steps:
                return warmup_lr(step, warmup_steps, base_lr)
            return base_lr

    elif schedule_type == 'linear':
        end_lr = kwargs.get('end_lr', 0.0)
        def scheduler(step: int) -> float:
            return polynomial_lr(step, total_steps, base_lr, end_lr,
                               power=1.0, warmup_steps=warmup_steps)

    elif schedule_type == 'cosine':
        min_lr = kwargs.get('min_lr', 0.0)
        def scheduler(step: int) -> float:
            return cosine_lr(step, total_steps, base_lr, min_lr, warmup_steps)

    elif schedule_type == 'polynomial':
        end_lr = kwargs.get('end_lr', 0.0)
        power = kwargs.get('power', 2.0)
        def scheduler(step: int) -> float:
            return polynomial_lr(step, total_steps, base_lr, end_lr,
                               power, warmup_steps)

    elif schedule_type == 'exponential':
        decay_rate = kwargs.get('decay_rate', 0.96)
        decay_steps = kwargs.get('decay_steps', 1000)
        staircase = kwargs.get('staircase', False)
        def scheduler(step: int) -> float:
            if step < warmup_steps:
                return warmup_lr(step, warmup_steps, base_lr)
            return exponential_lr(step - warmup_steps, base_lr,
                                 decay_rate, decay_steps, staircase)

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    return scheduler


def initialize_parameters(
    shape: Tuple[int, ...],
    init_type: str = 'xavier_uniform',
    gain: float = 1.0,
    **kwargs
) -> np.ndarray:
    """
    Initialize parameters using various strategies.

    Args:
        shape: Shape of the parameter array.
        init_type: Initialization type. One of:
            - 'zeros': All zeros
            - 'ones': All ones
            - 'constant': Constant value (requires 'value' kwarg)
            - 'uniform': Uniform distribution (requires 'low', 'high')
            - 'normal': Normal distribution (requires 'mean', 'std')
            - 'xavier_uniform': Xavier/Glorot uniform
            - 'xavier_normal': Xavier/Glorot normal
            - 'he_uniform': He/Kaiming uniform (for ReLU)
            - 'he_normal': He/Kaiming normal (for ReLU)
            - 'orthogonal': Orthogonal initialization
        gain: Multiplier for the initialized values.
        **kwargs: Additional arguments for specific initializations.

    Returns:
        Initialized numpy array.

    Example:
        >>> W = initialize_parameters((256, 128), 'he_normal')
        >>> b = initialize_parameters((128,), 'zeros')
    """
    if init_type == 'zeros':
        return np.zeros(shape)

    elif init_type == 'ones':
        return np.ones(shape) * gain

    elif init_type == 'constant':
        value = kwargs.get('value', 0.0)
        return np.full(shape, value)

    elif init_type == 'uniform':
        low = kwargs.get('low', -1.0)
        high = kwargs.get('high', 1.0)
        return np.random.uniform(low, high, shape) * gain

    elif init_type == 'normal':
        mean = kwargs.get('mean', 0.0)
        std = kwargs.get('std', 1.0)
        return np.random.normal(mean, std, shape) * gain

    elif init_type == 'xavier_uniform':
        if len(shape) < 2:
            fan_in = fan_out = shape[0]
        else:
            fan_in = shape[1] if len(shape) > 1 else shape[0]
            fan_out = shape[0]

        bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-bound, bound, shape)

    elif init_type == 'xavier_normal':
        if len(shape) < 2:
            fan_in = fan_out = shape[0]
        else:
            fan_in = shape[1] if len(shape) > 1 else shape[0]
            fan_out = shape[0]

        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, std, shape)

    elif init_type == 'he_uniform':
        if len(shape) < 2:
            fan_in = shape[0]
        else:
            fan_in = shape[1] if len(shape) > 1 else shape[0]

        bound = gain * np.sqrt(6.0 / fan_in)
        return np.random.uniform(-bound, bound, shape)

    elif init_type == 'he_normal':
        if len(shape) < 2:
            fan_in = shape[0]
        else:
            fan_in = shape[1] if len(shape) > 1 else shape[0]

        std = gain * np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std, shape)

    elif init_type == 'orthogonal':
        # Only works for 2D arrays
        if len(shape) != 2:
            raise ValueError("Orthogonal init only works for 2D arrays")

        a = np.random.normal(0, 1, shape)
        u, _, vh = np.linalg.svd(a, full_matrices=False)

        # Pick the one with correct shape
        q = u if u.shape == shape else vh
        return q * gain

    else:
        raise ValueError(f"Unknown initialization type: {init_type}")


def count_parameters(params: List[np.ndarray]) -> int:
    """
    Count total number of parameters.

    Args:
        params: List of parameter arrays.

    Returns:
        Total number of parameters.
    """
    return sum(p.size for p in params if p is not None)


def flatten_grads(grads: List[np.ndarray]) -> np.ndarray:
    """
    Flatten and concatenate all gradients into a single vector.

    Args:
        grads: List of gradient arrays.

    Returns:
        1D array containing all gradient values.
    """
    flat_grads = []
    for g in grads:
        if g is not None:
            flat_grads.append(g.flatten())

    if len(flat_grads) == 0:
        return np.array([])

    return np.concatenate(flat_grads)


def unflatten_grads(
    flat_grads: np.ndarray,
    shapes: List[Tuple[int, ...]]
) -> List[np.ndarray]:
    """
    Unflatten a gradient vector back to original shapes.

    Args:
        flat_grads: 1D array of gradient values.
        shapes: List of shapes to unflatten into.

    Returns:
        List of gradient arrays with original shapes.
    """
    grads = []
    offset = 0

    for shape in shapes:
        size = np.prod(shape)
        grads.append(flat_grads[offset:offset+size].reshape(shape))
        offset += size

    return grads


def check_gradients(
    grads: List[np.ndarray],
    warn_threshold: float = 1e-7,
    error_threshold: float = 1e6
) -> Tuple[bool, str]:
    """
    Check gradients for common issues (vanishing, exploding, NaN).

    Args:
        grads: List of gradient arrays.
        warn_threshold: Warn if max gradient below this value.
        error_threshold: Error if max gradient above this value.

    Returns:
        Tuple of (is_ok, message).
    """
    valid_grads = [g for g in grads if g is not None]

    if len(valid_grads) == 0:
        return False, "No valid gradients"

    # Check for NaN
    for i, g in enumerate(valid_grads):
        if np.any(np.isnan(g)):
            return False, f"NaN detected in gradient {i}"

    # Check for Inf
    for i, g in enumerate(valid_grads):
        if np.any(np.isinf(g)):
            return False, f"Inf detected in gradient {i}"

    # Compute statistics
    max_abs = max(np.abs(g).max() for g in valid_grads)
    min_abs = min(np.abs(g[g != 0]).min() if np.any(g != 0) else float('inf')
                 for g in valid_grads)

    # Check for exploding gradients
    if max_abs > error_threshold:
        return False, f"Exploding gradients: max={max_abs:.2e}"

    # Check for vanishing gradients
    if max_abs < warn_threshold:
        return True, f"Warning: Possibly vanishing gradients: max={max_abs:.2e}"

    return True, f"Gradients OK: max={max_abs:.2e}, min_nonzero={min_abs:.2e}"
