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
        warmup_init_lr: Warmup floor interpolated toward base_lr.
            With zero-indexed warmup, step 0 already applies the first
            warmup increment and step warmup_steps - 1 reaches base_lr.

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
        # Zero-indexed linear warmup: step 0 is the first increment and
        # step warmup_steps - 1 reaches base_lr exactly.
        return warmup_init_lr + (base_lr - warmup_init_lr) * (step + 1) / warmup_steps
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

    At step == total_steps - 1 (the final step), returns exactly min_lr.

    Args:
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.
        base_lr: Maximum learning rate (after warmup).
        min_lr: Minimum learning rate (reached at final step).
        warmup_steps: Number of warmup steps.

    Returns:
        Learning rate for the current step.
    """
    if step < warmup_steps:
        return warmup_lr(step, warmup_steps, base_lr)

    # Beyond total_steps: clamp to min_lr
    if step >= total_steps:
        return min_lr

    # Cosine annealing: progress goes from 0 to 1 over decay phase
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return min_lr

    progress = (step - warmup_steps) / (decay_steps - 1)
    progress = min(1.0, progress)

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

    Raises:
        ValueError: If total_steps <= warmup_steps (no decay steps).
    """
    if step < warmup_steps:
        return warmup_lr(step, warmup_steps, base_lr)

    decay_steps = total_steps - warmup_steps

    # Edge case: no decay steps available
    if decay_steps <= 0:
        raise ValueError(
            f"total_steps ({total_steps}) must be greater than warmup_steps ({warmup_steps}) "
            "to allow for polynomial decay."
        )

    step_offset = step - warmup_steps

    # Edge case: step beyond total_steps - return end_lr
    if step_offset >= decay_steps:
        return end_lr

    decay_factor = (1 - step_offset / decay_steps) ** power
    # decay_factor is guaranteed non-negative due to the check above

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

    Raises:
        ValueError: If decay_steps <= 0 or decay_rate <= 0.
    """
    if decay_steps <= 0:
        raise ValueError(f"decay_steps must be positive, got {decay_steps}")
    if decay_rate <= 0:
        raise ValueError(f"decay_rate must be positive, got {decay_rate}")

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


def _compute_fans(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Compute fan_in and fan_out for weight initialization.

    Handles both linear layers (2D) and convolutional layers (4D).

    Args:
        shape: Shape of the weight tensor.
            - 1D: (features,) -> fan_in = fan_out = features
            - 2D: (out_features, in_features) -> fan_in = in_features, fan_out = out_features
            - 4D: (out_channels, in_channels, kH, kW) -> fan_in = in_channels * kH * kW

    Returns:
        Tuple of (fan_in, fan_out).
    """
    if len(shape) < 2:
        # 1D tensor (e.g., bias)
        return shape[0], shape[0]
    elif len(shape) == 2:
        # Linear layer: (out_features, in_features)
        fan_out, fan_in = shape[0], shape[1]
    else:
        # Convolutional layer: (out_channels, in_channels, *kernel_size)
        # Receptive field size = product of spatial dimensions
        receptive_field_size = int(np.prod(shape[2:]))
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size

    return fan_in, fan_out


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
        fan_in, fan_out = _compute_fans(shape)
        bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-bound, bound, shape)

    elif init_type == 'xavier_normal':
        fan_in, fan_out = _compute_fans(shape)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, std, shape)

    elif init_type == 'he_uniform':
        fan_in, _ = _compute_fans(shape)
        bound = gain * np.sqrt(6.0 / fan_in)
        return np.random.uniform(-bound, bound, shape)

    elif init_type == 'he_normal':
        fan_in, _ = _compute_fans(shape)
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

    Raises:
        ValueError: If flat_grads length doesn't match the total size of shapes.
    """
    expected_size = sum(int(np.prod(shape)) for shape in shapes)
    if flat_grads.size != expected_size:
        raise ValueError(
            f"flat_grads has {flat_grads.size} elements, but shapes require {expected_size}"
        )

    grads = []
    offset = 0

    for shape in shapes:
        size = int(np.prod(shape))
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


# =============================================================================
# Learning Rate Finder (PyTorch)
# =============================================================================

def lr_finder(model, train_loader, optimizer_class=None, criterion=None,
              start_lr: float = 1e-8, end_lr: float = 10, num_steps: int = 100):
    """
    Learning rate range test to find optimal learning rate.

    NOTE: This function saves and restores model state, so the model
    is unchanged after the test completes.

    Args:
        model: PyTorch model to test
        train_loader: DataLoader for training data
        optimizer_class: Optimizer class (default: torch.optim.SGD)
        criterion: Loss function (default: CrossEntropyLoss)
        start_lr: Starting learning rate (very small)
        end_lr: Ending learning rate (very large)
        num_steps: Number of steps to test

    Returns:
        lrs: List of learning rates tested
        losses: List of corresponding losses

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If model parameters cannot be accessed

    Example:
        >>> lrs, losses = lr_finder(model, train_loader)
        >>> plt.plot(lrs, losses)
        >>> plt.xscale('log')
        >>> plt.xlabel('Learning Rate')
        >>> plt.ylabel('Loss')
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("lr_finder requires PyTorch. Install with: pip install torch")

    if optimizer_class is None:
        optimizer_class = torch.optim.SGD

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Validate num_steps
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")

    # Validate learning rate range
    if start_lr <= 0 or end_lr <= 0 or start_lr >= end_lr:
        raise ValueError(f"Invalid LR range: [{start_lr}, {end_lr}]")

    # Check model has parameters
    try:
        model_params = list(model.parameters())
        if not model_params:
            raise ValueError("Model has no learnable parameters")
    except Exception as e:
        raise RuntimeError(f"Cannot access model parameters: {e}")

    # Guard against empty loader (handles both map-style and iterable datasets)
    try:
        loader_len = len(train_loader)
        if loader_len == 0:
            raise ValueError("train_loader is empty")
    except TypeError:
        # IterableDataset / streaming loader: __len__ not supported, skip check
        pass

    # CRITICAL: Save model state AND training mode before mutation
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    was_training = model.training

    try:
        # Create optimizer with initial learning rate
        optimizer = optimizer_class(model.parameters(), lr=start_lr)
        lr_mult = (end_lr / start_lr) ** (1 / (num_steps - 1))

        lrs, losses = [], []
        best_loss = float('inf')
        model.train()
        data_iter = iter(train_loader)

        # Infer device from model parameters
        device = next(model.parameters()).device

        for step in range(num_steps):
            # Get batch (cycle through data if needed)
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    raise ValueError("train_loader produced no data")

            # Move batch to model's device (critical for CUDA/MPS)
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Record learning rate and loss
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            losses.append(loss.item())

            # Stop if loss explodes
            if loss.item() > 4 * best_loss:
                break
            best_loss = min(best_loss, loss.item())

            # Backward pass and update
            loss.backward()
            optimizer.step()

            # Increase learning rate exponentially
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_mult

    finally:
        # CRITICAL: Restore model state and training mode after test
        model.load_state_dict(initial_state)
        model.train(was_training)

    return lrs, losses


# =============================================================================
# Random Hyperparameter Search
# =============================================================================

def random_hyperparameter_search(
    train_fn: Callable,
    n_trials: int = 20,
    lr_range: Tuple[float, float] = (1e-5, 1e-2),
    wd_range: Tuple[float, float] = (1e-6, 1e-2),
    beta1_range: Tuple[float, float] = (0.85, 0.95),
    beta2_range: Tuple[float, float] = (0.99, 0.9999),
) -> Tuple[dict, float]:
    """
    Random hyperparameter search for optimizer settings.

    Uses log-uniform sampling for learning rate and weight decay,
    and uniform sampling for beta1 and beta2.

    Args:
        train_fn: Function that takes learning_rate, weight_decay, beta1, beta2
                  as keyword arguments and returns validation loss.
        n_trials: Number of random configurations to try.
        lr_range: (min, max) range for learning rate (log-uniform).
        wd_range: (min, max) range for weight decay (log-uniform).
        beta1_range: (min, max) range for beta1 (uniform).
        beta2_range: (min, max) range for beta2 (uniform).

    Returns:
        Tuple of (best_params dict, best_loss).

    Raises:
        TypeError: If train_fn does not return a numeric value.

    Example:
        >>> def train_fn(learning_rate, weight_decay, beta1, beta2):
        ...     # Train model and return validation loss
        ...     return val_loss
        >>> best_params, best_loss = random_hyperparameter_search(train_fn, n_trials=50)
    """
    try:
        from scipy.stats import loguniform
    except ImportError:
        raise ImportError("random_hyperparameter_search requires scipy. "
                         "Install with: pip install scipy")

    best_loss, best_params = float('inf'), None

    for trial in range(n_trials):
        params = {
            'learning_rate': loguniform.rvs(lr_range[0], lr_range[1]),
            'weight_decay': loguniform.rvs(wd_range[0], wd_range[1]),
            'beta1': np.random.uniform(beta1_range[0], beta1_range[1]),
            'beta2': np.random.uniform(beta2_range[0], beta2_range[1]),
        }

        val_loss = train_fn(**params)

        if not isinstance(val_loss, (int, float)):
            raise TypeError(f"train_fn must return a number, got {type(val_loss)}")

        if val_loss < best_loss:
            best_loss, best_params = val_loss, params

    return best_params, best_loss
