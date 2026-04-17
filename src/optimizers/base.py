"""
Base Optimizer Class

This module provides the abstract base class for all optimization algorithms.
It keeps PyTorch-like parameter groups and state dictionaries for familiarity,
but these NumPy implementations require gradients to be passed explicitly.

Author: Optimization for Machine Learning Book
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Iterator, Optional, Any, Tuple
import numpy as np


class Optimizer(ABC):
    """
    Abstract base class for all optimizers.

    This class provides the interface and common functionality shared by
    all optimization algorithms. It uses a PyTorch-inspired structure, but
    concrete subclasses expect explicit gradient lists on every step.

    Attributes:
        param_groups (List[Dict]): List of parameter groups, each containing
            parameters and their associated hyperparameters.
        defaults (Dict): Default hyperparameters for the optimizer.
        state (Dict): State dictionary for storing optimizer state per parameter.

    Example:
        >>> params = [np.random.randn(10, 5), np.random.randn(5, 2)]
        >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
        >>> for epoch in range(100):
        ...     grads = compute_gradients(params)
        ...     optimizer.step(grads)
    """

    def __init__(self, params: Iterator[np.ndarray], defaults: Dict[str, Any]):
        """
        Initialize the optimizer.

        Args:
            params: Iterator of parameters to optimize. Can be:
                - A list/iterator of numpy arrays
                - A list of dicts with 'params' key and optional hyperparameters
            defaults: Default hyperparameters (lr, weight_decay, etc.)
        """
        self.defaults = defaults
        self.state: Dict[str, Dict[str, Any]] = {}
        self.param_groups: List[Dict[str, Any]] = []

        # Convert params to list for consistent handling
        param_groups = list(params) if not isinstance(params, list) else params

        if len(param_groups) == 0:
            raise ValueError("Optimizer got an empty parameter list")

        # Handle both list of arrays and list of dicts
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        Add a parameter group to the optimizer.

        Args:
            param_group: Dict with 'params' key containing parameters
                and optional hyperparameter overrides.
        """
        if 'params' not in param_group:
            raise ValueError("Parameter group must have 'params' key")

        params = list(param_group['params']) if not isinstance(
            param_group['params'], list) else param_group['params']

        if len(params) == 0:
            raise ValueError("Parameter group contains no parameters")

        # Set defaults for missing hyperparameters
        for key, value in self.defaults.items():
            param_group.setdefault(key, value)

        param_group['params'] = params
        self.param_groups.append(param_group)

    @abstractmethod
    def step(self, grads: Optional[List[np.ndarray]] = None) -> None:
        """
        Perform a single optimization step.

        Args:
            grads: Gradient list matching the optimizer parameter order.
                Passing ``None`` is unsupported in these NumPy optimizers;
                concrete subclasses raise ``ValueError`` when gradients are
                omitted.
        """
        raise NotImplementedError

    def zero_grad(self) -> None:
        """
        Reset all gradients to zero.

        This is typically called before computing new gradients to prevent
        gradient accumulation across iterations.
        """
        # NumPy parameters do not store .grad attributes.
        # This method is provided for API familiarity only.
        pass

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state of the optimizer as a dict.

        Returns:
            Dict containing optimizer state including param_groups and state.
        """
        # Pack the state
        packed_state = {}
        for idx, state in self.state.items():
            packed_state[idx] = {k: v.copy() if isinstance(v, np.ndarray) else v
                                for k, v in state.items()}

        return {
            'state': packed_state,
            'param_groups': [{k: v for k, v in group.items() if k != 'params'}
                           for group in self.param_groups]
        }

    def load_state_dict(self, state_dict: Dict[str, Any], validate_shapes: bool = True) -> None:
        """
        Load optimizer state.

        Args:
            state_dict: Optimizer state dict from state_dict().
            validate_shapes: If True, validate that the number of parameters and
                            their shapes match. Recommended to catch silent mismatches.

        Raises:
            ValueError: If param group count doesn't match, or if validate_shapes=True
                        and parameter count or shapes don't match.
        """
        # Validate param groups match
        if len(state_dict['param_groups']) != len(self.param_groups):
            raise ValueError(
                f"Param group count mismatch: state_dict has {len(state_dict['param_groups'])} "
                f"param groups, but optimizer has {len(self.param_groups)}"
            )

        # Validate parameter count per group
        saved_state_count = len(state_dict['state'])
        current_param_count = sum(len(group['params']) for group in self.param_groups)

        if validate_shapes and saved_state_count != current_param_count:
            raise ValueError(
                f"Parameter count mismatch: saved state has {saved_state_count} parameters, "
                f"but optimizer has {current_param_count}. "
                f"Use validate_shapes=False to skip this check."
            )

        # Validate shapes match by extracting shape info from state keys
        if validate_shapes:
            saved_shapes = []
            for key in sorted(state_dict['state'].keys()):
                # Extract shape from key like 'group0:param0:shape10x5'
                if ':shape' in key:
                    shape_str = key.split(':shape')[-1]
                    shape = tuple(int(d) for d in shape_str.split('x'))
                    saved_shapes.append(shape)

            current_shapes = []
            for group in self.param_groups:
                for param in group['params']:
                    if hasattr(param, 'shape'):
                        current_shapes.append(param.shape)

            if saved_shapes and current_shapes and saved_shapes != current_shapes:
                raise ValueError(
                    f"Parameter shape mismatch: saved shapes {saved_shapes} != "
                    f"current shapes {current_shapes}. "
                    f"Use validate_shapes=False to skip this check."
                )

        # Load state
        self.state = {}
        for idx, state in state_dict['state'].items():
            param_id = str(idx)
            self.state[param_id] = {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in state.items()
            }

        # Load param groups (except params themselves)
        for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
            for key, value in saved_group.items():
                if key != 'params':
                    group[key] = value

    def get_lr(self) -> List[float]:
        """
        Get current learning rates for all parameter groups.

        Returns:
            List of learning rates.
        """
        return [group['lr'] for group in self.param_groups]

    def set_lr(self, lr: float, group_idx: Optional[int] = None) -> None:
        """
        Set learning rate for parameter group(s).

        Args:
            lr: New learning rate value.
            group_idx: If specified, only update this group. Otherwise update all.
        """
        if group_idx is not None:
            self.param_groups[group_idx]['lr'] = lr
        else:
            for group in self.param_groups:
                group['lr'] = lr

    def _get_param_id(self, param: np.ndarray, group_idx: int, param_idx: int) -> str:
        """
        Get unique ID for a parameter array.

        Uses a deterministic hash based on position in param_groups rather than
        id(param), which changes across sessions and would break checkpoint resume.

        Args:
            param: The parameter array.
            group_idx: Index of the parameter group.
            param_idx: Index of the parameter within its group.

        Returns:
            Deterministic string ID for this parameter.
        """
        shape_key = "x".join(str(dim) for dim in param.shape)
        return f"group{group_idx}:param{param_idx}:shape{shape_key}"

    def _init_state(self, param: np.ndarray, param_id: str) -> Dict[str, Any]:
        """
        Initialize state for a parameter.

        Override this in subclasses to initialize algorithm-specific state.

        Args:
            param: The parameter array.
            param_id: Unique ID for the parameter.

        Returns:
            Initial state dict for this parameter.
        """
        return {}

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += f'Parameter Group {i}\n'
            for key, value in group.items():
                if key != 'params':
                    format_string += f'    {key}: {value}\n'
        format_string += ')'
        return format_string


class LRScheduler(ABC):
    """
    Abstract base class for learning rate schedulers.

    Learning rate schedulers adjust the learning rate during training
    according to various strategies (step decay, cosine annealing, etc.).
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        """
        Initialize the scheduler.

        Args:
            optimizer: The optimizer to modify.
            last_epoch: Index of last epoch. Use -1 to start from beginning.
                        Note: Unlike PyTorch, we do NOT call step() on init.
                        This avoids modifying the LR before the first training step.
                        Call scheduler.step() AFTER each epoch's training.
        """
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Initialize last_epoch to -1 (before any training)
        # The first call to step() will increment to 0 and set the initial LR
        # NOTE: We intentionally do NOT call step() here to avoid changing LR
        # before training starts. This differs from PyTorch's behavior which
        # calls step() on init, causing a confusing initial LR change.
        if last_epoch == -1:
            self.last_epoch = -1
        else:
            self.last_epoch = last_epoch

    @abstractmethod
    def get_lr(self) -> List[float]:
        """
        Compute learning rates for the current epoch.

        Returns:
            List of learning rates for each parameter group.
        """
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> None:
        """
        Update learning rates.

        Args:
            epoch: If provided, use this epoch number. Otherwise increment.
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state as dict."""
        return {
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from dict."""
        self.last_epoch = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']


class StepLR(LRScheduler):
    """
    Decay learning rate by gamma every step_size epochs.

    lr = base_lr * gamma^(epoch // step_size)

    Example:
        >>> optimizer = SGD(params, lr=0.1)
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(self, optimizer: Optimizer, step_size: int,
                 gamma: float = 0.1, last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer.
            step_size: Period of learning rate decay. Must be > 0.
            gamma: Multiplicative factor of learning rate decay.
            last_epoch: Index of last epoch.

        Raises:
            ValueError: If step_size <= 0.
        """
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute current learning rates."""
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate schedule.

    lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2

    Example:
        >>> optimizer = SGD(params, lr=0.1)
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=100)
    """

    def __init__(self, optimizer: Optimizer, T_max: int,
                 eta_min: float = 0.0, last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer.
            T_max: Maximum number of iterations. Must be > 0.
            eta_min: Minimum learning rate.
            last_epoch: Index of last epoch.

        Raises:
            ValueError: If T_max <= 0.
        """
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute current learning rates."""
        # Guard against division by zero (T_max validated in __init__)
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
