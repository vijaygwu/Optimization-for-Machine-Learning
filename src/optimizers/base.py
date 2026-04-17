"""
Base Optimizer Class

This module provides the abstract base class for all optimization algorithms.
It follows PyTorch's optimizer API design for familiarity and compatibility.

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
    all optimization algorithms. It follows the PyTorch optimizer API design.

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
        self._param_to_key: Dict[int, str] = {}  # Maps id(param) to stable key
        self._next_param_idx = 0  # Counter for stable parameter keys

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
            grads: Optional list of gradients. If None, assumes gradients
                are stored in parameter.grad attributes (for torch-like usage).
        """
        raise NotImplementedError

    def zero_grad(self) -> None:
        """
        Reset all gradients to zero.

        This is typically called before computing new gradients to prevent
        gradient accumulation across iterations.
        """
        # For numpy arrays, we don't have .grad attributes
        # This method is provided for API compatibility
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

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load optimizer state.

        Args:
            state_dict: Optimizer state dict from state_dict().
        """
        # Load state - keys are now stable strings, not memory addresses
        self.state = {}
        for key, state in state_dict['state'].items():
            # Support both old int-based keys (for backwards compatibility) and new string keys
            state_key = str(key) if not isinstance(key, str) else key
            self.state[state_key] = {k: v.copy() if isinstance(v, np.ndarray) else v
                                     for k, v in state.items()}

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

    def _get_param_key(self, param: np.ndarray) -> str:
        """
        Get a stable, deterministic key for a parameter array.

        Unlike id(param) which changes across sessions, this returns
        a stable string key based on parameter index that survives
        checkpoint save/restore cycles.
        """
        param_id = id(param)
        if param_id not in self._param_to_key:
            # Assign a new stable key based on the order parameters were first seen
            self._param_to_key[param_id] = f"param_{self._next_param_idx}"
            self._next_param_idx += 1
        return self._param_to_key[param_id]

    def _get_param_id(self, param: np.ndarray) -> str:
        """Get unique ID for a parameter array (deprecated, use _get_param_key)."""
        return self._get_param_key(param)

    def _init_state(self, param: np.ndarray, param_id: int) -> Dict[str, Any]:
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
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Step once if starting fresh
        if last_epoch == -1:
            self.last_epoch = 0
            self.step()

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
            step_size: Period of learning rate decay.
            gamma: Multiplicative factor of learning rate decay.
            last_epoch: Index of last epoch.
        """
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
            T_max: Maximum number of iterations.
            eta_min: Minimum learning rate.
            last_epoch: Index of last epoch.
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute current learning rates."""
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
