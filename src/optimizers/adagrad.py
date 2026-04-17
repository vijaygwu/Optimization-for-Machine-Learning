"""
AdaGrad Optimizer

AdaGrad (Adaptive Gradient) adapts the learning rate for each parameter
based on the historical sum of squared gradients.

Author: Optimization for Machine Learning Book
License: MIT
"""

from typing import Dict, List, Optional, Any, Iterator
import numpy as np

from .base import Optimizer


class Adagrad(Optimizer):
    """
    Adagrad optimizer (Adaptive Gradient).

    Adagrad adapts the learning rate for each parameter by scaling
    inversely with the square root of the sum of all historical
    squared gradients.

    Update rule:
        G_t = G_{t-1} + g_t^2                    # Accumulate squared gradients
        w_{t+1} = w_t - lr * g_t / (sqrt(G_t) + eps)

    Key properties:
    - Parameters with large gradients get smaller learning rates
    - Parameters with small gradients get larger learning rates
    - Learning rate naturally decays over time (can be a limitation)
    - Works well for sparse features (e.g., NLP embeddings)

    Attributes:
        lr (float): Learning rate (default: 0.01).
        lr_decay (float): Learning rate decay (default: 0).
        eps (float): Term for numerical stability (default: 1e-10).
        initial_accumulator_value (float): Initial value for accumulator.
        weight_decay (float): L2 regularization coefficient.

    Example:
        >>> params = [np.random.randn(10, 5), np.random.randn(5, 2)]
        >>> optimizer = Adagrad(params, lr=0.01)
        >>> for epoch in range(100):
        ...     grads = compute_gradients(params)
        ...     optimizer.step(grads)

    Limitations:
        - Accumulated squared gradients grow monotonically, causing
          learning rate to continually decrease and eventually become
          too small for effective learning
        - Not ideal for non-convex problems like deep neural networks

    Reference:
        Duchi et al., "Adaptive Subgradient Methods for Online Learning
        and Stochastic Optimization", JMLR 2011
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 0.01,
        lr_decay: float = 0.0,
        eps: float = 1e-10,
        initial_accumulator_value: float = 0.0,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adagrad optimizer.

        Args:
            params: Iterator of parameters to optimize.
            lr: Learning rate (default: 0.01).
            lr_decay: Learning rate decay applied at each step (default: 0).
                Effective lr = lr / (1 + step * lr_decay)
            eps: Term added to denominator for numerical stability
                (default: 1e-10).
            initial_accumulator_value: Initial value for gradient accumulator
                (default: 0). Setting this to a small positive value can
                help stabilize initial updates.
            weight_decay: L2 regularization coefficient (default: 0).

        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_decay < 0.0:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if initial_accumulator_value < 0.0:
            raise ValueError(f"Invalid initial_accumulator_value: {initial_accumulator_value}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            'lr': lr,
            'lr_decay': lr_decay,
            'eps': eps,
            'initial_accumulator_value': initial_accumulator_value,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def _init_state(
        self,
        param: np.ndarray,
        param_id: int,
        group: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Initialize optimizer state for a parameter."""
        group = self.defaults if group is None else group
        initial_value = group['initial_accumulator_value']
        return {
            'step': 0,
            'sum': np.full_like(param, initial_value),  # Accumulated squared gradients
        }

    def step(self, grads: Optional[List[np.ndarray]] = None) -> None:
        """
        Perform a single optimization step.

        Args:
            grads: List of gradients corresponding to parameters.
        """
        if grads is None:
            raise ValueError("Adagrad requires gradients to be passed explicitly")

        grad_idx = 0
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            lr_decay = group['lr_decay']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for param_idx, param in enumerate(group['params']):
                if grad_idx >= len(grads):
                    raise ValueError("Not enough gradients provided")

                grad = grads[grad_idx]
                grad_idx += 1

                if grad is None:
                    continue

                # Get or initialize state
                param_id = self._get_param_id(param, group_idx, param_idx)
                if param_id not in self.state:
                    self.state[param_id] = self._init_state(param, param_id, group)
                state = self.state[param_id]
                state['step'] += 1
                step = state['step']

                # Add weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad + weight_decay * param

                # Apply learning rate decay
                effective_lr = lr / (1 + (step - 1) * lr_decay)

                # Accumulate squared gradients
                sum_sq = state['sum']
                sum_sq += grad ** 2

                # Update: w = w - lr * g / (sqrt(sum) + eps)
                std = np.sqrt(sum_sq) + eps
                param -= effective_lr * grad / std


class AdagradSparse(Adagrad):
    """
    Adagrad variant optimized for sparse gradients.

    This version only updates the accumulator for non-zero gradient
    entries, which is more efficient for sparse data like word embeddings.

    Note: This is a simplified version. For true sparse optimization,
    you'd want to use sparse array representations.
    """

    def step(self, grads: Optional[List[np.ndarray]] = None) -> None:
        """Perform optimization step, optimized for sparse gradients."""
        if grads is None:
            raise ValueError("AdagradSparse requires gradients")

        grad_idx = 0
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            lr_decay = group['lr_decay']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for param_idx, param in enumerate(group['params']):
                if grad_idx >= len(grads):
                    raise ValueError("Not enough gradients provided")

                grad = grads[grad_idx]
                grad_idx += 1

                if grad is None:
                    continue

                param_id = self._get_param_id(param, group_idx, param_idx)
                if param_id not in self.state:
                    self.state[param_id] = self._init_state(param, param_id, group)
                state = self.state[param_id]
                state['step'] += 1
                step = state['step']

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                effective_lr = lr / (1 + (step - 1) * lr_decay)

                # Find non-zero gradient indices (sparse optimization)
                nonzero_mask = grad != 0

                # Only update accumulator for non-zero gradients
                sum_sq = state['sum']
                sum_sq[nonzero_mask] += grad[nonzero_mask] ** 2

                # Update only parameters with non-zero gradients
                std = np.sqrt(sum_sq) + eps
                param[nonzero_mask] -= effective_lr * grad[nonzero_mask] / std[nonzero_mask]


class Adadelta(Optimizer):
    """
    Adadelta optimizer.

    Adadelta is an extension of Adagrad that seeks to reduce its
    aggressive, monotonically decreasing learning rate. It uses a
    window of accumulated past gradients and also accumulates past
    parameter updates.

    Update rules:
        g_acc_t = rho * g_acc_{t-1} + (1 - rho) * g_t^2
        update_t = sqrt(delta_acc_{t-1} + eps) / sqrt(g_acc_t + eps) * g_t
        delta_acc_t = rho * delta_acc_{t-1} + (1 - rho) * update_t^2
        w_{t+1} = w_t - update_t

    Note: Adadelta doesn't require a learning rate!

    Attributes:
        rho (float): Decay rate for moving averages (default: 0.9).
        eps (float): Term for numerical stability (default: 1e-6).
        lr (float): Learning rate multiplier (default: 1.0).
        weight_decay (float): L2 regularization coefficient.

    Reference:
        Zeiler, "ADADELTA: An Adaptive Learning Rate Method", 2012
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adadelta optimizer.

        Args:
            params: Iterator of parameters to optimize.
            lr: Learning rate multiplier (default: 1.0). Unlike other
                optimizers, this just scales the update.
            rho: Decay rate for running averages (default: 0.9).
            eps: Term for numerical stability (default: 1e-6).
            weight_decay: L2 regularization coefficient (default: 0).
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            'lr': lr,
            'rho': rho,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def _init_state(self, param: np.ndarray, param_id: int) -> Dict[str, Any]:
        """Initialize optimizer state."""
        return {
            'step': 0,
            'square_avg': np.zeros_like(param),    # E[g^2]
            'acc_delta': np.zeros_like(param),      # E[delta^2]
        }

    def step(self, grads: Optional[List[np.ndarray]] = None) -> None:
        """Perform optimization step."""
        if grads is None:
            raise ValueError("Adadelta requires gradients")

        grad_idx = 0
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            rho = group['rho']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for param_idx, param in enumerate(group['params']):
                if grad_idx >= len(grads):
                    raise ValueError("Not enough gradients provided")

                grad = grads[grad_idx]
                grad_idx += 1

                if grad is None:
                    continue

                param_id = self._get_param_id(param, group_idx, param_idx)
                if param_id not in self.state:
                    self.state[param_id] = self._init_state(param, param_id)
                state = self.state[param_id]
                state['step'] += 1

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                square_avg = state['square_avg']
                acc_delta = state['acc_delta']

                # Update running average of squared gradients
                # E[g^2]_t = rho * E[g^2]_{t-1} + (1-rho) * g_t^2
                square_avg *= rho
                square_avg += (1 - rho) * (grad ** 2)

                # Compute update: sqrt(E[delta^2] + eps) / sqrt(E[g^2] + eps) * g
                std = np.sqrt(square_avg + eps)
                delta = np.sqrt(acc_delta + eps) / std * grad

                # Update running average of squared updates
                # E[delta^2]_t = rho * E[delta^2]_{t-1} + (1-rho) * delta_t^2
                acc_delta *= rho
                acc_delta += (1 - rho) * (delta ** 2)

                # Apply update
                param -= lr * delta
