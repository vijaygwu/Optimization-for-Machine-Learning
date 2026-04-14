"""
RMSprop Optimizer

RMSprop (Root Mean Square Propagation) maintains a moving average of
squared gradients to adapt the learning rate for each parameter.

Author: Optimization for Machine Learning Book
License: MIT
"""

from typing import Dict, List, Optional, Any, Iterator
import numpy as np

from .base import Optimizer


class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    RMSprop adapts learning rates by dividing by a running average of
    squared gradients. This helps deal with the diminishing learning
    rates problem of AdaGrad.

    Update rules:
        v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
        w_{t+1} = w_t - lr * g_t / (sqrt(v_t) + eps)

    With momentum:
        v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
        buf_t = momentum * buf_{t-1} + g_t / (sqrt(v_t) + eps)
        w_{t+1} = w_t - lr * buf_t

    With centered gradient (reduces variance):
        g_avg_t = alpha * g_avg_{t-1} + (1 - alpha) * g_t
        v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
        v_centered_t = v_t - g_avg_t^2
        w_{t+1} = w_t - lr * g_t / (sqrt(v_centered_t) + eps)

    Attributes:
        lr (float): Learning rate (default: 0.01).
        alpha (float): Smoothing constant / decay rate (default: 0.99).
        eps (float): Term for numerical stability (default: 1e-8).
        weight_decay (float): L2 regularization coefficient (default: 0).
        momentum (float): Momentum factor (default: 0).
        centered (bool): Whether to use centered RMSprop (default: False).

    Example:
        >>> params = [np.random.randn(10, 5), np.random.randn(5, 2)]
        >>> optimizer = RMSprop(params, lr=0.01, alpha=0.99)
        >>> for epoch in range(100):
        ...     grads = compute_gradients(params)
        ...     optimizer.step(grads)

    References:
        - Hinton, "Neural Networks for Machine Learning", Coursera Lecture 6e
        - Graves, "Generating Sequences With Recurrent Neural Networks", 2013
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False
    ):
        """
        Initialize RMSprop optimizer.

        Args:
            params: Iterator of parameters to optimize.
            lr: Learning rate (default: 0.01).
            alpha: Smoothing constant / decay rate for running average of
                squared gradients (default: 0.99).
            eps: Term added to denominator for numerical stability (default: 1e-8).
            weight_decay: L2 regularization coefficient (default: 0).
            momentum: Momentum factor (default: 0).
            centered: If True, compute centered RMSprop by maintaining
                running average of gradients (reduces variance).

        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = {
            'lr': lr,
            'alpha': alpha,
            'eps': eps,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'centered': centered
        }
        super().__init__(params, defaults)

    def _init_state(self, param: np.ndarray, param_id: int) -> Dict[str, Any]:
        """Initialize optimizer state for a parameter."""
        state = {
            'step': 0,
            'square_avg': np.zeros_like(param),  # Running avg of squared gradients
        }
        if self.defaults['momentum'] > 0:
            state['momentum_buffer'] = np.zeros_like(param)
        if self.defaults['centered']:
            state['grad_avg'] = np.zeros_like(param)  # Running avg of gradients
        return state

    def step(self, grads: Optional[List[np.ndarray]] = None) -> None:
        """
        Perform a single optimization step.

        Args:
            grads: List of gradients corresponding to parameters.
        """
        if grads is None:
            raise ValueError("RMSprop requires gradients to be passed explicitly")

        grad_idx = 0
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            centered = group['centered']

            for param in group['params']:
                if grad_idx >= len(grads):
                    raise ValueError("Not enough gradients provided")

                grad = grads[grad_idx]
                grad_idx += 1

                if grad is None:
                    continue

                # Get or initialize state
                param_id = self._get_param_id(param)
                if param_id not in self.state:
                    self.state[param_id] = self._init_state(param, param_id)
                state = self.state[param_id]
                state['step'] += 1

                # Add weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad + weight_decay * param

                square_avg = state['square_avg']

                # Update running average of squared gradients
                # v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
                square_avg *= alpha
                square_avg += (1 - alpha) * (grad ** 2)

                if centered:
                    # Centered RMSprop: also track gradient average
                    grad_avg = state['grad_avg']
                    grad_avg *= alpha
                    grad_avg += (1 - alpha) * grad

                    # Use centered second moment: E[g^2] - E[g]^2
                    avg = np.sqrt(square_avg - grad_avg ** 2 + eps)
                else:
                    avg = np.sqrt(square_avg) + eps

                if momentum > 0:
                    # Apply momentum
                    buf = state['momentum_buffer']
                    buf *= momentum
                    buf += grad / avg
                    param -= lr * buf
                else:
                    # Standard update
                    param -= lr * grad / avg


class RMSpropTF(RMSprop):
    """
    RMSprop with TensorFlow-style implementation.

    TensorFlow's RMSprop has slightly different epsilon placement:
        w = w - lr * g / sqrt(v + eps)

    vs PyTorch-style:
        w = w - lr * g / (sqrt(v) + eps)

    This can make a difference for very small gradient values.
    """

    def step(self, grads: Optional[List[np.ndarray]] = None) -> None:
        """Perform optimization step with TF-style epsilon."""
        if grads is None:
            raise ValueError("RMSpropTF requires gradients to be passed explicitly")

        grad_idx = 0
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            centered = group['centered']

            for param in group['params']:
                if grad_idx >= len(grads):
                    raise ValueError("Not enough gradients provided")

                grad = grads[grad_idx]
                grad_idx += 1

                if grad is None:
                    continue

                param_id = self._get_param_id(param)
                if param_id not in self.state:
                    self.state[param_id] = self._init_state(param, param_id)
                state = self.state[param_id]
                state['step'] += 1

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                square_avg = state['square_avg']
                square_avg *= alpha
                square_avg += (1 - alpha) * (grad ** 2)

                if centered:
                    grad_avg = state['grad_avg']
                    grad_avg *= alpha
                    grad_avg += (1 - alpha) * grad
                    # TF-style: eps inside sqrt
                    avg = np.sqrt(square_avg - grad_avg ** 2 + eps)
                else:
                    # TF-style: eps inside sqrt
                    avg = np.sqrt(square_avg + eps)

                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf *= momentum
                    buf += grad / avg
                    param -= lr * buf
                else:
                    param -= lr * grad / avg
