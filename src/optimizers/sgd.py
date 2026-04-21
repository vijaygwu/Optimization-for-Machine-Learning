"""
Stochastic Gradient Descent (SGD) Optimizer

This module implements SGD with optional momentum and Nesterov acceleration.
SGD is the foundational optimization algorithm in deep learning.

Author: Optimization for Machine Learning Book
License: MIT
"""

from typing import Dict, List, Optional, Any, Iterator
import numpy as np

from .base import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.

    Implements the update rule:

    Without momentum:
        v_t = g_t
        w_{t+1} = w_t - lr * v_t

    With momentum (classical):
        v_t = momentum * v_{t-1} + g_t
        w_{t+1} = w_t - lr * v_t

    With PyTorch-style Nesterov momentum:
        b_t = momentum * b_{t-1} + g_t
        u_t = g_t + momentum * b_t
        w_{t+1} = w_t - lr * u_t

    Where g_t is the gradient and w_t are the parameters.
    This practical buffer form is motivated by look-ahead gradients, but it is
    not identical to evaluating grad(w_t - lr * momentum * b_{t-1}) on
    general nonlinear objectives.

    Attributes:
        lr (float): Learning rate.
        momentum (float): Momentum factor (default: 0).
        weight_decay (float): L2 regularization coefficient (default: 0).
        dampening (float): Dampening for momentum (default: 0).
        nesterov (bool): Use Nesterov momentum (default: False).

    Example:
        >>> params = [np.random.randn(10, 5), np.random.randn(5, 2)]
        >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
        >>> for epoch in range(100):
        ...     grads = compute_gradients(params)
        ...     optimizer.step(grads)

    References:
        - Sutskever et al., "On the importance of initialization and momentum
          in deep learning", ICML 2013
        - Nesterov, "A method for unconstrained convex minimization problem
          with the rate of convergence O(1/k^2)", 1983
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False
    ):
        """
        Initialize SGD optimizer.

        Args:
            params: Iterator of parameters to optimize.
            lr: Learning rate (default: 0.01).
            momentum: Momentum factor (default: 0). Typical values: 0.9, 0.99.
            weight_decay: L2 regularization coefficient (default: 0).
            dampening: Dampening for momentum (default: 0).
            nesterov: Use Nesterov momentum (default: False).
                Requires momentum > 0 and dampening = 0.

        Raises:
            ValueError: If lr < 0, momentum < 0, weight_decay < 0, or
                invalid Nesterov configuration.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening = 0")

        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'dampening': dampening,
            'nesterov': nesterov
        }
        super().__init__(params, defaults)

    def _init_state(self, param: np.ndarray, param_id: str) -> Dict[str, Any]:
        """Initialize optimizer state for a parameter."""
        return {
            'momentum_buffer': None,
            'step': 0
        }

    def step(self, grads: Optional[List[np.ndarray]] = None) -> None:
        """
        Perform a single optimization step.

        Args:
            grads: List of gradients corresponding to parameters.
                Must be in same order as parameters were added.

        Example:
            >>> # Forward pass
            >>> y_pred = model_forward(x, params)
            >>> loss = compute_loss(y_pred, y_true)
            >>>
            >>> # Backward pass
            >>> grads = compute_gradients(loss, params)
            >>>
            >>> # Update
            >>> optimizer.step(grads)
        """
        if grads is None:
            raise ValueError("SGD requires gradients to be passed explicitly")

        grad_idx = 0
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']

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
                    self.state[param_id] = self._init_state(param, param_id)
                state = self.state[param_id]
                state['step'] += 1

                # Add weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad + weight_decay * param

                # Apply momentum
                if momentum != 0:
                    if state['momentum_buffer'] is None:
                        # First step: initialize buffer with gradient
                        buf = state['momentum_buffer'] = grad.copy()
                    else:
                        buf = state['momentum_buffer']
                        # buf = momentum * buf + (1 - dampening) * grad
                        buf *= momentum
                        buf += (1 - dampening) * grad

                    if nesterov:
                        # PyTorch-style Nesterov buffer correction.
                        grad = grad + momentum * buf
                    else:
                        grad = buf

                # Update parameters: w = w - lr * grad
                param -= lr * grad

        self._raise_if_extra_gradients(grads, grad_idx)


class SGDW(SGD):
    """
    SGD with decoupled weight decay (SGDW).

    Unlike standard SGD with weight decay (which is equivalent to L2
    regularization), SGDW decouples weight decay from the gradient update.

    Update rule:
        g_t = gradient + weight_decay * w_t  (standard L2)
        vs.
        w_{t+1} = w_t - lr * g_t - lr * weight_decay * w_t  (decoupled)

    The decoupled version is equivalent to standard weight decay rather
    than L2 regularization, which can lead to better generalization.

    Reference:
        Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
    """

    def step(self, grads: Optional[List[np.ndarray]] = None) -> None:
        """Perform optimization step with decoupled weight decay."""
        if grads is None:
            raise ValueError("SGDW requires gradients to be passed explicitly")

        grad_idx = 0
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']

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
                    self.state[param_id] = self._init_state(param, param_id)
                state = self.state[param_id]
                state['step'] += 1

                # Apply momentum (without weight decay in gradient)
                if momentum != 0:
                    if state['momentum_buffer'] is None:
                        buf = state['momentum_buffer'] = grad.copy()
                    else:
                        buf = state['momentum_buffer']
                        buf *= momentum
                        buf += (1 - dampening) * grad

                    if nesterov:
                        # PyTorch-style Nesterov buffer correction.
                        grad = grad + momentum * buf
                    else:
                        grad = buf

                # Decoupled weight decay + gradient update (single step)
                # Weight decay is applied to PRE-update params
                if weight_decay != 0:
                    param *= (1 - lr * weight_decay)  # Apply weight decay first
                param -= lr * grad  # Then gradient update

        self._raise_if_extra_gradients(grads, grad_idx)
