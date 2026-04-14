"""
Adam and AdamW Optimizers

This module implements Adam (Adaptive Moment Estimation) and its variants.
Adam combines the benefits of AdaGrad and RMSprop with momentum.

Author: Optimization for Machine Learning Book
License: MIT
"""

from typing import Dict, List, Optional, Any, Iterator, Tuple
import numpy as np

from .base import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).

    Adam maintains per-parameter adaptive learning rates using estimates
    of first and second moments of the gradients.

    Update rules:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t           # First moment
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2         # Second moment
        m_hat_t = m_t / (1 - beta1^t)                        # Bias correction
        v_hat_t = v_t / (1 - beta2^t)                        # Bias correction
        w_{t+1} = w_t - lr * m_hat_t / (sqrt(v_hat_t) + eps)

    Attributes:
        lr (float): Learning rate (default: 0.001).
        betas (Tuple[float, float]): Coefficients for moment estimates.
        eps (float): Term for numerical stability (default: 1e-8).
        weight_decay (float): L2 regularization coefficient.
        amsgrad (bool): Whether to use AMSGrad variant.

    Example:
        >>> params = [np.random.randn(10, 5), np.random.randn(5, 2)]
        >>> optimizer = Adam(params, lr=0.001)
        >>> for epoch in range(100):
        ...     grads = compute_gradients(params)
        ...     optimizer.step(grads)

    References:
        - Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
        - Reddi et al., "On the Convergence of Adam and Beyond", ICLR 2018 (AMSGrad)
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False
    ):
        """
        Initialize Adam optimizer.

        Args:
            params: Iterator of parameters to optimize.
            lr: Learning rate (default: 0.001).
            betas: Coefficients for computing running averages of gradient
                and its square. Default: (0.9, 0.999).
            eps: Term added to denominator for numerical stability (default: 1e-8).
            weight_decay: L2 regularization coefficient (default: 0).
            amsgrad: Whether to use the AMSGrad variant that maintains
                maximum of past squared gradients (default: False).

        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad
        }
        super().__init__(params, defaults)

    def _init_state(self, param: np.ndarray, param_id: int) -> Dict[str, Any]:
        """Initialize optimizer state for a parameter."""
        state = {
            'step': 0,
            'exp_avg': np.zeros_like(param),      # First moment (m)
            'exp_avg_sq': np.zeros_like(param),   # Second moment (v)
        }
        if self.defaults['amsgrad']:
            state['max_exp_avg_sq'] = np.zeros_like(param)  # v_max for AMSGrad
        return state

    def step(self, grads: Optional[List[np.ndarray]] = None) -> None:
        """
        Perform a single optimization step.

        Args:
            grads: List of gradients corresponding to parameters.
        """
        if grads is None:
            raise ValueError("Adam requires gradients to be passed explicitly")

        grad_idx = 0
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

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

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step = state['step']

                # Update biased first moment estimate: m_t = beta1*m_{t-1} + (1-beta1)*g
                exp_avg *= beta1
                exp_avg += (1 - beta1) * grad

                # Update biased second moment estimate: v_t = beta2*v_{t-1} + (1-beta2)*g^2
                exp_avg_sq *= beta2
                exp_avg_sq += (1 - beta2) * (grad ** 2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Bias-corrected estimates
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                if amsgrad:
                    # Use maximum of past squared gradients
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    np.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    v_hat = max_exp_avg_sq / bias_correction2

                # Update: w = w - lr * m_hat / (sqrt(v_hat) + eps)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)


class AdamW(Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay).

    AdamW fixes the weight decay implementation in Adam by decoupling
    weight decay from the gradient-based update. This leads to better
    generalization in many cases.

    The key difference from Adam:
        Adam:  w = w - lr * m_hat / (sqrt(v_hat) + eps)  where grad += wd * w
        AdamW: w = w - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * w

    Example:
        >>> optimizer = AdamW(params, lr=0.001, weight_decay=0.01)

    Reference:
        Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ):
        """
        Initialize AdamW optimizer.

        Args:
            params: Iterator of parameters to optimize.
            lr: Learning rate (default: 0.001).
            betas: Coefficients for computing running averages (default: (0.9, 0.999)).
            eps: Term for numerical stability (default: 1e-8).
            weight_decay: Decoupled weight decay coefficient (default: 0.01).
            amsgrad: Whether to use AMSGrad variant (default: False).
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad
        }
        super().__init__(params, defaults)

    def _init_state(self, param: np.ndarray, param_id: int) -> Dict[str, Any]:
        """Initialize optimizer state for a parameter."""
        state = {
            'step': 0,
            'exp_avg': np.zeros_like(param),
            'exp_avg_sq': np.zeros_like(param),
        }
        if self.defaults['amsgrad']:
            state['max_exp_avg_sq'] = np.zeros_like(param)
        return state

    def step(self, grads: Optional[List[np.ndarray]] = None) -> None:
        """Perform optimization step with decoupled weight decay."""
        if grads is None:
            raise ValueError("AdamW requires gradients to be passed explicitly")

        grad_idx = 0
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

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

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step = state['step']

                # Note: No weight decay added to gradient (decoupled)

                # Update moments
                exp_avg *= beta1
                exp_avg += (1 - beta1) * grad

                exp_avg_sq *= beta2
                exp_avg_sq += (1 - beta2) * (grad ** 2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    np.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    v_hat = max_exp_avg_sq / bias_correction2

                # Adam update
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

                # Decoupled weight decay (applied directly to weights)
                if weight_decay != 0:
                    param -= lr * weight_decay * param


class NAdam(Optimizer):
    """
    NAdam optimizer (Nesterov-accelerated Adam).

    NAdam incorporates Nesterov momentum into Adam, combining the
    benefits of both approaches.

    Reference:
        Dozat, "Incorporating Nesterov Momentum into Adam", ICLR Workshop 2016
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 0.002,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004
    ):
        """
        Initialize NAdam optimizer.

        Args:
            params: Iterator of parameters to optimize.
            lr: Learning rate (default: 0.002).
            betas: Coefficients for moment estimates (default: (0.9, 0.999)).
            eps: Term for numerical stability (default: 1e-8).
            weight_decay: L2 regularization coefficient (default: 0).
            momentum_decay: Decay for momentum schedule (default: 0.004).
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")

        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'momentum_decay': momentum_decay
        }
        super().__init__(params, defaults)

        # Precompute momentum schedule
        self._mu_schedule_product = 1.0

    def _init_state(self, param: np.ndarray, param_id: int) -> Dict[str, Any]:
        """Initialize optimizer state."""
        return {
            'step': 0,
            'exp_avg': np.zeros_like(param),
            'exp_avg_sq': np.zeros_like(param),
        }

    def step(self, grads: Optional[List[np.ndarray]] = None) -> None:
        """Perform optimization step."""
        if grads is None:
            raise ValueError("NAdam requires gradients to be passed explicitly")

        grad_idx = 0
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum_decay = group['momentum_decay']

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
                step = state['step']

                # Weight decay
                if weight_decay != 0:
                    grad = grad + weight_decay * param

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Momentum schedule
                mu_t = beta1 * (1 - 0.5 * (0.96 ** (step * momentum_decay)))
                mu_t_next = beta1 * (1 - 0.5 * (0.96 ** ((step + 1) * momentum_decay)))

                # Update biased first moment
                exp_avg *= beta1
                exp_avg += (1 - beta1) * grad

                # Update biased second moment
                exp_avg_sq *= beta2
                exp_avg_sq += (1 - beta2) * (grad ** 2)

                # Bias correction for second moment
                bias_correction2 = 1 - beta2 ** step
                v_hat = exp_avg_sq / bias_correction2

                # Nesterov-style update using current and next momentum
                # This is the key difference from Adam
                mu_product = 1.0
                for i in range(1, step + 1):
                    mu_i = beta1 * (1 - 0.5 * (0.96 ** (i * momentum_decay)))
                    mu_product *= mu_i

                m_hat = (mu_t_next * exp_avg / (1 - mu_product * mu_t_next) +
                        (1 - mu_t) * grad / (1 - mu_product))

                # Update
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
