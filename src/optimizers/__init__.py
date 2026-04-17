"""
Optimizers Module for Machine Learning

This module provides clean, documented, production-ready implementations
of popular optimization algorithms. All optimizers follow the PyTorch API
design pattern for familiarity and ease of use.

Available Optimizers:
    - SGD: Stochastic Gradient Descent with momentum
    - SGDW: SGD with decoupled weight decay
    - Adam: Adaptive Moment Estimation
    - AdamW: Adam with decoupled weight decay
    - NAdam: Nesterov-accelerated Adam
    - RMSprop: Root Mean Square Propagation
    - RMSpropTF: RMSprop with TensorFlow-style implementation
    - Adagrad: Adaptive Gradient
    - AdagradSparse: Adagrad optimized for sparse gradients
    - Adadelta: Adadelta optimizer

Learning Rate Schedulers:
    - StepLR: Step decay scheduler
    - CosineAnnealingLR: Cosine annealing scheduler

Usage Example:
    >>> from src.optimizers import Adam, clip_grad_norm_
    >>> params = [np.random.randn(10, 5), np.random.randn(5, 2)]
    >>> optimizer = Adam(params, lr=0.001)
    >>> for epoch in range(100):
    ...     grads = compute_gradients(params)
    ...     clip_grad_norm_(grads, max_norm=1.0)
    ...     optimizer.step(grads)

Author: Optimization for Machine Learning Book
License: MIT
"""

# Base classes
from .base import Optimizer, LRScheduler, StepLR, CosineAnnealingLR

# SGD variants
from .sgd import SGD, SGDW

# Adam variants
from .adam import Adam, AdamW, NAdam

# RMSprop variants
from .rmsprop import RMSprop, RMSpropTF

# Adagrad variants
from .adagrad import Adagrad, AdagradSparse, Adadelta

# Utility functions
from .utils import (
    clip_grad_norm_,
    clip_grad_value_,
    compute_grad_norm,
    warmup_lr,
    cosine_lr,
    polynomial_lr,
    exponential_lr,
    get_lr_scheduler,
    initialize_parameters,
    count_parameters,
    flatten_grads,
    unflatten_grads,
    check_gradients,
    lr_finder,
    random_hyperparameter_search,
)

__all__ = [
    # Base
    'Optimizer',
    'LRScheduler',
    'StepLR',
    'CosineAnnealingLR',
    # SGD
    'SGD',
    'SGDW',
    # Adam
    'Adam',
    'AdamW',
    'NAdam',
    # RMSprop
    'RMSprop',
    'RMSpropTF',
    # Adagrad
    'Adagrad',
    'AdagradSparse',
    'Adadelta',
    # Utils
    'clip_grad_norm_',
    'clip_grad_value_',
    'compute_grad_norm',
    'warmup_lr',
    'cosine_lr',
    'polynomial_lr',
    'exponential_lr',
    'get_lr_scheduler',
    'initialize_parameters',
    'count_parameters',
    'flatten_grads',
    'unflatten_grads',
    'check_gradients',
    'lr_finder',
    'random_hyperparameter_search',
]

__version__ = '1.0.0'
