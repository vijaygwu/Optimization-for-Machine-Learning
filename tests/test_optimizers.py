"""Unit tests for optimizer implementations."""
import numpy as np
import sys
sys.path.insert(0, '..')
from src.optimizers import SGD, Adam, RMSprop, Adagrad

def test_sgd_updates_in_correct_direction():
    """Test that SGD moves parameters in negative gradient direction."""
    params = [np.array([1.0, 2.0])]
    opt = SGD(params, lr=0.1)
    grads = [np.array([1.0, 1.0])]  # Positive gradient
    opt.step(grads)
    assert params[0][0] < 1.0, "Parameter should decrease with positive gradient"
    assert params[0][1] < 2.0, "Parameter should decrease with positive gradient"

def test_adam_bias_correction():
    """Test Adam bias correction at early iterations."""
    params = [np.array([0.0])]
    opt = Adam(params, lr=0.001, betas=(0.9, 0.999))
    grads = [np.array([1.0])]
    opt.step(grads)
    # First step should have significant update due to bias correction
    assert abs(params[0][0]) > 0.0005, "Adam should update on first step"

def test_momentum_accumulation():
    """Test that momentum accumulates over steps."""
    params = [np.array([0.0])]
    opt = SGD(params, lr=0.1, momentum=0.9)
    grads = [np.array([1.0])]
    opt.step(grads)
    first_update = abs(params[0][0])
    opt.step(grads)
    second_update = abs(params[0][0]) - first_update
    assert second_update > first_update * 0.5, "Momentum should accelerate updates"

if __name__ == "__main__":
    test_sgd_updates_in_correct_direction()
    test_adam_bias_correction()
    test_momentum_accumulation()
    print("All tests passed!")
