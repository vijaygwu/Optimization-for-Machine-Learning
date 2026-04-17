"""Unit tests for optimizer implementations."""
import numpy as np
import sys
import os
import pickle
import subprocess
import tempfile
import textwrap

# Add the repository root to Python path for imports to work from any directory
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.optimizers import SGD, Adam, AdamW, RMSprop, Adagrad

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


def test_rmsprop_state_respects_param_group_overrides():
    """RMSprop state layout must follow each parameter group's options."""
    params = [np.array([1.0]), np.array([1.0])]
    opt = RMSprop(
        [
            {"params": [params[0]], "momentum": 0.9},
            {"params": [params[1]], "centered": True},
        ],
        lr=0.01,
        momentum=0.0,
        centered=False,
    )

    opt.step([np.array([1.0]), np.array([1.0])])

    state0 = opt.state[opt._get_param_id(params[0], 0, 0)]
    state1 = opt.state[opt._get_param_id(params[1], 1, 0)]

    assert "momentum_buffer" in state0
    assert "momentum_buffer" not in state1
    assert "grad_avg" not in state0
    assert "grad_avg" in state1


def test_adam_amsgrad_state_respects_param_group_overrides():
    """Adam must initialize AMSGrad state per parameter group."""
    params = [np.array([0.0]), np.array([0.0])]
    opt = Adam(
        [
            {"params": [params[0]]},
            {"params": [params[1]], "amsgrad": True},
        ],
        lr=0.001,
        amsgrad=False,
    )

    opt.step([np.array([1.0]), np.array([1.0])])

    state0 = opt.state[opt._get_param_id(params[0], 0, 0)]
    state1 = opt.state[opt._get_param_id(params[1], 1, 0)]

    assert "max_exp_avg_sq" not in state0
    assert "max_exp_avg_sq" in state1


def test_adamw_amsgrad_state_respects_param_group_overrides():
    """AdamW must initialize AMSGrad state per parameter group."""
    params = [np.array([0.0]), np.array([0.0])]
    opt = AdamW(
        [
            {"params": [params[0]]},
            {"params": [params[1]], "amsgrad": True},
        ],
        lr=0.001,
        amsgrad=False,
    )

    opt.step([np.array([1.0]), np.array([1.0])])

    state0 = opt.state[opt._get_param_id(params[0], 0, 0)]
    state1 = opt.state[opt._get_param_id(params[1], 1, 0)]

    assert "max_exp_avg_sq" not in state0
    assert "max_exp_avg_sq" in state1


def test_adagrad_initial_accumulator_respects_param_group_overrides():
    """Adagrad accumulators must honor per-group initialization values."""
    params = [np.array([0.0]), np.array([0.0])]
    opt = Adagrad(
        [
            {"params": [params[0]]},
            {"params": [params[1]], "initial_accumulator_value": 0.5},
        ],
        lr=0.1,
        initial_accumulator_value=0.0,
    )

    opt.step([np.array([1.0]), np.array([1.0])])

    state0 = opt.state[opt._get_param_id(params[0], 0, 0)]
    state1 = opt.state[opt._get_param_id(params[1], 1, 0)]

    np.testing.assert_allclose(state0["sum"], np.array([1.0]))
    np.testing.assert_allclose(state1["sum"], np.array([1.5]))

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


def test_nesterov_matches_buffer_form_not_exact_lookahead_gradient():
    """The SGD implementation follows the PyTorch-style buffer formula."""
    params = [np.array([1.0])]
    opt = SGD(params, lr=0.1, momentum=0.9, nesterov=True)

    def grad(theta):
        return 4.0 * theta ** 3

    opt.step([np.array([grad(params[0][0])])])

    theta_t = params[0][0].copy()
    prev_buffer = opt.state[next(iter(opt.state))]["momentum_buffer"].copy()
    current_grad = grad(theta_t)

    expected_update = current_grad + 0.9 * (0.9 * prev_buffer[0] + current_grad)
    lookahead_update = 0.9 * prev_buffer[0] + grad(theta_t - 0.1 * 0.9 * prev_buffer[0])

    opt.step([np.array([current_grad])])
    actual_update = (theta_t - params[0][0]) / 0.1

    assert np.isclose(actual_update, expected_update)
    assert not np.isclose(actual_update, lookahead_update)


def test_adam_state_checkpoint_survives_param_recreation():
    """
    Test that optimizer state survives save/load/recreate cycle.

    This tests the critical fix for checkpointing: optimizer state must use
    deterministic parameter identification (group_idx, param_idx, shape)
    instead of id(param), which changes when parameters are recreated.
    """
    # Create initial parameters and optimizer
    np.random.seed(42)
    original_params = [np.random.randn(10, 5), np.random.randn(5)]
    opt1 = Adam(original_params, lr=0.001)

    # Run a few steps to build up optimizer state
    for _ in range(5):
        grads = [np.random.randn(*p.shape) for p in original_params]
        opt1.step(grads)

    # Save state
    state_dict = opt1.state_dict()

    # Record parameter values after 5 steps
    params_after_5_steps = [p.copy() for p in original_params]

    # Simulate loading: create NEW parameter arrays (different id(param))
    # This simulates what happens when loading a model from disk
    new_params = [p.copy() for p in params_after_5_steps]

    # Create new optimizer with the new parameters and load state
    opt2 = Adam(new_params, lr=0.001)
    opt2.load_state_dict(state_dict)

    # Run one more step with the same gradient on both
    np.random.seed(123)  # Same seed for reproducibility
    test_grad = [np.random.randn(*p.shape) for p in original_params]

    # Step original optimizer
    opt1.step([g.copy() for g in test_grad])

    # Step restored optimizer (with recreated params)
    opt2.step([g.copy() for g in test_grad])

    # The parameters should be identical if state was properly restored
    for i, (p1, p2) in enumerate(zip(original_params, new_params)):
        np.testing.assert_allclose(
            p1, p2, rtol=1e-10, atol=1e-10,
            err_msg=f"Parameter {i} diverged after state restore. "
                    "Optimizer state was not properly restored after parameter recreation."
        )

    print("  - Adam checkpoint/restore test passed")


def test_adam_checkpoint_restores_across_processes():
    """Checkpoint keys must remain stable when saving and loading in new interpreters."""
    checkpoint_script = textwrap.dedent(
        """
        import pickle
        import sys
        from pathlib import Path

        import numpy as np

        repo_root = Path(sys.argv[1])
        artifact_dir = Path(sys.argv[2])
        sys.path.insert(0, str(repo_root))

        from src.optimizers import Adam

        np.random.seed(7)
        params = [np.random.randn(4, 3), np.random.randn(3)]
        optimizer = Adam(params, lr=1e-3)

        for _ in range(4):
            grads = [np.random.randn(*param.shape) for param in params]
            optimizer.step(grads)

        next_grads = [np.random.randn(*param.shape) for param in params]

        with open(artifact_dir / "checkpoint.pkl", "wb") as fh:
            pickle.dump(optimizer.state_dict(), fh)
        with open(artifact_dir / "params.pkl", "wb") as fh:
            pickle.dump([param.copy() for param in params], fh)
        with open(artifact_dir / "next_grads.pkl", "wb") as fh:
            pickle.dump(next_grads, fh)
        with open(artifact_dir / "state_keys.pkl", "wb") as fh:
            pickle.dump(sorted(optimizer.state_dict()["state"].keys()), fh)
        """
    )
    restore_script = textwrap.dedent(
        """
        import pickle
        import sys
        from pathlib import Path

        repo_root = Path(sys.argv[1])
        artifact_dir = Path(sys.argv[2])
        sys.path.insert(0, str(repo_root))

        from src.optimizers import Adam

        with open(artifact_dir / "checkpoint.pkl", "rb") as fh:
            checkpoint = pickle.load(fh)
        with open(artifact_dir / "params.pkl", "rb") as fh:
            params = pickle.load(fh)
        with open(artifact_dir / "next_grads.pkl", "rb") as fh:
            next_grads = pickle.load(fh)

        recreated_params = [param.copy() for param in params]
        optimizer = Adam(recreated_params, lr=1e-3)
        optimizer.load_state_dict(checkpoint)
        optimizer.step(next_grads)

        with open(artifact_dir / "restored_final.pkl", "wb") as fh:
            pickle.dump([param.copy() for param in recreated_params], fh)
        """
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        env = os.environ.copy()
        artifact_dir = os.path.abspath(tmp_dir)
        subprocess.run(
            [sys.executable, "-c", checkpoint_script, _repo_root, artifact_dir],
            check=True,
            env={**env, "PYTHONHASHSEED": "1"},
        )
        subprocess.run(
            [sys.executable, "-c", restore_script, _repo_root, artifact_dir],
            check=True,
            env={**env, "PYTHONHASHSEED": "2"},
        )

        with open(os.path.join(artifact_dir, "params.pkl"), "rb") as fh:
            uninterrupted_params = pickle.load(fh)
        with open(os.path.join(artifact_dir, "next_grads.pkl"), "rb") as fh:
            next_grads = pickle.load(fh)
        with open(os.path.join(artifact_dir, "state_keys.pkl"), "rb") as fh:
            state_keys = pickle.load(fh)
        with open(os.path.join(artifact_dir, "restored_final.pkl"), "rb") as fh:
            restored_final = pickle.load(fh)

        reference_optimizer = Adam(uninterrupted_params, lr=1e-3)
        with open(os.path.join(artifact_dir, "checkpoint.pkl"), "rb") as fh:
            reference_optimizer.load_state_dict(pickle.load(fh))
        reference_optimizer.step([grad.copy() for grad in next_grads])

        assert all(key.startswith("group") for key in state_keys)
        for ref_param, restored_param in zip(uninterrupted_params, restored_final):
            np.testing.assert_allclose(ref_param, restored_param, rtol=1e-10, atol=1e-10)

    print("  - Adam checkpoint/restore cross-process test passed")


def test_optimizers_require_explicit_gradient_lists():
    """NumPy optimizers should reject step() calls without explicit gradients."""
    optimizers = [
        SGD([np.array([1.0])], lr=0.1),
        Adam([np.array([1.0])], lr=1e-3),
        RMSprop([np.array([1.0])], lr=1e-3),
        Adagrad([np.array([1.0])], lr=1e-2),
    ]

    for optimizer in optimizers:
        try:
            optimizer.step()
            assert False, f"{type(optimizer).__name__} should require explicit gradients"
        except ValueError as exc:
            assert "requires gradients" in str(exc)

    print("  - Explicit gradient contract test passed")


def test_sgd_momentum_state_checkpoint():
    """Test that SGD with momentum state survives checkpoint/restore."""
    np.random.seed(42)
    original_params = [np.random.randn(10, 5)]
    opt1 = SGD(original_params, lr=0.01, momentum=0.9)

    # Build up momentum
    for _ in range(3):
        grads = [np.random.randn(*p.shape) for p in original_params]
        opt1.step(grads)

    state_dict = opt1.state_dict()
    params_after_steps = [p.copy() for p in original_params]

    # Recreate parameters and optimizer
    new_params = [p.copy() for p in params_after_steps]
    opt2 = SGD(new_params, lr=0.01, momentum=0.9)
    opt2.load_state_dict(state_dict)

    # Same gradient for both
    np.random.seed(456)
    test_grad = [np.random.randn(*p.shape) for p in original_params]

    opt1.step([g.copy() for g in test_grad])
    opt2.step([g.copy() for g in test_grad])

    for i, (p1, p2) in enumerate(zip(original_params, new_params)):
        np.testing.assert_allclose(
            p1, p2, rtol=1e-10, atol=1e-10,
            err_msg=f"SGD momentum state not properly restored for param {i}"
        )

    print("  - SGD momentum checkpoint/restore test passed")


def test_lr_scheduler_edge_cases():
    """Test edge cases in learning rate schedulers."""
    from src.optimizers import CosineAnnealingLR, StepLR, polynomial_lr, cosine_lr, warmup_lr

    # Test warmup_lr with zero warmup steps
    lr = warmup_lr(step=5, warmup_steps=0, base_lr=0.1)
    assert lr == 0.1, "warmup_lr should return base_lr when warmup_steps=0"

    # Zero-indexed warmup should take its first step immediately and hit
    # base_lr on the final warmup step.
    lr = warmup_lr(step=0, warmup_steps=4, base_lr=0.1)
    assert abs(lr - 0.025) < 1e-12, "warmup_lr step 0 should be base_lr / warmup_steps"

    lr = warmup_lr(step=3, warmup_steps=4, base_lr=0.1)
    assert abs(lr - 0.1) < 1e-12, "warmup_lr should reach base_lr at step warmup_steps - 1"

    lr = warmup_lr(step=0, warmup_steps=4, base_lr=0.1, warmup_init_lr=0.02)
    assert abs(lr - 0.04) < 1e-12, "warmup_lr should interpolate from warmup_init_lr on step 0"

    lr = warmup_lr(step=3, warmup_steps=4, base_lr=0.1, warmup_init_lr=0.02)
    assert abs(lr - 0.1) < 1e-12, "warmup_lr with warmup_init_lr should still reach base_lr"

    # Test polynomial_lr with step beyond total_steps
    lr = polynomial_lr(step=100, total_steps=50, base_lr=0.1, end_lr=0.01)
    assert lr == 0.01, "polynomial_lr should return end_lr when step >= total_steps"

    # Test polynomial_lr raises error when total_steps <= warmup_steps
    # Use step >= warmup_steps to trigger the validation
    try:
        polynomial_lr(step=10, total_steps=10, base_lr=0.1, warmup_steps=10)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be greater than warmup_steps" in str(e)

    # Test cosine_lr with edge cases
    lr = cosine_lr(step=0, total_steps=100, base_lr=0.1, warmup_steps=0)
    assert abs(lr - 0.1) < 1e-10, "cosine_lr at step 0 should be base_lr"

    lr = cosine_lr(step=0, total_steps=10, base_lr=0.1, warmup_steps=4)
    assert abs(lr - 0.025) < 1e-12, "cosine_lr warmup should share zero-indexed warmup convention"

    # Scheduler construction must not change LR before the first explicit step
    params = [np.array([1.0])]
    opt = SGD(params, lr=0.1)
    scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=0.01)
    assert abs(opt.param_groups[0]["lr"] - 0.1) < 1e-12
    assert scheduler.last_epoch == -1

    # Test CosineAnnealingLR rejects T_max=0
    params = [np.array([1.0])]
    opt = SGD(params, lr=0.1)
    try:
        CosineAnnealingLR(opt, T_max=0)
        assert False, "Should have raised ValueError for T_max=0"
    except ValueError as e:
        assert "T_max must be positive" in str(e)

    # Test StepLR rejects step_size=0
    params = [np.array([1.0])]
    opt = SGD(params, lr=0.1)
    try:
        StepLR(opt, step_size=0)
        assert False, "Should have raised ValueError for step_size=0"
    except ValueError as e:
        assert "step_size must be positive" in str(e)

    print("  - LR scheduler edge case tests passed")


def test_convolutional_initialization_uses_kernel_area():
    """Convolutional fan-in/out must include kernel area."""
    from src.optimizers import initialize_parameters

    np.random.seed(0)
    shape = (64, 3, 7, 7)
    weights = initialize_parameters(shape, "he_normal")

    expected_var = 2.0 / (3 * 7 * 7)
    actual_var = float(np.var(weights))
    assert np.isclose(actual_var, expected_var, rtol=0.25), (
        f"Expected variance near {expected_var}, got {actual_var}"
    )


if __name__ == "__main__":
    print("Running optimizer tests...")
    test_sgd_updates_in_correct_direction()
    print("  - SGD direction test passed")
    test_adam_bias_correction()
    print("  - Adam bias correction test passed")
    test_momentum_accumulation()
    print("  - Momentum accumulation test passed")
    test_adam_state_checkpoint_survives_param_recreation()
    test_adam_checkpoint_restores_across_processes()
    test_optimizers_require_explicit_gradient_lists()
    test_sgd_momentum_state_checkpoint()
    test_lr_scheduler_edge_cases()
    test_convolutional_initialization_uses_kernel_area()
    print("\nAll tests passed!")
