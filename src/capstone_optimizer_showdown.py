"""Runnable companion code for the Book 2 optimizer-showdown capstone."""

from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((784, 256)) * np.sqrt(2.0 / 784)
        self.b1 = np.zeros((1, 256))
        self.W2 = rng.standard_normal((256, 128)) * np.sqrt(2.0 / 256)
        self.b2 = np.zeros((1, 128))
        self.W3 = rng.standard_normal((128, 10)) * np.sqrt(2.0 / 128)
        self.b3 = np.zeros((1, 10))
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.cache["z1"] = X @ self.W1 + self.b1
        self.cache["a1"] = self.relu(self.cache["z1"])
        self.cache["z2"] = self.cache["a1"] @ self.W2 + self.b2
        self.cache["a2"] = self.relu(self.cache["z2"])
        self.cache["z3"] = self.cache["a2"] @ self.W3 + self.b3
        self.cache["X"] = X
        return self.cache["z3"]

    def backward(self, y_true):
        batch_size = y_true.shape[0]
        grads = {}
        probs = self.softmax(self.cache["z3"])
        dz3 = probs - y_true
        grads["W3"] = self.cache["a2"].T @ dz3 / batch_size
        grads["b3"] = np.mean(dz3, axis=0, keepdims=True)
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_derivative(self.cache["z2"])
        grads["W2"] = self.cache["a1"].T @ dz2 / batch_size
        grads["b2"] = np.mean(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.cache["z1"])
        grads["W1"] = self.cache["X"].T @ dz1 / batch_size
        grads["b1"] = np.mean(dz1, axis=0, keepdims=True)
        return grads

    def cross_entropy_loss(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-15, 1.0)
        log_probs = np.log(probs)
        return -np.mean(np.sum(y_true * log_probs, axis=1))

    def predict_proba(self, X):
        return self.softmax(self.forward(X))

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = params

    def copy_params(self):
        return [p.copy() for p in self.get_params()]


class Optimizer:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self, model, grads):
        raise NotImplementedError("Subclasses must implement step()")

    def reset(self):
        pass

    def get_name(self):
        return self.__class__.__name__


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def step(self, model, grads):
        param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
        params = model.get_params()
        for i, name in enumerate(param_names):
            params[i] = params[i] - self.lr * grads[name]
        model.set_params(params)


class SGDMomentum(Optimizer):
    def __init__(self, learning_rate=0.05, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None

    def step(self, model, grads):
        param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
        params = model.get_params()
        if self.velocity is None:
            self.velocity = {}
            for i, name in enumerate(param_names):
                self.velocity[name] = np.zeros_like(params[i])
        for i, name in enumerate(param_names):
            self.velocity[name] = self.momentum * self.velocity[name] + grads[name]
            params[i] = params[i] - self.lr * self.velocity[name]
        model.set_params(params)

    def reset(self):
        self.velocity = None

    def get_name(self):
        return f"SGD with Momentum(momentum={self.momentum})"


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.99, epsilon=1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.cache = None

    def step(self, model, grads):
        param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
        params = model.get_params()
        if self.cache is None:
            self.cache = {}
            for name in param_names:
                self.cache[name] = np.zeros_like(grads[name])
        for i, name in enumerate(param_names):
            self.cache[name] = self.rho * self.cache[name] + (1 - self.rho) * grads[name] ** 2
            params[i] = params[i] - self.lr * grads[name] / (
                np.sqrt(self.cache[name]) + self.epsilon
            )
        model.set_params(params)

    def reset(self):
        self.cache = None

    def get_name(self):
        return f"RMSprop(rho={self.rho})"


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, model, grads):
        param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
        params = model.get_params()
        if self.m is None:
            self.m = {}
            self.v = {}
            for name in param_names:
                self.m[name] = np.zeros_like(grads[name])
                self.v[name] = np.zeros_like(grads[name])
        self.t += 1
        for i, name in enumerate(param_names):
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name] ** 2
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            params[i] = params[i] - (self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon))
        model.set_params(params)

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

    def get_name(self):
        return f"Adam(beta1={self.beta1}, beta2={self.beta2})"


def load_mnist():
    from sklearn.datasets import fetch_openml, load_digits
    from sklearn.model_selection import train_test_split

    print("Loading MNIST dataset...")
    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        X = X / 255.0
        dataset_name = "MNIST"
    except Exception as e:
        # Offline-safe fallback: upsample sklearn's bundled digits to the
        # 28x28 shape expected by the capstone MLP.
        print(
            f"  OpenML unavailable ({type(e).__name__}); using "
            "sklearn digits upsampled to 28x28 for offline smoke testing"
        )
        digits = load_digits()
        X = digits.data.reshape(-1, 8, 8).astype(np.float64) / 16.0
        X = np.kron(X, np.ones((1, 3, 3)))
        X = np.pad(X, ((0, 0), (2, 2), (2, 2)), mode="constant")
        X = X.reshape(len(X), -1)
        y = digits.target
        dataset_name = "sklearn digits (upsampled to 28x28)"
    y_onehot = np.zeros((len(y), 10))
    y_onehot[np.arange(len(y)), y] = 1
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_onehot, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, random_state=42, stratify=np.argmax(y_temp, axis=1)
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Dataset source: {dataset_name}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_accuracy(model, X, y):
    logits = model.forward(X)
    pred_classes = np.argmax(logits, axis=1)
    true_classes = np.argmax(y, axis=1)
    return np.mean(pred_classes == true_classes)


def create_batches(X, y, batch_size, shuffle=True, epoch_seed=None):
    """
    Generate minibatches for training.

    Args:
        X: Input features
        y: Target labels
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        epoch_seed: If provided, uses a local RNG seeded with this value for
            deterministic shuffling. This is critical for fair optimizer
            comparisons - all optimizers see identical batch orderings when
            using the same epoch_seed sequence.

    Yields:
        (X_batch, y_batch) tuples
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    if shuffle:
        if epoch_seed is None:
            # Non-deterministic shuffle (uses global RNG state)
            np.random.shuffle(indices)
        else:
            # Deterministic shuffle (isolated local RNG for reproducibility)
            rng = np.random.default_rng(epoch_seed)
            rng.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def train(
    model,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=20,
    batch_size=128,
    verbose=True,
    base_seed=42,
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
        "best_params": None,
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }
    import time

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses = []
        epoch_seed = base_seed + epoch
        for X_batch, y_batch in create_batches(
            X_train, y_train, batch_size, epoch_seed=epoch_seed
        ):
            logits = model.forward(X_batch)
            loss = model.cross_entropy_loss(logits, y_batch)
            epoch_losses.append(loss)
            grads = model.backward(y_batch)
            optimizer.step(model, grads)
        epoch_time = time.time() - epoch_start
        train_loss = np.mean(epoch_losses)
        train_acc = compute_accuracy(model, X_train, y_train)
        val_logits = model.forward(X_val)
        val_loss = model.cross_entropy_loss(val_logits, y_val)
        val_acc = compute_accuracy(model, X_val, y_val)
        if val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_params"] = model.copy_params()
            history["best_epoch"] = epoch + 1
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_time)
        if verbose:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )
    return history


def run_optimizer_showdown():
    """
    Run a fair head-to-head comparison of optimizers on MNIST.

    Fairness guarantees:
    1. All optimizers start from identical initial weights (saved and restored)
    2. All optimizers see identical epoch-by-epoch batch permutations
       (via deterministic seeding in create_batches with base_seed=42)
    3. Each optimizer's internal state is reset before training

    This ensures the comparison measures optimizer performance, not shuffle luck.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist()
    optimizers = [
        SGD(learning_rate=0.1),
        SGDMomentum(learning_rate=0.05, momentum=0.9),
        RMSprop(learning_rate=0.001, rho=0.99),
        Adam(learning_rate=0.001, beta1=0.9, beta2=0.999),
        AdamW(learning_rate=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01),
    ]

    # Save initial weights - all optimizers will start from this checkpoint
    base_model = MLP(seed=42)
    initial_params = base_model.copy_params()

    all_results = {}
    trained_models = {}

    # Fixed seed for batch shuffling - ensures all optimizers see same data order
    training_seed = 42

    for opt in optimizers:
        print(f"\n{'='*60}")
        print(f"Training with {opt.get_name()}")
        print(f"{'='*60}")

        # Reset to identical starting point for fair comparison
        model = MLP(seed=42)
        model.set_params([p.copy() for p in initial_params])
        opt.reset()

        # Train with deterministic batch ordering (same for all optimizers)
        history = train(
            model,
            opt,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=20,
            batch_size=128,
            base_seed=training_seed,  # Ensures identical batch order across optimizers
        )
        all_results[opt.get_name()] = history
        trained_models[opt.get_name()] = model

    # Select the winner using validation performance only.
    best_name = max(all_results, key=lambda name: all_results[name]["best_val_acc"])
    best_history = all_results[best_name]
    best_val_acc = best_history["best_val_acc"]
    best_epoch = best_history["best_epoch"]

    # Restore the checkpoint that achieved best validation accuracy.
    best_model = MLP(seed=42)
    best_model.set_params(best_history["best_params"])

    print(f"\n{'='*60}")
    print("FINAL TEST SET EVALUATION (selected model only)")
    print(f"{'='*60}")
    test_acc = compute_accuracy(best_model, X_test, y_test)
    all_results[best_name]["selected_test_acc"] = test_acc
    print(
        f"Selected by validation: {best_name.split('(')[0]} "
        f"(best val acc: {best_val_acc*100:.2f}% at epoch {best_epoch})"
    )
    print(f"Held-out test accuracy: {test_acc*100:.2f}%")
    return all_results


def plot_results(all_results):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {
        "SGD": "#1f77b4",
        "SGD with Momentum": "#ff7f0e",
        "RMSprop": "#2ca02c",
        "Adam": "#d62728",
        "AdamW": "#17becf",
    }

    def get_color(name):
        if "SGD with Momentum" in name:
            return colors["SGD with Momentum"]
        for key in ["AdamW", "Adam", "RMSprop", "SGD"]:
            if key in name:
                return colors[key]
        return "black"

    sample_history = next(iter(all_results.values()))
    num_epochs = len(sample_history["train_loss"])
    epochs = range(1, num_epochs + 1)
    ax1 = axes[0, 0]
    for name, history in all_results.items():
        ax1.plot(
            range(1, len(history["train_loss"]) + 1),
            history["train_loss"],
            label=name.split("(")[0],
            color=get_color(name),
            linewidth=2,
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    ax2 = axes[0, 1]
    for name, history in all_results.items():
        ax2.plot(
            range(1, len(history["val_acc"]) + 1),
            [a * 100 for a in history["val_acc"]],
            label=name.split("(")[0],
            color=get_color(name),
            linewidth=2,
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("Validation Accuracy Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=95, color="gray", linestyle="--", alpha=0.5)
    ax3 = axes[1, 0]
    for name, history in all_results.items():
        train_acc = [a * 100 for a in history["train_acc"]]
        val_acc = [a * 100 for a in history["val_acc"]]
        gap = [t - v for t, v in zip(train_acc, val_acc)]
        ax3.plot(
            range(1, len(gap) + 1),
            gap,
            label=name.split("(")[0],
            color=get_color(name),
            linewidth=2,
        )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Train - Validation Accuracy (%)")
    ax3.set_title("Generalization Gap")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax4 = axes[1, 1]
    final_metrics = []
    for name, history in all_results.items():
        val_acc = history["val_acc"]
        total_epochs = len(val_acc)
        epochs_to_95 = None
        for i, acc in enumerate(val_acc):
            if acc >= 0.95:
                epochs_to_95 = i + 1
                break
        final_metrics.append(
            {
                "name": name.split("(")[0],
                "final_val_acc": val_acc[-1] * 100,
                "epochs_to_95": epochs_to_95 if epochs_to_95 else f">{total_epochs}",
                "total_time": sum(history["epoch_times"]),
            }
        )
    names = [m["name"] for m in final_metrics]
    epochs_values = [
        m["epochs_to_95"] if isinstance(m["epochs_to_95"], int) else num_epochs + 1
        for m in final_metrics
    ]
    bar_colors = [get_color(m["name"]) for m in final_metrics]
    bars = ax4.bar(names, epochs_values, color=bar_colors)
    ax4.set_ylabel("Epochs to 95% Validation Accuracy")
    ax4.set_title("Convergence Speed (fewer epochs = faster)")
    ax4.set_ylim(0, num_epochs + 5)
    for bar, m in zip(bars, final_metrics):
        label = str(m["epochs_to_95"])
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig("optimizer_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    return final_metrics


def print_summary_table(final_metrics, all_results):
    print("\n" + "=" * 80)
    print("OPTIMIZER SHOWDOWN RESULTS")
    print("=" * 80)
    print(
        f"\n{'Optimizer':<20} {'Val Acc':<12} "
        f"{'Epochs to 95%':<15} {'Total Time':<12}"
    )
    print("-" * 80)
    for name, history in all_results.items():
        val_acc = history["val_acc"]
        final_val_acc = val_acc[-1] * 100
        total_time = sum(history["epoch_times"])
        epochs_to_95 = "N/A"
        for i, acc in enumerate(val_acc):
            if acc >= 0.95:
                epochs_to_95 = str(i + 1)
                break
        short_name = name.split("(")[0]
        print(
            f"{short_name:<20} {final_val_acc:<12.2f}% "
            f"{epochs_to_95:<15} {total_time:<12.2f}s"
        )

    selected_name = next(
        (name for name, history in all_results.items() if "selected_test_acc" in history),
        None,
    )
    if selected_name is not None:
        selected_test_acc = all_results[selected_name]["selected_test_acc"] * 100
        print(f"\nSelected by validation: {selected_name.split('(')[0]}")
        print(f"Held-out test accuracy: {selected_test_acc:.2f}%")

    print("\nNote: Table ranks optimizers by validation metrics only.")
    print("      Held-out test accuracy is evaluated once after selection.")


def learning_rate_sensitivity(X_train, y_train, X_val, y_val):
    multipliers = [0.1, 0.3, 1.0, 3.0, 10.0]
    base_lrs = {
        "SGD": 0.1,
        "SGD with Momentum": 0.05,
        "RMSprop": 0.001,
        "Adam": 0.001,
        "AdamW": 0.001,
    }
    sensitivity_results = {name: {"lrs": [], "accs": []} for name in base_lrs}
    base_model = MLP(seed=42)
    initial_params = base_model.copy_params()
    for mult in multipliers:
        print(f"\nTesting learning rate multiplier: {mult}x")
        for name, base_lr in base_lrs.items():
            lr = base_lr * mult
            if name == "SGD":
                opt = SGD(learning_rate=lr)
            elif name == "SGD with Momentum":
                opt = SGDMomentum(learning_rate=lr, momentum=0.9)
            elif name == "RMSprop":
                opt = RMSprop(learning_rate=lr)
            elif name == "Adam":
                opt = Adam(learning_rate=lr)
            else:
                opt = AdamW(
                    learning_rate=lr, beta1=0.9, beta2=0.999, weight_decay=0.01
                )
            model = MLP(seed=42)
            model.set_params([p.copy() for p in initial_params])
            opt.reset()
            history = train(
                model,
                opt,
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=10,
                batch_size=128,
                verbose=False,
            )
            final_acc = history["val_acc"][-1]
            sensitivity_results[name]["lrs"].append(lr)
            sensitivity_results[name]["accs"].append(final_acc)
            print(f"  {name}: lr={lr:.4f}, val_acc={final_acc:.4f}")
    return sensitivity_results


def plot_sensitivity(sensitivity_results):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    colors = {
        "SGD": "#1f77b4",
        "SGD with Momentum": "#ff7f0e",
        "RMSprop": "#2ca02c",
        "Adam": "#d62728",
        "AdamW": "#17becf",
    }
    for name, data in sensitivity_results.items():
        plt.plot(
            data["lrs"],
            [a * 100 for a in data["accs"]],
            "o-",
            label=name,
            color=colors[name],
            linewidth=2,
            markersize=8,
        )
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Accuracy after 10 epochs (%)")
    plt.title("Learning Rate Sensitivity")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("lr_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()


def cosine_schedule(epoch, total_epochs, initial_lr, min_lr=1e-6):
    return min_lr + 0.5 * (initial_lr - min_lr) * (
        1 + np.cos(np.pi * epoch / max(1, total_epochs - 1))
    )


def step_decay_schedule(
    epoch, initial_lr, drop_factor=0.1, drop_epochs=None
):
    if drop_epochs is None:
        drop_epochs = [30, 60, 80]
    lr = initial_lr
    for drop_epoch in drop_epochs:
        if epoch >= drop_epoch:
            lr *= drop_factor
    return lr


def warmup_schedule(epoch, warmup_epochs, initial_lr):
    if warmup_epochs <= 0:
        return initial_lr
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    return initial_lr


class AdamW(Adam):
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    ):
        super().__init__(learning_rate, beta1, beta2, epsilon)
        self.weight_decay = weight_decay

    def step(self, model, grads):
        param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
        params = model.get_params()
        if self.m is None:
            self.m = {}
            self.v = {}
            for name in param_names:
                self.m[name] = np.zeros_like(grads[name])
                self.v[name] = np.zeros_like(grads[name])
        self.t += 1
        for i, name in enumerate(param_names):
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name] ** 2
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            adam_update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            is_weight = name.startswith("W")
            if is_weight and self.weight_decay > 0:
                weight_decay_term = self.lr * self.weight_decay * params[i]
                params[i] = params[i] - adam_update - weight_decay_term
            else:
                params[i] = params[i] - adam_update
        model.set_params(params)

    def get_name(self):
        return f"AdamW(wd={self.weight_decay})"


def load_cifar10():
    from sklearn.datasets import fetch_openml, load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    try:
        cifar = fetch_openml("CIFAR_10", version=1, as_frame=False)
        X = cifar.data / 255.0
        y = LabelEncoder().fit_transform(cifar.target)
    except Exception as e:
        # Offline-safe fallback: upsample sklearn digits to 32x32 and repeat
        # across RGB channels so the feature shape matches CIFAR-10.
        print(
            f"  OpenML unavailable ({type(e).__name__}); using "
            "sklearn digits upsampled to 32x32x3 for offline smoke testing"
        )
        digits = load_digits()
        X = digits.data.reshape(-1, 8, 8).astype(np.float64) / 16.0
        X = np.kron(X, np.ones((1, 4, 4)))
        X = np.repeat(X[:, :, :, None], 3, axis=3)
        X = X.reshape(len(X), -1)
        y = digits.target
    y_onehot = np.zeros((len(y), 10))
    class_indices = np.arange(len(y))
    y_onehot[class_indices, y] = 1
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_onehot, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1111,
        random_state=42,
        stratify=np.argmax(y_temp, axis=1),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def clip_gradients(grads, max_norm=1.0):
    total_norm = 0
    for name, grad in grads.items():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        grads = {name: grad * scale for name, grad in grads.items()}
    return grads, total_norm


class LAMB(Optimizer):
    """
    LAMB: Layer-wise Adaptive Moments optimizer.

    Key idea: Scale the Adam update by the ratio of parameter norm
    to update norm, computed per-layer. This allows stable training
    with very large batch sizes.

    Reference: You et al., "Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes" (2019)
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-6, weight_decay=0.01):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, model, grads):
        """Apply LAMB update with layer-wise trust ratio."""
        param_names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        params = model.get_params()

        if self.m is None:
            self.m = {name: np.zeros_like(grads[name]) for name in param_names}
            self.v = {name: np.zeros_like(grads[name]) for name in param_names}

        self.t += 1

        for i, name in enumerate(param_names):
            # Adam moment updates
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name]**2

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)

            # Adam update (before trust ratio scaling)
            adam_update = m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Add weight decay (decoupled, like AdamW)
            is_weight = name.startswith('W')
            if is_weight and self.weight_decay > 0:
                adam_update = adam_update + self.weight_decay * params[i]

            # LAMB trust ratio: scale by ||param|| / ||update||
            param_norm = np.linalg.norm(params[i])
            update_norm = np.linalg.norm(adam_update)

            if param_norm > 0 and update_norm > 0:
                trust_ratio = param_norm / update_norm
            else:
                trust_ratio = 1.0

            # Apply scaled update
            params[i] = params[i] - self.lr * trust_ratio * adam_update

        model.set_params(params)

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

    def get_name(self):
        return f"LAMB(wd={self.weight_decay})"


def convergence_analysis(model, optimizer, X_train, y_train,
                         X_val, y_val, epochs=50, batch_size=128, base_seed=42):
    """
    Track metrics relevant to convergence theory:
    - Gradient norm decay
    - Loss suboptimality gap
    - Distance from initialization
    """
    initial_params = model.copy_params()

    history = {
        'loss': [],
        'grad_norm': [],
        'param_distance': [],
        'loss_variance': []  # Within-epoch variance
    }

    for epoch in range(epochs):
        epoch_losses = []
        epoch_grad_norms = []

        epoch_seed = base_seed + epoch
        for X_batch, y_batch in create_batches(
            X_train, y_train, batch_size, epoch_seed=epoch_seed
        ):
            logits = model.forward(X_batch)
            loss = model.cross_entropy_loss(logits, y_batch)
            grads = model.backward(y_batch)

            # Track gradient norm before update
            grad_norm = sum(np.sum(g**2) for g in grads.values())
            grad_norm = np.sqrt(grad_norm)
            epoch_grad_norms.append(grad_norm)
            epoch_losses.append(loss)

            optimizer.step(model, grads)

        # Epoch-level metrics
        history['loss'].append(np.mean(epoch_losses))
        history['grad_norm'].append(np.mean(epoch_grad_norms))
        history['loss_variance'].append(np.var(epoch_losses))

        # Distance from initialization (in parameter space)
        current_params = model.get_params()
        param_dist = sum(np.sum((c - i)**2)
                        for c, i in zip(current_params, initial_params))
        history['param_distance'].append(np.sqrt(param_dist))

    return history


def plot_convergence_analysis(results_dict):
    """
    Create publication-quality convergence analysis plots.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {'SGD': '#1f77b4', 'SGD with Momentum': '#ff7f0e',
              'RMSprop': '#2ca02c', 'Adam': '#d62728', 'LAMB': '#9467bd'}

    def get_color(name):
        for key, color in colors.items():
            if key in name:
                return color
        return 'black'

    # Plot 1: Gradient norm decay (should decrease for convergence)
    ax1 = axes[0, 0]
    for name, history in results_dict.items():
        ax1.semilogy(history['grad_norm'], label=name.split('(')[0],
                     color=get_color(name), linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Gradient Norm (log scale)')
    ax1.set_title('Gradient Norm Decay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss trajectory
    ax2 = axes[0, 1]
    for name, history in results_dict.items():
        ax2.semilogy(history['loss'], label=name.split('(')[0],
                     color=get_color(name), linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss (log scale)')
    ax2.set_title('Loss Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Parameter distance from initialization
    ax3 = axes[1, 0]
    for name, history in results_dict.items():
        ax3.plot(history['param_distance'], label=name.split('(')[0],
                 color=get_color(name), linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('||theta - theta_0||')
    ax3.set_title('Distance from Initialization')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Loss variance (stability indicator)
    ax4 = axes[1, 1]
    for name, history in results_dict.items():
        ax4.semilogy(history['loss_variance'], label=name.split('(')[0],
                     color=get_color(name), linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Within-Epoch Loss Variance (log)')
    ax4.set_title('Training Stability')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def load_fashion_mnist():
    """
    Load Fashion-MNIST for more challenging evaluation.
    Same format as MNIST but harder classification task.
    """
    from sklearn.datasets import fetch_openml, load_digits
    from sklearn.model_selection import train_test_split

    print("Loading Fashion-MNIST...")
    try:
        fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
        X, y = fmnist.data, fmnist.target.astype(int)
        X = X / 255.0
    except Exception as e:
        # Offline-safe fallback for smoke tests. The published extension still
        # targets true Fashion-MNIST when network/cache access is available.
        print(
            f"  OpenML unavailable ({type(e).__name__}); using "
            "sklearn digits upsampled to 28x28 for offline smoke testing"
        )
        digits = load_digits()
        X = digits.data.reshape(-1, 8, 8).astype(np.float64) / 16.0
        X = np.kron(X, np.ones((1, 3, 3)))
        X = np.pad(X, ((0, 0), (2, 2), (2, 2)), mode="constant")
        X = X.reshape(len(X), -1)
        y = digits.target
    y_onehot = np.zeros((len(y), 10))
    y_onehot[np.arange(len(y)), y] = 1

    # Same split as MNIST
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_onehot, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, random_state=42,
        stratify=np.argmax(y_temp, axis=1))

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_matched_comparison(X_train, y_train, X_val, y_val,
                               time_budget_seconds=60):
    """
    Compare optimizers given the same compute budget.
    """
    import time

    optimizers = [
        SGD(learning_rate=0.1),
        SGDMomentum(learning_rate=0.05, momentum=0.9),
        RMSprop(learning_rate=0.001),
        Adam(learning_rate=0.001),
    ]

    base_model = MLP(seed=42)
    initial_params = base_model.copy_params()

    results = {}

    for opt in optimizers:
        model = MLP(seed=42)
        model.set_params([p.copy() for p in initial_params])
        opt.reset()

        start_time = time.time()
        epoch = 0
        history = {'val_acc': [], 'elapsed_time': []}

        while time.time() - start_time < time_budget_seconds:
            # Train one epoch
            epoch_seed = 42 + epoch
            for X_batch, y_batch in create_batches(
                X_train, y_train, 128, epoch_seed=epoch_seed
            ):
                logits = model.forward(X_batch)
                grads = model.backward(y_batch)
                opt.step(model, grads)

            # Record metrics
            val_acc = compute_accuracy(model, X_val, y_val)
            elapsed = time.time() - start_time
            history['val_acc'].append(val_acc)
            history['elapsed_time'].append(elapsed)
            epoch += 1

        history['total_epochs'] = epoch
        results[opt.get_name()] = history
        print(f"{opt.get_name()}: {epoch} epochs in {time_budget_seconds}s, "
              f"final val_acc={history['val_acc'][-1]:.4f}")

    return results


class AdaFactor(Optimizer):
    """
    AdaFactor: Memory-efficient adaptive optimizer.

    Key idea: Factorize second moments as outer product of row/column
    means. This reduces memory from O(mn) to O(m+n) per weight matrix.

    Reference: Shazeer & Stern, "Adafactor: Adaptive Learning Rates
    with Sublinear Memory Cost" (2018)

    Note: This is a simplified version for 2D weight matrices.
    Full AdaFactor handles 1D biases differently.
    """

    def __init__(self, learning_rate=None, epsilon1=1e-30, epsilon2=1e-3,
                 clip_threshold=1.0, decay_rate=-0.8, beta1=None):
        # AdaFactor can compute lr automatically; None means auto-scale
        super().__init__(learning_rate if learning_rate else 1.0)
        self.auto_lr = learning_rate is None
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta1 = beta1  # Optional first moment (like Adam)
        self.v_row = None
        self.v_col = None
        self.m = None
        self.t = 0

    def _get_rho(self):
        """Compute decay rate as function of step."""
        return min(self.decay_rate, -(self.t ** self.decay_rate))

    def step(self, model, grads):
        """Apply AdaFactor update with factorized second moments."""
        param_names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        params = model.get_params()

        if self.v_row is None:
            self.v_row = {}
            self.v_col = {}
            if self.beta1 is not None:
                self.m = {}
            for i, name in enumerate(param_names):
                shape = params[i].shape
                if len(shape) == 2:
                    self.v_row[name] = np.zeros(shape[0])
                    self.v_col[name] = np.zeros(shape[1])
                else:  # 1D (biases)
                    self.v_row[name] = np.zeros_like(params[i])
                    self.v_col[name] = None
                if self.beta1 is not None:
                    self.m[name] = np.zeros_like(grads[name])

        self.t += 1
        rho = min(0.999, (1 + self.t) ** self.decay_rate)

        for i, name in enumerate(param_names):
            g = grads[name]
            shape = g.shape

            if len(shape) == 2:
                # Factorized second moment for matrices
                g_squared = g ** 2
                # Row mean of squared gradients
                row_mean = np.mean(g_squared, axis=1)
                # Column mean of squared gradients
                col_mean = np.mean(g_squared, axis=0)

                self.v_row[name] = rho * self.v_row[name] + (1 - rho) * row_mean
                self.v_col[name] = rho * self.v_col[name] + (1 - rho) * col_mean

                # Reconstruct second moment estimate
                row_factor = self.v_row[name][:, np.newaxis]
                col_factor = self.v_col[name][np.newaxis, :]
                v_full = row_factor * col_factor / (np.mean(self.v_row[name]) + self.epsilon1)
            else:
                # Standard second moment for vectors (biases)
                self.v_row[name] = rho * self.v_row[name] + (1 - rho) * g ** 2
                v_full = self.v_row[name]

            # Compute update
            update = g / (np.sqrt(v_full) + self.epsilon1)

            # Optional first moment (momentum)
            if self.beta1 is not None:
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * update
                update = self.m[name]

            # Gradient clipping by RMS
            rms = np.sqrt(np.mean(update ** 2))
            if rms > self.clip_threshold:
                update = update * self.clip_threshold / rms

            # Learning rate (auto or specified)
            if self.auto_lr:
                lr = max(self.epsilon2, 1.0 / np.sqrt(self.t))
            else:
                lr = self.lr

            params[i] = params[i] - lr * update

        model.set_params(params)

    def reset(self):
        self.v_row = None
        self.v_col = None
        self.m = None
        self.t = 0

    def get_name(self):
        return "AdaFactor"


__all__ = [
    "AdaFactor",
    "Adam",
    "AdamW",
    "LAMB",
    "MLP",
    "Optimizer",
    "RMSprop",
    "SGD",
    "SGDMomentum",
    "clip_gradients",
    "compute_matched_comparison",
    "compute_accuracy",
    "convergence_analysis",
    "cosine_schedule",
    "create_batches",
    "learning_rate_sensitivity",
    "load_cifar10",
    "load_fashion_mnist",
    "load_mnist",
    "plot_convergence_analysis",
    "plot_results",
    "plot_sensitivity",
    "print_summary_table",
    "run_optimizer_showdown",
    "step_decay_schedule",
    "train",
    "warmup_schedule",
]
