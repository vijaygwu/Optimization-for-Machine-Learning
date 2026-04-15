import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

output_dir = '/Users/vraghavan/Desktop/Book/publish/images/diagrams/ch15'
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# Figure 4: Momentum Effect
# ==============================================================================
def create_momentum_figure():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create ill-conditioned quadratic contours (elongated ellipse)
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    # Ill-conditioned: eigenvalues ratio ~ 10:1
    Z = 10 * X**2 + Y**2

    # Plot contours
    levels = [0.5, 2, 5, 10, 20, 40, 80]
    contour = ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6, linewidths=0.8)
    ax.contourf(X, Y, Z, levels=levels, alpha=0.15, cmap='Blues')

    # Starting point
    start = np.array([2.5, 1.8])
    optimum = np.array([0.0, 0.0])

    # Simulate GD without momentum (zigzag pattern)
    def gradient(pos):
        return np.array([20 * pos[0], 2 * pos[1]])

    # GD without momentum - small learning rate to see zigzag
    lr_gd = 0.04
    pos_gd = start.copy()
    trajectory_gd = [pos_gd.copy()]
    for _ in range(50):
        grad = gradient(pos_gd)
        pos_gd = pos_gd - lr_gd * grad
        trajectory_gd.append(pos_gd.copy())
        if np.linalg.norm(pos_gd) < 0.05:
            break
    trajectory_gd = np.array(trajectory_gd)

    # GD with momentum (smoother convergence)
    lr_mom = 0.04
    beta = 0.9
    pos_mom = start.copy()
    velocity = np.array([0.0, 0.0])
    trajectory_mom = [pos_mom.copy()]
    for _ in range(50):
        grad = gradient(pos_mom)
        velocity = beta * velocity + lr_mom * grad
        pos_mom = pos_mom - velocity
        trajectory_mom.append(pos_mom.copy())
        if np.linalg.norm(pos_mom) < 0.05:
            break
    trajectory_mom = np.array(trajectory_mom)

    # Plot trajectories
    ax.plot(trajectory_gd[:, 0], trajectory_gd[:, 1], 'b--', linewidth=2,
            label='Gradient Descent', marker='o', markersize=3, markevery=3)
    ax.plot(trajectory_mom[:, 0], trajectory_mom[:, 1], 'r-', linewidth=2,
            label='With Momentum', marker='s', markersize=3, markevery=2)

    # Mark start and end points
    ax.plot(start[0], start[1], 'ko', markersize=12, markerfacecolor='white',
            markeredgewidth=2, zorder=10)
    ax.annotate('Start', xy=start, xytext=(start[0]+0.2, start[1]+0.2),
                fontsize=11, fontweight='bold')

    ax.plot(optimum[0], optimum[1], 'k*', markersize=15, zorder=10)
    ax.annotate('Optimum', xy=optimum, xytext=(0.15, -0.4),
                fontsize=11, fontweight='bold')

    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_title('Effect of Momentum on Gradient Descent')
    ax.legend(loc='upper right')
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1.5, 2.2)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(f'{output_dir}/fig-momentum-effect.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Figure 4 (Momentum Effect) saved.")

# ==============================================================================
# Figure 5: SGD vs GD Trajectories
# ==============================================================================
def create_sgd_vs_gd_figure():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create contours for a smooth bowl
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = 2 * X**2 + 2 * Y**2 + X * Y

    # Plot contours
    levels = [0.5, 2, 5, 10, 20, 40]
    contour = ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6, linewidths=0.8)
    ax.contourf(X, Y, Z, levels=levels, alpha=0.15, cmap='Greens')

    # Starting point
    start = np.array([2.5, 2.0])
    optimum = np.array([0.0, 0.0])

    # Gradient of the loss function
    def gradient(pos):
        return np.array([4 * pos[0] + pos[1], 4 * pos[1] + pos[0]])

    # Full-batch GD (smooth, direct path)
    np.random.seed(42)
    lr = 0.08
    pos_gd = start.copy()
    trajectory_gd = [pos_gd.copy()]
    for _ in range(30):
        grad = gradient(pos_gd)
        pos_gd = pos_gd - lr * grad
        trajectory_gd.append(pos_gd.copy())
        if np.linalg.norm(pos_gd) < 0.05:
            break
    trajectory_gd = np.array(trajectory_gd)

    # SGD (noisy, wandering path)
    lr_sgd = 0.08
    pos_sgd = start.copy()
    trajectory_sgd = [pos_sgd.copy()]
    for i in range(60):
        grad = gradient(pos_sgd)
        # Add noise to simulate stochastic gradient
        noise_scale = 0.4 * np.exp(-i/30)  # Decaying noise
        noise = np.random.randn(2) * noise_scale * np.linalg.norm(grad)
        pos_sgd = pos_sgd - lr_sgd * (grad + noise)
        trajectory_sgd.append(pos_sgd.copy())
        if np.linalg.norm(pos_sgd) < 0.08:
            break
    trajectory_sgd = np.array(trajectory_sgd)

    # Plot trajectories
    ax.plot(trajectory_gd[:, 0], trajectory_gd[:, 1], 'b-', linewidth=2.5,
            label='Full-batch GD', marker='o', markersize=4, markevery=2)
    ax.plot(trajectory_sgd[:, 0], trajectory_sgd[:, 1], 'r-', linewidth=1.5,
            label='Mini-batch SGD', alpha=0.8, marker='.', markersize=3, markevery=2)

    # Mark start and end points
    ax.plot(start[0], start[1], 'ko', markersize=12, markerfacecolor='white',
            markeredgewidth=2, zorder=10)
    ax.annotate('Start', xy=start, xytext=(start[0]+0.2, start[1]+0.2),
                fontsize=11, fontweight='bold')

    ax.plot(optimum[0], optimum[1], 'k*', markersize=15, zorder=10)
    ax.annotate('Optimum', xy=optimum, xytext=(0.15, -0.5),
                fontsize=11, fontweight='bold')

    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_title('Comparison of Full-batch GD vs Mini-batch SGD')
    ax.legend(loc='upper right')
    ax.set_xlim(-1, 3.2)
    ax.set_ylim(-1, 2.8)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(f'{output_dir}/fig-sgd-vs-gd.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Figure 5 (SGD vs GD) saved.")

# ==============================================================================
# Figure 6: Learning Rate Failure Modes
# ==============================================================================
def create_lr_failure_modes_figure():
    fig, ax = plt.subplots(figsize=(9, 6))

    iterations = np.arange(0, 101)

    # Simulate loss curves for different learning rates
    # Using simple quadratic model: loss = (w - w*)^2, gradient = 2(w - w*)
    # Update: w_new = w - lr * grad = w - lr * 2 * (w - w*) = w(1 - 2*lr) + 2*lr*w*
    # For w* = 0: w_new = w * (1 - 2*lr)
    # Loss = w^2, so loss_new = loss * (1 - 2*lr)^2

    w0 = 10.0  # Initial weight

    # Optimal LR (smooth decrease to low value) - lr = 0.25
    lr_optimal = 0.25
    loss_optimal = []
    w = w0
    for _ in iterations:
        loss_optimal.append(w**2)
        w = w * (1 - 2 * lr_optimal)
    loss_optimal = np.array(loss_optimal)

    # LR too low (very slow decrease) - lr = 0.02
    lr_low = 0.02
    loss_low = []
    w = w0
    for _ in iterations:
        loss_low.append(w**2)
        w = w * (1 - 2 * lr_low)
    loss_low = np.array(loss_low)

    # LR slightly high (oscillating but converging) - lr = 0.45
    lr_high = 0.45
    loss_oscillate = []
    w = w0
    for _ in iterations:
        loss_oscillate.append(w**2)
        w = w * (1 - 2 * lr_high)
    loss_oscillate = np.array(loss_oscillate)

    # LR too high (diverges/explodes) - lr = 0.6
    lr_diverge = 0.6
    loss_diverge = []
    w = w0
    for _ in iterations:
        loss_diverge.append(min(w**2, 500))  # Cap for visualization
        w = w * (1 - 2 * lr_diverge)
        if abs(w**2) > 500:
            w = np.sign(w) * np.sqrt(500)  # Cap the growth
    loss_diverge = np.array(loss_diverge)

    # Plot all curves
    ax.plot(iterations, loss_optimal, 'g-', linewidth=2.5,
            label=f'Optimal LR ($\\alpha$ = {lr_optimal})')
    ax.plot(iterations, loss_low, 'b-', linewidth=2.5,
            label=f'LR too low ($\\alpha$ = {lr_low})')
    ax.plot(iterations, loss_oscillate, color='orange', linestyle='-', linewidth=2.5,
            label=f'LR slightly high ($\\alpha$ = {lr_high})')
    ax.plot(iterations, loss_diverge, 'r-', linewidth=2.5,
            label=f'LR too high ($\\alpha$ = {lr_diverge})')

    # Add annotations
    ax.annotate('Slow convergence', xy=(80, loss_low[80]), xytext=(60, 45),
                fontsize=10, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1))

    ax.annotate('Fast, stable\nconvergence', xy=(30, loss_optimal[30]), xytext=(40, 25),
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=1))

    ax.annotate('Oscillating', xy=(20, loss_oscillate[20]), xytext=(35, 80),
                fontsize=10, color='orange',
                arrowprops=dict(arrowstyle='->', color='orange', lw=1))

    ax.annotate('Diverging!', xy=(30, loss_diverge[30]), xytext=(45, 400),
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Rate Failure Modes')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 150)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/fig-lr-failure-modes.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Figure 6 (Learning Rate Failure Modes) saved.")

# ==============================================================================
# Generate all figures
# ==============================================================================
if __name__ == '__main__':
    create_momentum_figure()
    create_sgd_vs_gd_figure()
    create_lr_failure_modes_figure()
    print(f"\nAll figures saved to: {output_dir}")
