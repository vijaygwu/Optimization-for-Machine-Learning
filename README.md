# Optimization for AI

Companion code for **"Optimization for AI: From Gradient Descent to Modern Optimizers"** by Dr. Vijay Raghavan.

Part of *The AI Engineer's Library* series.

## Structure

```
Optimization-for-Machine-Learning/
├── notebooks/          # Jupyter notebooks for each chapter
├── src/                # Reusable Python modules
├── summaries/          # Chapter summaries and quick references
├── requirements.txt    # Python dependencies
└── README.md
```

## Chapters Covered

1. **Convex Optimization** - Convex sets, convex functions, optimality conditions
2. **Gradient Descent** - Vanilla GD, momentum, Nesterov acceleration
3. **Stochastic Optimization** - SGD fundamentals, mini-batch training, variance reduction
4. **Advanced Optimizers** - Momentum, RMSprop, Adam, AdaGrad
5. **Regularization** - L1/L2 penalties, dropout, early stopping
6. **Loss Functions** - Cross-entropy, MSE, contrastive losses, custom losses
7. **Convergence Analysis** - Rate analysis, conditions for convergence

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy, Matplotlib, Jupyter

## Capstone Project

The capstone project (`notebooks/capstone_optimizer_showdown.ipynb`) implements SGD, Momentum, RMSprop, and Adam from scratch and benchmarks them on MNIST.

## License

MIT License - See LICENSE file for details.

## Author

Dr. Vijay Raghavan  
Contact: vijayrag@gwu.edu
