# Optimization for AI

Companion code for **"Optimization for AI: From Gradient Descent to Modern Optimizers"** (Book 2 of *The AI Engineer's Library*) by Dr. Vijay Raghavan.

## Repository Structure

```
Optimization-for-Machine-Learning/
├── notebooks/              # Jupyter notebooks for each chapter
│   ├── ch01_convex_optimization.ipynb
│   ├── ch02_gradient_descent.ipynb
│   ├── ch03_stochastic_optimization.ipynb
│   ├── ch04_advanced_optimizers.ipynb
│   ├── ch05_regularization.ipynb
│   ├── ch06_loss_functions.ipynb
│   ├── ch07_convergence_analysis.ipynb
│   ├── capstone_optimizer_showdown.ipynb
│   ├── from_scratch_library.ipynb
│   ├── debugging_lab.ipynb
│   ├── interview_prep.ipynb
│   └── optimizer_cheatsheet.ipynb
├── src/                    # Reusable Python modules
│   ├── optimizers/         # From-scratch optimizer implementations
│   │   ├── base.py         # BaseOptimizer with checkpoint support
│   │   ├── sgd.py          # SGD with momentum
│   │   ├── adagrad.py      # AdaGrad
│   │   ├── rmsprop.py      # RMSprop
│   │   ├── adam.py         # Adam and AdamW
│   │   └── utils.py        # Learning rate schedulers
│   ├── loss_examples.py    # PerceptualLoss, SuperResolutionNet
│   └── training_examples.py # Warmup, cosine schedule, cutout
├── tests/                  # Validation tests
├── summaries/              # Chapter summaries and quick references
├── requirements.txt        # Python dependencies
└── README.md
```

## Chapters Covered

| Chapter | Topic | Notebook |
|---------|-------|----------|
| 1 | Convex Optimization | `ch01_convex_optimization.ipynb` |
| 2 | Gradient Descent & Momentum | `ch02_gradient_descent.ipynb` |
| 3 | Stochastic Optimization | `ch03_stochastic_optimization.ipynb` |
| 4 | Advanced Optimizers | `ch04_advanced_optimizers.ipynb` |
| 5 | Regularization | `ch05_regularization.ipynb` |
| 6 | Loss Functions | `ch06_loss_functions.ipynb` |
| 7 | Convergence Analysis | `ch07_convergence_analysis.ipynb` |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/vijaygwu/Optimization-for-Machine-Learning.git
cd Optimization-for-Machine-Learning

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
- PyTorch 2.0+ (for loss_examples and some notebooks)
- NumPy, Matplotlib, Jupyter

**Note:** The `src/optimizers` module is NumPy-based and works without PyTorch. PyTorch is only required for `src/loss_examples.py` and the Chapter 6 notebook.

## Key Features

### From-Scratch Optimizers (`src/optimizers/`)

Clean, educational implementations of major optimizers:

```python
from src.optimizers import SGD, Adam, AdamW, RMSprop, AdaGrad

# All optimizers support checkpointing with stable parameter keys
optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
state = optimizer.state_dict()  # Save
optimizer.load_state_dict(state)  # Restore
```

### Loss Functions (`src/loss_examples.py`)

Production-ready loss implementations for Chapter 6:

```python
from src import PerceptualLoss, SuperResolutionNet

# VGG-based perceptual loss
perceptual_loss = PerceptualLoss(
    layers=('relu2_2', 'relu4_4'),
    weights=[1.0, 0.5]
).to(device)
```

### Training Utilities (`src/training_examples.py`)

Common training patterns:

```python
from src import warmup_schedule, cosine_schedule, cutout

# Learning rate schedules
lr = warmup_schedule(step, warmup_steps=1000, base_lr=1e-3)
lr = cosine_schedule(step, total_steps=10000, base_lr=1e-3)

# Data augmentation
augmented = cutout(image, n_holes=1, length=16)
```

## Capstone Project

The capstone (`notebooks/capstone_optimizer_showdown.ipynb`) implements SGD, Momentum, RMSprop, and Adam from scratch and benchmarks them on MNIST, reproducing key results from the book.

## Additional Resources

- **Interview Prep:** `notebooks/interview_prep.ipynb` — Common optimization interview questions
- **Debugging Lab:** `notebooks/debugging_lab.ipynb` — Diagnose training issues
- **Cheatsheet:** `notebooks/optimizer_cheatsheet.ipynb` — Quick reference for hyperparameters

## Testing

```bash
# Run validation tests
python -m pytest tests/ -v
```

## License

MIT License — See LICENSE file for details.

## Author

Dr. Vijay Raghavan  
George Washington University  
Contact: vijayrag@gwu.edu

## Related

- **Book 1:** [The Math That Powers AI](https://github.com/vijaygwu/Math-for-AI)
- **Book 2:** Optimization for AI (this repository)
- **Series:** The AI Engineer's Library
