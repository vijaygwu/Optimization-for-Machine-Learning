# Optimization for AI

Companion code for **"Optimization for AI: From Gradient Descent to Modern Optimizers"** (Book 2 of *The AI Engineer's Library*) by Dr. Vijay Raghavan.

## Repository Structure

```
Optimization-for-Machine-Learning/
├── notebooks/              # Jupyter notebooks; several chapter notebooks are exercise workbooks with intentional TODOs
│   ├── ch01_convex_optimization.ipynb
│   ├── ch02_gradient_descent.ipynb
│   ├── ch03_stochastic_optimization.ipynb
│   ├── ch04_advanced_optimizers.ipynb
│   ├── ch05_loss_functions.ipynb
│   ├── ch06_regularization.ipynb
│   ├── ch07_convergence_analysis.ipynb
│   ├── capstone_optimizer_showdown.ipynb
│   ├── from_scratch_library.ipynb
│   ├── debugging_lab.ipynb
│   ├── interview_prep.ipynb
│   └── optimizer_cheatsheet.ipynb
├── src/                    # Reusable Python modules (the maintained, tested reference code)
│   ├── optimizers/         # From-scratch optimizer implementations (NumPy)
│   │   ├── base.py         # Optimizer ABC (checkpointing) + LRScheduler, StepLR, CosineAnnealingLR
│   │   ├── sgd.py          # SGD, SGDW (momentum, Nesterov, decoupled weight decay)
│   │   ├── adam.py         # Adam, AdamW, NAdam
│   │   ├── rmsprop.py      # RMSprop, RMSpropTF (centered variants)
│   │   ├── adagrad.py      # Adagrad, AdagradSparse, Adadelta
│   │   └── utils.py        # Grad clipping, LR schedules, param init, grad checks, LR finder, HP search
│   ├── loss_examples.py    # cross_entropy_manual, PerceptualLoss, SuperResolutionNet (PyTorch)
│   ├── training_examples.py # mixup, cutout, warmup/cosine/step schedules (PyTorch)
│   └── capstone_optimizer_showdown.py  # Full runnable optimizer-showdown benchmark
├── tests/                  # Validation tests
├── summaries/              # Chapter summaries and quick references
├── requirements.txt        # Python dependencies
└── README.md
```

## Chapters Covered

| Chapter | Topic | Notebook | Status |
|---------|-------|----------|--------|
| 1 | Convex Optimization | `ch01_convex_optimization.ipynb` | Exercise workbook |
| 2 | Gradient Descent and Momentum | `ch02_gradient_descent.ipynb` | Exercise workbook |
| 3 | Stochastic Optimization Methods | `ch03_stochastic_optimization.ipynb` | Exercise workbook |
| 4 | Advanced Optimizers: Adaptive Methods, Schedules, and Tuning | `ch04_advanced_optimizers.ipynb` | Exercise workbook |
| 5 | Loss Functions and Optimization Objectives | `ch05_loss_functions.ipynb` | Exercise workbook |
| 6 | Regularization and Training Stability | `ch06_regularization.ipynb` | Exercise workbook |
| 7 | Convergence Analysis | `ch07_convergence_analysis.ipynb` | Reference notebook |

## Notebook Status

- The maintained runnable reference code for the book lives in `src/` and is validated by `tests/`.
- `ch01` through `ch06`, `capstone_optimizer_showdown.ipynb`, and `interview_prep.ipynb` are exercise workbooks: they intentionally include `TODO` prompts or `NotImplementedError` placeholders for the reader to complete.
- `debugging_lab.ipynb`, `optimizer_cheatsheet.ipynb`, and other utility notebooks are supplemental references; read their opening note before assuming every cell is a turnkey script.

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
- PyTorch 2.0+ (for `src/loss_examples.py`, `src/training_examples.py`, and some notebooks)
- NumPy, SciPy, Matplotlib, Jupyter

**Note:** The `src/optimizers` module is NumPy-based and works without PyTorch. PyTorch is required for `src/loss_examples.py`, `src/training_examples.py`, and the notebook examples that import those modules.

## Key Features

### From-Scratch Optimizers (`src/optimizers/`)

Clean, educational implementations of ten optimizers — each unit-tested against its
PyTorch equivalent to machine precision, and kept in sync with the listings printed in
the book:

```python
from src.optimizers import (
    SGD, SGDW, Adam, AdamW, NAdam, RMSprop, RMSpropTF, Adagrad, Adadelta,
)

# All optimizers support checkpointing with stable parameter keys
optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
state = optimizer.state_dict()  # Save
optimizer.load_state_dict(state)  # Restore
```

The module is pure NumPy (no PyTorch dependency), so the update rules are visible and
hackable. Built-in learning-rate schedulers (`StepLR`, `CosineAnnealingLR`) and gradient
utilities (`clip_grad_norm_`, `lr_finder`, `initialize_parameters`, `check_gradients`)
live in `base.py` and `utils.py`.

### Loss Functions (`src/loss_examples.py`)

Production-ready loss implementations for Chapter 5:

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
lr = warmup_schedule(epoch=3, warmup_epochs=5, initial_lr=1e-3)
lr = cosine_schedule(epoch=10, total_epochs=100, initial_lr=1e-3)

# Data augmentation
augmented = cutout(image, mask_size=16)
```

## Capstone Project

The capstone notebook (`notebooks/capstone_optimizer_showdown.ipynb`) is an exercise workbook that mirrors the book chapter and leaves some implementation checkpoints for the reader. The fully maintained runnable benchmark lives in `src/capstone_optimizer_showdown.py`.

## Additional Resources

- **Interview Prep:** `notebooks/interview_prep.ipynb` — Exercise workbook with interview questions and coding prompts
- **Debugging Lab:** `notebooks/debugging_lab.ipynb` — Diagnose training issues
- **Cheatsheet:** `notebooks/optimizer_cheatsheet.ipynb` — Quick reference for hyperparameters

## Testing

```bash
# Run validation tests (16 tests; optimizers, training, and loss examples)
python -m pytest tests/ -v
```

`tests/test_optimizers.py` checks every optimizer against its PyTorch reference;
`validate_training_examples.py` and `validate_loss_examples.py` exercise the schedule
and loss modules. `validate_runtime_stack.py` additionally requires `torchvision`
(used only for the super-resolution data pipeline) and is skipped if it is not installed.

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
