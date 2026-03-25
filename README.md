# Multiple Linear Regression from Scratch with Gradient Descent

**DATA 527: Predictive Modeling — Team Project 1** | Katie Elms, Srinila Pogalla, Siddartha Bandi

---

## Overview

This project implements **Multiple Linear Regression (MLR)** from scratch using **NumPy** — without relying on scikit-learn's regression API. The model is trained using **Gradient Descent** optimization on a two-feature dataset, with full convergence analysis across multiple learning rates. The implementation includes custom MSE computation, parameter logging, R² evaluation, and comparative visualizations.

---

## Problem Statement

Given a dataset with two independent variables (`x1`, `x2`) and a continuous target `y`, train a linear regression model by minimizing Mean Squared Error (MSE) via gradient descent. Analyze how different learning rates affect convergence speed and final model performance.

---

## Methodology

### Preprocessing
- **Min-Max Normalization** applied to features `x1` and `x2`, scaling all values to [0, 1]
- Bias term (intercept) added as a column of ones to the feature matrix

### Model: Multiple Linear Regression
```
ŷ = θ₀ + θ₁·x1 + θ₂·x2
```

### Optimization: Gradient Descent
```
θ ← θ - α × (2/m) × Xᵀ(Xθ - y)
```

| Hyperparameter | Values Tested |
|---------------|---------------|
| Learning Rate (α) | 0.001, 0.01, 0.1, user-defined |
| Iterations | User-defined (default: 1,000–1,500) |

### Evaluation Metrics
- **MSE** (Mean Squared Error) tracked per iteration
- **R²** (Coefficient of Determination) computed on full dataset

### Outputs Generated
| File | Description |
|------|-------------|
| `MLRTraining[N][lr]MSE.txt` | Per-iteration MSE log |
| `MLRModelParameters_LR{lr}_Iter{n}.txt` | Final θ parameters and metrics |
| `MLR_MSE_Iteration.png` | MSE convergence curve |
| `MLR_Actual_vs_Predicted.png` | Actual vs. predicted scatter plot |
| `MLR_Multi_MSE_Comparison_With_Main_Model.png` | Learning rate comparison chart |

---

## Repository Contents

```
Team Project/
├── README.md
├── mlr.py                                                    # Main MLR implementation script
├── Multiple-Linear-Regression-from-Scratch-with-Gradient-Descent.pdf  # Team submission report
```

---

## Setup & Usage

### Install Dependencies
```bash
pip install numpy pandas matplotlib
```

### Run the Model
```bash
python mlr.py
```

You will be prompted to enter:
- `Learning rate` (e.g., `0.01`)
- `Number of iterations` (e.g., `1500`)

The script will train the model, print final parameters, and save 3 plots automatically.

### Example Output
```
Model results:
Learning Rate: 0.01
Iterations: 1500
Final MSE: 0.0XXX
Intercept: X.XXXX
Slope x1: X.XXXX
Slope x2: X.XXXX
R-squared: 0.XXXX
```

---

## Key Results

- Lower learning rates (0.001) converge slowly but stably
- Higher learning rates (0.1) converge faster but risk overshooting
- Learning rate of 0.01 provides the best balance of speed and stability
- The MLR model achieved strong R² fit on the normalized dataset

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Visualization | Matplotlib |
| Environment | Google Colab |

---

## Authors

**Katie Elms, Srinila Pogalla, Siddartha Bandi**
DATA 527: Predictive Modeling — Team Project 1
