# Automated ML Experiment Tracker

An automated machine learning experimentation and tracking system that trains multiple machine learning models, evaluates their performance, tracks experiments using MLflow, and detects performance regressions through an automated CI/CD pipeline.

## Overview

Machine learning experimentation often involves manually training different models, changing hyperparameters, comparing evaluation metrics, and tracking results.

This project automates that workflow by providing a centralized system to:

- Train multiple machine learning models with different hyperparameters.
- Evaluate models using standard classification metrics.
- Track experiments and parameters using MLflow.
- Compare model performance through a leaderboard.
- Detect performance degradation using baseline regression testing.
- Automate the entire experimentation pipeline using GitHub Actions.

The goal is to make machine learning experimentation more reproducible, organized, and efficient.

## Key Features

### 1. Automated Model Training

The system automatically trains nine model configurations:

| Algorithm | Configurations |
|---|---|
| Logistic Regression | C = 0.1, 1, 10 |
| Multi-Layer Perceptron | Hidden layers = (50,), (100,), (50, 50) |
| Random Forest | Estimators = 50, 100, 200 |

Each model is trained using the same training dataset and evaluated using the same test dataset.

### 2. Model Evaluation

Each trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Training time

The results are organized into a leaderboard to simplify model comparison.

### 3. Experiment Tracking with MLflow

MLflow is used to track machine learning experiments.

The system records:
- Model parameters and hyperparameters.
- Evaluation metrics.
- Training time.
- Trained model artifacts.

Experiments are stored using a local SQLite-backed MLflow tracking database.

### 4. Automated Performance Regression Testing

The project implements baseline-based regression testing.

A baseline JSON file stores the expected F1-score for each model.

The regression checker compares newly generated model performance against the baseline.

**Regression calculation:**

Drop = Baseline F1-score − Current F1-score

If the performance drop exceeds the configured tolerance, the regression check fails.

This helps identify unexpected model performance degradation after code or configuration changes.

### 5. CI/CD Automation with GitHub Actions

GitHub Actions automatically executes the experimentation pipeline when changes are pushed to the main branch or a pull request is created.

The workflow performs:

1. Repository checkout.
2. Python environment setup.
3. Dependency installation.
4. Python syntax validation.
5. Automated model training.
6. Performance regression testing.
7. Experiment leaderboard artifact upload.

## System Architecture

```text
              Dataset
                 |
                 v
             Data Loader
                 |
                 v
          Train-Test Split
                 |
                 v
          Model Training
                 |
       +---------+---------+
       |         |         |
       v         v         v
   Logistic    MLP       Random
  Regression            Forest
       |         |         |
       +---------+---------+
                 |
                 v
          Model Evaluation
                 |
                 v
          MLflow Tracking
                 |
                 v
          Results Leaderboard
                 |
                 v
        Regression Checker
                 |
                 v
          GitHub Actions
                 |
          Pass / Fail
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Scikit-learn | Machine learning algorithms |
| Pandas | Dataset processing and result management |
| NumPy | Numerical operations |
| MLflow | Experiment tracking and model logging |
| Skops | Model serialization |
| SQLite | MLflow tracking backend |
| Git & GitHub | Version control |
| GitHub Actions | CI/CD automation |

## Project Structure

```text
ml-experiment-tracker/
│
├── .github/
│   └── workflows/
│       └── ml-experiments.yml
│
├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── main.py
│   └── check_regression.py
│
├── baselines/
│   └── baseline.json
│
├── results/
│   └── leaderboard.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/A-Mathiyazhagan-2004/ml-repository-tracker.git
cd ml-repository-tracker
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Default Dataset

Run the pipeline using the default Breast Cancer Wisconsin dataset provided by Scikit-learn:

```bash
python src/main.py
```

### Custom Dataset

You can also provide your own CSV dataset:

```bash
python src/main.py --data path/to/dataset.csv --target target_column
```

**Current dataset requirements:**

- Binary classification target.
- Numeric input features.
- No missing values.
- A valid target column.

The target labels in custom datasets are encoded before training.

## Viewing Experiment Results

The experiment leaderboard is generated inside the `results/` directory.

MLflow experiments can be viewed through the MLflow interface:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

The interface allows you to inspect logged experiments, model parameters, metrics, and artifacts.

## Running Regression Tests

Run:

```bash
python src/check_regression.py
```

The regression checker compares the current leaderboard against:

```text
baselines/baseline.json
```

The default tolerance is configured in `check_regression.py`.

A regression check passes when all monitored models remain within the allowed performance drop.

## Future Improvements

Planned extensions include:

- Support for multiclass classification.
- Regression model experimentation.
- Automatic missing-value handling.
- Categorical feature preprocessing.
- Train-validation-test splitting.
- Dataset-specific experiment comparison.
- Automated hyperparameter optimization.
- Additional evaluation metrics.
- Experiment visualization and reporting.

## Project Objective

This project demonstrates how machine learning experimentation can be integrated with experiment tracking, model evaluation, reproducibility, and CI/CD practices.

Rather than treating model training as an isolated task, it establishes an automated workflow for continuously evaluating and monitoring machine learning model performance.

## Author

**Mathiyazhagan A.**

GitHub: [A-Mathiyazhagan-2004](https://github.com/A-Mathiyazhagan-2004)
