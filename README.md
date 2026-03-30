# Transaction Fraud Classifier

This project trains and evaluates machine learning models to detect fraudulent credit card transactions using the Kaggle **Fraud Detection** dataset.

## Project structure

- `fraud_detection.py` – main script that runs the full pipeline (entrypoint).
- `data_utils.py` – data loading, basic EDA, and temporal feature engineering.
- `modeling.py` – feature definitions, preprocessing, models, training, and plots.
- `config.py` – shared configuration (e.g., random seed, test size, output directory).
- `requirements.txt` – Python dependencies.

You only need to run `fraud_detection.py`.

## Setup

From the project root, you can optionally create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

When you run the script, it will automatically:

- Look for `fraudTrain.csv` and `fraudTest.csv` in the project directory.
- If they are not found, download the Kaggle **fraud-detection** dataset using `kagglehub`.

You do not need to download the files manually—just run the script.

## How to run

From the project root (and with the virtual environment activated, if you created one):

```bash
python3 fraud_detection.py
```

The script will:

1. Load the dataset (local CSVs or Kaggle download).
2. Run basic exploratory data analysis (EDA) and save a class balance plot.
3. Engineer temporal features.
4. Train and evaluate multiple models (Logistic Regression, Random Forest, XGBoost) on:
  - Raw features only.
  - Raw + temporal features.
5. Print metrics and a summary comparison table to the terminal.
6. Save confusion matrices, ROC curves, and comparison plots.

All figures are written to the `outputs/` directory.
