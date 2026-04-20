import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from config import OUTPUT_DIR, RANDOM_STATE, TEST_SIZE


RAW_FEATURE_COLS = [
    "amt",
    "city_pop",
    "trans_hour",
    "trans_dayofweek",
    "trans_month",
    "category",
    "gender",
    "state",
]

TEMPORAL_FEATURE_COLS = [
    "time_since_last_tx",
    "rolling_tx_count_1h",
    "rolling_tx_count_24h",
    "rolling_amt_mean_24h",
]

CATEGORICAL_COLS = ["category", "gender", "state"]
CONTINUOUS_COLS = ["amt", "city_pop", "trans_hour", "trans_dayofweek", "trans_month"]
TEMPORAL_CONTINUOUS_COLS = [
    "time_since_last_tx",
    "rolling_tx_count_1h",
    "rolling_tx_count_24h",
    "rolling_amt_mean_24h",
]


def preprocess(df, feature_cols, label_encoders=None, scalers=None, fit=True):
    """Return feature matrix plus fitted/loaded encoders and scalers."""
    data = df[feature_cols].copy()

    # 1. Fill missing temporal nulls (first transaction per card has no history)
    for col in TEMPORAL_CONTINUOUS_COLS:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    # 2. Encode string categories into integers using LabelEncoder
    cats_in_scope = [c for c in CATEGORICAL_COLS if c in feature_cols]
    if fit:
        label_encoders = {}
        for col in cats_in_scope:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
    else:
        for col in cats_in_scope:
            le = label_encoders[col]
            data[col] = data[col].astype(str).map(
                lambda x, _le=le: (
                    _le.transform([x])[0]
                    if x in _le.classes_
                    else -1
                )
            )

    # 3. Standardize continuous features (Z-score normalization)
    cont_in_scope = [
        c for c in (CONTINUOUS_COLS + TEMPORAL_CONTINUOUS_COLS) if c in feature_cols
    ]
    if fit:
        scalers = {}
        for col in cont_in_scope:
            sc = StandardScaler()
            data[col] = sc.fit_transform(data[[col]])
            scalers[col] = sc
    else:
        for col in cont_in_scope:
            sc = scalers[col]
            data[col] = sc.transform(data[[col]])

    X = data.values.astype(np.float32)
    return X, label_encoders, scalers


def get_models():
    """Return mapping of model name to unfitted estimator."""
    from sklearn.neural_network import MLPClassifier
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="saga",
            C=0.1,
            penalty="l2",
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=20,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ),
    }


def evaluate(model, X_test, y_test, model_name, feature_set_name):
    """Compute metrics, print them, and save confusion matrix plot."""
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n  [{feature_set_name}] {model_name}")
    print(f"  Accuracy={metrics['accuracy']:.4f}  Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}"
          f"  F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"],
                                zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Legitimate", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name}\n({feature_set_name})")
    fig.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    safe_fs = feature_set_name.replace(" ", "_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, f"cm_{safe_fs}_{safe_name}.png"), dpi=150)
    plt.close(fig)

    return metrics, y_proba


def plot_roc_curves(roc_data, feature_set_name):
    """Plot ROC curves for all models of a given feature set."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, (fpr, tpr, auc_val) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {feature_set_name}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    safe_fs = feature_set_name.replace(" ", "_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, f"roc_{safe_fs}.png"), dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve -> {OUTPUT_DIR}/roc_{safe_fs}.png")


def plot_metric_comparison(results_raw, results_temporal):
    """Bar chart comparing F1 and ROC-AUC across models and feature sets."""
    model_names = list(results_raw.keys())
    x = np.arange(len(model_names))
    width = 0.2

    for metric in ("f1", "roc_auc"):
        raw_vals = [results_raw[m][metric] for m in model_names]
        temp_vals = [results_temporal[m][metric] for m in model_names]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, raw_vals, width, label="Raw Features", color="steelblue")
        ax.bar(x + width / 2, temp_vals, width, label="+ Temporal Features",
               color="salmon")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.set_ylabel(metric.upper().replace("_", "-"))
        ax.set_title(f"{metric.upper().replace('_', '-')} – Raw vs. Temporal Features")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, f"comparison_{metric}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved comparison chart -> {OUTPUT_DIR}/comparison_{metric}.png")


def run_pipeline(df, feature_cols, feature_set_name):
    """Train and evaluate all models on a given feature set.

    Returns
    -------
    results : dict  {model_name: metrics_dict}
    """
    print(f"\n{'='*60}")
    print(f"Feature set: {feature_set_name}  ({len(feature_cols)} features)")
    print(f"{'='*60}")

    y = df["is_fraud"].values

    df_train, df_test, y_train, y_test = train_test_split(
        df, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, le, sc = preprocess(df_train, feature_cols, fit=True)
    X_test, _, _ = preprocess(df_test, feature_cols,
                               label_encoders=le, scalers=sc, fit=False)

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Fraud rate – train: {y_train.mean():.4f}  test: {y_test.mean():.4f}")

    models = get_models()
    results = {}
    roc_data = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name} ...")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model.fit(X_train, y_train)
        metrics, y_proba = evaluate(model, X_test, y_test, model_name, feature_set_name)
        results[model_name] = metrics

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[model_name] = (fpr, tpr, metrics["roc_auc"])

    plot_roc_curves(roc_data, feature_set_name)
    return results
