"""Wrapper module that orchestrates the fraud detection pipeline.

This file holds the high-level workflow; the heavy lifting is delegated to:

- config.py      : global constants
- data_utils.py  : data loading, EDA, and temporal feature engineering
- modeling.py    : preprocessing, model definitions, training, and plots

Running ``python fraud_detection.py`` executes the full experiment.
"""

import warnings

from config import OUTPUT_DIR
from data_utils import load_data, explore_data, engineer_temporal_features
from modeling import (
    RAW_FEATURE_COLS,
    TEMPORAL_FEATURE_COLS,
    run_pipeline,
    plot_metric_comparison,
)



warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    """Run the full fraud-detection experiment pipeline."""
    # 1. Load data
    df = load_data()

    # 2. EDA
    explore_data(df)

    # 3. Engineer temporal features (leakage-safe)
    df = engineer_temporal_features(df)

    # 4. Run pipeline with RAW features
    raw_feature_cols = RAW_FEATURE_COLS
    results_raw = run_pipeline(df, raw_feature_cols, "Raw Features")

    # 5. Run pipeline with RAW + TEMPORAL features
    temporal_feature_cols = RAW_FEATURE_COLS + TEMPORAL_FEATURE_COLS
    results_temporal = run_pipeline(df, temporal_feature_cols, "Raw + Temporal Features")

    # 6. Comparison summary
    print("\n" + "=" * 60)
    print("SUMMARY — Raw vs. Temporal Feature Sets")
    print("=" * 60)
    header = f"{'Model':<30} {'Raw F1':>8} {'Temp F1':>8} {'Raw AUC':>9} {'Temp AUC':>9}"
    print(header)
    print("-" * len(header))
    for model_name in results_raw:
        r = results_raw[model_name]
        t = results_temporal[model_name]
        print(
            f"{model_name:<30} "
            f"{r['f1']:>8.4f} {t['f1']:>8.4f} "
            f"{r['roc_auc']:>9.4f} {t['roc_auc']:>9.4f}"
        )

    # 7. Save comparison charts
    plot_metric_comparison(results_raw, results_temporal)

    print(f"\nAll outputs saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
