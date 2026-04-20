import os
import numpy as np
import pandas as pd

from config import OUTPUT_DIR


def load_data(train_path="fraudTrain.csv", test_path="fraudTest.csv"):
    """Load dataset from local CSVs or via Kaggle API if missing."""
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Loading data from {train_path} and {test_path} ...")
        train_df = pd.read_csv(train_path, index_col=0)
        test_df = pd.read_csv(test_path, index_col=0)
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        print("Local CSV files not found – attempting kagglehub download ...")
        try:
            import kagglehub

            dataset_dir = kagglehub.dataset_download("kartik2112/fraud-detection")
            print("Downloaded dataset to:", dataset_dir)
            train_csv = os.path.join(dataset_dir, "fraudTrain.csv")
            test_csv = os.path.join(dataset_dir, "fraudTest.csv")

            if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
                raise FileNotFoundError(
                    f"fraudTrain.csv or fraudTest.csv not found under {dataset_dir}"
                )

            train_df = pd.read_csv(train_csv, index_col=0)
            test_df = pd.read_csv(test_csv, index_col=0)
            df = pd.concat([train_df, test_df], ignore_index=True)
        except Exception as exc:
            raise FileNotFoundError(
                "Dataset not found locally and kagglehub download failed.\n"
                "Please either:\n"
                "  1. Place fraudTrain.csv and fraudTest.csv in the working directory, or\n"
                "  2. Install and configure kagglehub / Kaggle access and re-run.\n"
                f"Original error: {exc}"
            )

    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df


def explore_data(df):
    """Print basic stats and save class-balance bar chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns  # noqa: F401  # kept for potential future use

    print("\n--- EDA ---")
    print(df.dtypes)
    print(df.describe(include="all").T)

    fraud_counts = df["is_fraud"].value_counts()
    fraud_pct = df["is_fraud"].mean() * 100
    print(f"\nClass balance: {fraud_counts.to_dict()}  (fraud = {fraud_pct:.2f}%)")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Legitimate", "Fraudulent"], fraud_counts.values, color=["steelblue", "salmon"])
    ax.set_title("Class Balance")
    ax.set_ylabel("Number of Transactions")
    for i, v in enumerate(fraud_counts.values):
        ax.text(i, v + 500, f"{v:,}", ha="center", fontsize=10)
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, "class_balance.png"), dpi=150)
    plt.close(fig)
    print(f"Saved class balance chart → {OUTPUT_DIR}/class_balance.png")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    legit_amt = df[df["is_fraud"] == 0]["amt"]
    fraud_amt = df[df["is_fraud"] == 1]["amt"]
    
    ax2.hist(legit_amt[legit_amt < 1000], bins=50, alpha=0.5, label='Legitimate (<=1000)', density=True, color='steelblue')
    ax2.hist(fraud_amt[fraud_amt < 1000], bins=50, alpha=0.5, label='Fraudulent (<=1000)', density=True, color='salmon')
    ax2.set_title("Transaction Amount Distribution")
    ax2.set_xlabel("Amount ($)")
    ax2.set_ylabel("Density")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "amount_distribution.png"), dpi=150)
    plt.close(fig2)
    print(f"Saved amount distribution chart → {OUTPUT_DIR}/amount_distribution.png")


def engineer_temporal_features(df):
    """Add temporal behavior features per card in a leakage-safe way."""
    print("\nEngineering temporal features ...")

    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
    df["trans_month"] = df["trans_date_trans_time"].dt.month

    ts_seconds = df["trans_date_trans_time"].astype(np.int64) // 10**9

    df["time_since_last_tx"] = (
        ts_seconds - ts_seconds.groupby(df["cc_num"]).shift(1)
    )

    def _rolling_count(group, window_str):
        group = group.sort_index()
        return (
            group.rolling(window=window_str, closed="left")
            .count()
            .values
        )

    def _rolling_mean(group, window_str):
        group = group.sort_index()
        return (
            group.rolling(window=window_str, closed="left")
            .mean()
            .values
        )

    dt_index = df.set_index("trans_date_trans_time")

    rolling_1h = (
        dt_index.groupby("cc_num")["amt"]
        .transform(lambda g: _rolling_count(g, "1h"))
    )
    rolling_24h = (
        dt_index.groupby("cc_num")["amt"]
        .transform(lambda g: _rolling_count(g, "24h"))
    )
    rolling_amt_24h = (
        dt_index.groupby("cc_num")["amt"]
        .transform(lambda g: _rolling_mean(g, "24h"))
    )

    df["rolling_tx_count_1h"] = rolling_1h.values
    df["rolling_tx_count_24h"] = rolling_24h.values
    df["rolling_amt_mean_24h"] = rolling_amt_24h.values

    print("Temporal features added.")
    return df
