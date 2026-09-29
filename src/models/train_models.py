"""
Task 2 Model Training & Evaluation Pipeline:
Trains Pure Time-Series and Combined (TS + NLP) Models across Chronological Splits.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

from src.common.config import load_config


def prepare_dataset_and_features():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    data_file = processed_dir / "dataset_with_nlp_features.csv"

    if not data_file.exists():
        raise SystemExit(f"{data_file} not found. Run extract_features.py first.")

    df = pd.read_csv(data_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["pct_change", "direction"]).reset_index(drop=True)

    # 1. Define Primary Directional Target
    df["target"] = (df["pct_change"] > 0).astype(int)

    # 2. Time-series autoregressive features
    df["lag_ret_1"] = df["pct_change"].shift(1)
    df["lag_ret_2"] = df["pct_change"].shift(2)
    df["lag_ret_3"] = df["pct_change"].shift(3)
    df["lag_ret_5"] = df["pct_change"].shift(5)
    df["rolling_vol_5"] = df["pct_change"].shift(1).rolling(5).std()
    df["rolling_vol_20"] = df["pct_change"].shift(1).rolling(20).std()
    df["calendar_gap"] = df["calendar_days_since_prev"].fillna(1)

    # Drop burn-in rolling rows
    df = df.dropna(subset=["rolling_vol_20"]).reset_index(drop=True)

    ts_cols = [
        "lag_ret_1", "lag_ret_2", "lag_ret_3", "lag_ret_5",
        "rolling_vol_5", "rolling_vol_20", "calendar_gap"
    ]

    nlp_cols = [
        c for c in df.columns
        if c.startswith(("lex_", "ent_", "lsa_topic_"))
        or c in ["news_exists_flag", "gdelt_tone_mean_clean", "headline_word_count", "news_intensity_per_day"]
    ]

    combined_cols = ts_cols + nlp_cols

    return df, ts_cols, nlp_cols, combined_cols


def run_experiment():
    df, ts_cols, nlp_cols, combined_cols = prepare_dataset_and_features()

    # Chronological Split (No leakage)
    train_mask = df["date"] < "2024-09-01"
    val_mask = (df["date"] >= "2024-09-01") & (df["date"] < "2025-09-01")
    test_mask = df["date"] >= "2025-09-01"

    y_train = df.loc[train_mask, "target"].values
    y_val = df.loc[val_mask, "target"].values
    y_test = df.loc[test_mask, "target"].values

    # Feature sets
    X_train_ts, X_val_ts, X_test_ts = df.loc[train_mask, ts_cols].values, df.loc[val_mask, ts_cols].values, df.loc[test_mask, ts_cols].values
    X_train_comb, X_val_comb, X_test_comb = df.loc[train_mask, combined_cols].values, df.loc[val_mask, combined_cols].values, df.loc[test_mask, combined_cols].values

    # Fit scalers strictly on training data
    scaler_ts = StandardScaler()
    X_train_ts_s = scaler_ts.fit_transform(X_train_ts)
    X_val_ts_s = scaler_ts.transform(X_val_ts)
    X_test_ts_s = scaler_ts.transform(X_test_ts)

    scaler_comb = StandardScaler()
    X_train_comb_s = scaler_comb.fit_transform(X_train_comb)
    X_val_comb_s = scaler_comb.transform(X_val_comb)
    X_test_comb_s = scaler_comb.transform(X_test_comb)

    models = {
        "0. Naive Majority Baseline": {
            "model": DummyClassifier(strategy="most_frequent"),
            "train_x": X_train_ts, "val_x": X_val_ts, "test_x": X_test_ts,
            "has_proba": False
        },
        "1. Pure TS (Logistic Regression)": {
            "model": LogisticRegression(C=0.1, random_state=42),
            "train_x": X_train_ts_s, "val_x": X_val_ts_s, "test_x": X_test_ts_s,
            "has_proba": True
        },
        "2. Pure TS (Gradient Boosting)": {
            "model": GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42),
            "train_x": X_train_ts, "val_x": X_val_ts, "test_x": X_test_ts,
            "has_proba": True
        },
        "3. Combined TS+NLP (Logistic Regression)": {
            "model": LogisticRegression(C=0.05, penalty="l2", random_state=42),
            "train_x": X_train_comb_s, "val_x": X_val_comb_s, "test_x": X_test_comb_s,
            "has_proba": True
        },
        "4. Combined TS+NLP (Gradient Boosting)": {
            "model": GradientBoostingClassifier(n_estimators=60, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42),
            "train_x": X_train_comb, "val_x": X_val_comb, "test_x": X_test_comb,
            "has_proba": True
        }
    }

    results = []
    print("=" * 80)
    print(f"MODEL BENCHMARK RESULTS (Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)})")
    print("=" * 80)

    for name, m_dict in models.items():
        clf = m_dict["model"]
        clf.fit(m_dict["train_x"], y_train)

        val_preds = clf.predict(m_dict["val_x"])
        test_preds = clf.predict(m_dict["test_x"])

        val_acc = accuracy_score(y_val, val_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds)
        test_bal_acc = balanced_accuracy_score(y_test, test_preds)

        if m_dict["has_proba"]:
            test_auc = roc_auc_score(y_test, clf.predict_proba(m_dict["test_x"])[:, 1])
        else:
            test_auc = 0.5

        results.append({
            "model": name,
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "test_f1": round(test_f1, 4),
            "test_balanced_acc": round(test_bal_acc, 4),
            "test_auc": round(test_auc, 4)
        })

        print(f"\n[{name}]")
        print(f"  Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test AUC: {test_auc:.4f}")

    res_df = pd.DataFrame(results)
    config = load_config()
    out_json = Path(config["paths"]["processed_dir"]) / "model_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nBenchmark saved to:", out_json)
    return res_df


if __name__ == "__main__":
    run_experiment()