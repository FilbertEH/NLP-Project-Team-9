"""
Evaluation and diagnostic suite for foreign exchange direction forecasting.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """Compute comprehensive directional classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'balanced_accuracy': round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    }

    if y_proba is not None:
        metrics['roc_auc'] = round(float(roc_auc_score(y_true, y_proba)), 4)
        metrics['brier_score'] = round(float(np.mean((y_proba - y_true) ** 2)), 4)
        metrics['log_loss'] = round(float(log_loss(y_true, y_proba)), 4)

    return metrics


def display_evaluation_report(results_json_path: Path):
    """Format and print JSON results."""
    with open(results_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df_res = pd.DataFrame(data)
    print('\n' + '=' * 80)
    print('TASK 2 BENCHMARK EVALUATION SUMMARY')
    print('=' * 80)
    print(df_res.to_string(index=False))
    print('=' * 80 + '\n')


if __name__ == '__main__':
    from src.common.config import load_config
    cfg = load_config()
    res_path = Path(cfg['paths']['processed_dir']) / 'model_benchmark_results.json'
    if res_path.exists():
        display_evaluation_report(res_path)
    else:
        print('No benchmark results found. Run src/models/train_models.py first.')