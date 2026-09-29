"""
Chronological dataset splitting pipeline for USD/IDR exchange rate forecasting.
Strictly enforces chronological boundaries without look-ahead bias.
"""

from pathlib import Path
import pandas as pd
from src.common.config import load_config


def split_and_export_dataset():
    config = load_config()
    processed_dir = Path(config['paths']['processed_dir'])
    project_root = processed_dir.parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    feature_file = processed_dir / 'dataset_with_nlp_features.csv'
    if not feature_file.exists():
        raise SystemExit(f'{feature_file} not found. Run src/features/extract_features.py first.')

    df = pd.read_csv(feature_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Remove rows without returns or target
    df = df.dropna(subset=['pct_change', 'direction']).reset_index(drop=True)

    # 1. Target definition: 1 if USD appreciated (IDR weakened), 0 if IDR strengthened/flat
    df['target'] = (df['pct_change'] > 0).astype(int)

    # 2. Time-series market autoregressive features
    df['lag_ret_1'] = df['pct_change'].shift(1)
    df['lag_ret_2'] = df['pct_change'].shift(2)
    df['lag_ret_3'] = df['pct_change'].shift(3)
    df['lag_ret_5'] = df['pct_change'].shift(5)
    df['rolling_vol_5'] = df['pct_change'].shift(1).rolling(5).std()
    df['rolling_vol_20'] = df['pct_change'].shift(1).rolling(20).std()
    df['calendar_gap'] = df['calendar_days_since_prev'].fillna(1)

    # Drop burn-in rolling rows
    df = df.dropna(subset=['rolling_vol_20']).reset_index(drop=True)

    # Chronological partition masks
    train_mask = df['date'] < '2024-09-01'
    val_mask = (df['date'] >= '2024-09-01') & (df['date'] < '2025-09-01')
    test_mask = df['date'] >= '2025-09-01'

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # Export to root data/ folder as required by assignment specification
    train_df.to_csv(data_dir / 'train.csv', index=False)
    val_df.to_csv(data_dir / 'val.csv', index=False)
    test_df.to_csv(data_dir / 'test.csv', index=False)

    print('Successfully generated chronological dataset partitions:')
    print(f"  - Train : {len(train_df):,} rows ({train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}) -> data/train.csv")
    print(f"  - Val   : {len(val_df):,} rows ({val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')}) -> data/val.csv")
    print(f"  - Test  : {len(test_df):,} rows ({test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')}) -> data/test.csv")


if __name__ == '__main__':
    split_and_export_dataset()