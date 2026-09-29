import glob
from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
ALIGNED_DIR = PROCESSED_DIR / "articles_aligned"
FINAL_DATASET = PROCESSED_DIR / "final_dataset.csv"


def inspect_aligned_parquets():
    print("=" * 70)
    print("1. INSPECTING ALIGNED ARTICLE PARQUETS")
    print("=" * 70)

    parquet_files = sorted(glob.glob(str(ALIGNED_DIR / "articles_*.parquet")))
    if not parquet_files:
        print(f"[!] No Parquet files found in {ALIGNED_DIR}")
        return None

    print(f"Found {len(parquet_files)} partition files:")
    for f in parquet_files:
        p = Path(f)
        size_mb = p.stat().st_size / (1024 * 1024)
        df_part = pd.read_parquet(p)
        print(f"  - {p.name:<25} : {len(df_part):>7,} rows | {size_mb:>6.2f} MB")

    # Load complete dataset
    df = pd.read_parquet(ALIGNED_DIR)
    print("\nTotal Loaded Schema & Dimensions:")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory footprint: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB\n")

    print("Column Types & Non-Null Counts:")
    info_df = pd.DataFrame({
        "Dtype": df.dtypes,
        "Non-Null Count": df.notna().sum(),
        "Null %": (df.isna().mean() * 100).round(2),
    })
    print(info_df.to_string())

    print("\nCoverage & Class Distributions:")
    print("  Source Breakdown:")
    for src, cnt in df["source"].value_counts().items():
        print(f"    - {src:<15}: {cnt:>7,} ({cnt / len(df) * 100:.2f}%)")

    print("\n  Full-Text Body Availability (`has_body`):")
    for has_b, cnt in df["has_body"].value_counts().items():
        print(f"    - has_body={str(has_b):<5}: {cnt:>7,} ({cnt / len(df) * 100:.2f}%)")

    print("\n  Body Availability by Source:")
    src_body = df.groupby("source")["has_body"].value_counts().unstack(fill_value=0)
    print(src_body.to_string())

    print("\n  Alignment Cases:")
    for case, cnt in df["alignment_case"].value_counts().items():
        print(f"    - {case:<16}: {cnt:>7,} ({cnt / len(df) * 100:.2f}%)")

    print("\nTemporal Coverage:")
    print(f"  - Calendar Date Range (WIB) : {df['calendar_date_wib'].min()} to {df['calendar_date_wib'].max()}")
    print(f"  - Trading Date Range (JISDOR): {df['trading_date'].min()} to {df['trading_date'].max()}")

    print("\nSample Record with Scraped Body (Kontan):")
    kontan_with_body = df[df["has_body"] & (df["source"] == "kontan")]
    if not kontan_with_body.empty:
        sample = kontan_with_body.iloc[0]
        print(f"  Article ID : {sample['article_id']}")
        print(f"  Trading Day: {sample['trading_date']} ({sample['alignment_case']})")
        print(f"  Title      : {sample['title']}")
        print(f"  URL        : {sample['url']}")
        print(f"  Themes     : {sample['themes'][:80]}...")
        print(f"  GDELT Tone : {sample['gdelt_tone']}")
        print(f"  Body (150 chars):\n    \"{sample['body'][:150]}...\"\n")
    else:
        print("  [!] No scraped bodies available yet.\n")

    return df


def inspect_final_dataset():
    print("=" * 70)
    print("2. INSPECTING FINAL DAILY DATASET (`final_dataset.csv`)")
    print("=" * 70)

    if not FINAL_DATASET.exists():
        print(f"[!] {FINAL_DATASET} not found.")
        return

    df_daily = pd.read_csv(FINAL_DATASET)
    print(f"Shape: {df_daily.shape[0]} trading days × {df_daily.shape[1]} features\n")
    print("Summary Statistics of Target & Market Indicators:")
    stat_cols = ["jisdor_rate", "pct_change", "log_return", "n_articles", "n_with_body", "gdelt_tone_mean"]
    available_cols = [c for c in stat_cols if c in df_daily.columns]
    print(df_daily[available_cols].describe().T[["mean", "std", "min", "50%", "max"]].to_string())

    if "direction" in df_daily.columns:
        print("\nTarget Label Distribution (`direction`):")
        label_map = {1: "+1 (IDR Weakened / USD Rose)", -1: "-1 (IDR Strengthened / USD Fell)", 0: " 0 (Flat)"}
        for val, cnt in df_daily["direction"].value_counts().items():
            print(f"  {label_map.get(val, str(val)):<35}: {cnt:>5} ({cnt / len(df_daily) * 100:.2f}%)")


if __name__ == "__main__":
    inspect_aligned_parquets()
    inspect_final_dataset()