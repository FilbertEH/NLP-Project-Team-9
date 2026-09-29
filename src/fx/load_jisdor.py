from pathlib import Path
import numpy as np
import pandas as pd
from src.common.config import PROJECT_ROOT, data_path, load_config


def read_bi_excel(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None, dtype=object)
    header_row = raw.index[
        raw.apply(lambda r: r.astype(str).str.strip().str.lower().isin(["tanggal"]).any(), axis=1)
    ][0]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(c).strip().lower() for c in raw.iloc[header_row]]
    df = df[["tanggal", "kurs"]].dropna()

    df["date"] = pd.to_datetime(df["tanggal"].astype(str), format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    bad = df["date"].isna()
    if bad.any():
        df.loc[bad, "date"] = pd.to_datetime(df.loc[bad, "tanggal"], errors="coerce")
    df["jisdor_rate"] = pd.to_numeric(df["kurs"], errors="coerce")
    return df[["date", "jisdor_rate"]].dropna()


def build_series(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = df.drop_duplicates("date").sort_values("date")
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    df["pct_change"] = df["jisdor_rate"].pct_change()
    df["log_return"] = np.log(df["jisdor_rate"]).diff()
    df["direction"] = np.sign(df["jisdor_rate"].diff()).astype("Int64")
    df["calendar_days_since_prev"] = df["date"].diff().dt.days.astype("Int64")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main():
    config = load_config()
    raw_path = PROJECT_ROOT / config["fx"]["raw_file"]
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found. Place the JISDOR Excel at this path.")

    start = pd.Timestamp(config["date_range"]["start"])
    end = pd.Timestamp(config["date_range"]["end"])
    series = build_series(read_bi_excel(raw_path), start, end)

    out_dir = data_path(config, "processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "usd_idr_jisdor.csv"
    series.to_csv(out_path, index=False)

    print(f"[load_jisdor] Wrote {len(series)} JISDOR trading days to {out_path}")


if __name__ == "__main__":
    main()