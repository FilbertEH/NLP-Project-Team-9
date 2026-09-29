"""
JISDOR rates are fixed and published at 10:00 AM WIB on Indonesian business days. News runs 24/7. To avoid look-ahead bias, each article is mapped to the first JISDOR fixing published at or after its capture timestamp

Scenario            Example (Cutoff: 10AM)      Assigned Trading Day
same_day            Tuesday 08:15               Tuesday
after_cutoff        Tuesday 14:30               Wednesday (next business day)
non_trading_day     Saturday/Holiday            Monday (next business day)
"""

import json
from pathlib import Path
import pandas as pd
from src.common.config import data_path, load_config


def fixing_times(jisdor: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    hours, minutes = (int(x) for x in cutoff.split(":"))
    fx = pd.DataFrame({"trading_date": pd.to_datetime(jisdor["date"])})
    fx["fixing_time"] = (
        fx["trading_date"] + pd.Timedelta(hours=hours, minutes=minutes)
    ).dt.tz_localize("Asia/Jakarta")
    return fx.sort_values("fixing_time").reset_index(drop=True)


def load_articles(interim_dir: Path) -> pd.DataFrame:
    headlines = pd.read_parquet(interim_dir / "id_headlines_clean.parquet")
    bodies_path = interim_dir / "bodies_clean.parquet"
    if bodies_path.exists():
        bodies = pd.read_parquet(bodies_path)
        df = headlines.merge(bodies, on="url", how="left")
    else:
        df = headlines.copy()
        df["body"] = None

    df = df.rename(columns={
        "GKGRECORDID": "article_id",
        "wib_datetime": "timestamp_wib",
        "wib_date": "calendar_date_wib",
        "site": "source",
        "V2Themes": "themes",
    })
    df["gdelt_tone"] = pd.to_numeric(df["V2Tone"].str.split(",").str[0], errors="coerce")
    df["has_body"] = df["body"].notna()
    df["themes"] = df["themes"].fillna("").map(
        lambda s: ";".join(sorted({t.split(",")[0] for t in s.split(";") if t}))
    )
    df["timestamp_wib"] = pd.to_datetime(df["timestamp_wib"]).dt.tz_convert("Asia/Jakarta")
    return df


def align(articles: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    articles = articles.sort_values("timestamp_wib").reset_index(drop=True)
    articles["timestamp_wib"] = articles["timestamp_wib"].astype(fx["fixing_time"].dtype)
    aligned = pd.merge_asof(
        articles, fx, left_on="timestamp_wib", right_on="fixing_time",
        direction="forward", allow_exact_matches=True,
    )

    trading_days = set(fx["trading_date"].dt.strftime("%Y-%m-%d"))
    aligned["trading_date"] = aligned["trading_date"].dt.strftime("%Y-%m-%d")
    same_day = aligned["calendar_date_wib"] == aligned["trading_date"]
    on_trading_day = aligned["calendar_date_wib"].isin(trading_days)

    aligned["alignment_case"] = "non_trading_day"
    aligned.loc[on_trading_day & ~same_day, "alignment_case"] = "after_cutoff"
    aligned.loc[same_day, "alignment_case"] = "same_day"
    return aligned


def main():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])

    jisdor_path = processed_dir / "usd_idr_jisdor.csv"
    if not jisdor_path.exists():
        raise SystemExit(f"{jisdor_path} not found. Run src.fx.load_jisdor first.")

    fx = fixing_times(pd.read_csv(jisdor_path), config["alignment"]["cutoff_time_wib"])
    articles = load_articles(interim_dir)
    n_loaded = len(articles)

    start = pd.Timestamp(config["date_range"]["start"], tz="Asia/Jakarta")
    articles = articles[articles["timestamp_wib"] >= start]
    n_in_range = len(articles)

    aligned = align(articles, fx)
    n_unmapped = int(aligned["trading_date"].isna().sum())
    aligned = aligned.dropna(subset=["trading_date"])

    cols = [
        "article_id", "timestamp_wib", "calendar_date_wib", "trading_date",
        "alignment_case", "source", "section", "url", "title", "body", "has_body",
        "themes", "gdelt_tone"
    ]
    aligned = aligned[cols].reset_index(drop=True)

    out_path = processed_dir / "articles_aligned"
    out_path.mkdir(parents=True, exist_ok=True)
    for old in out_path.glob("articles_*.parquet"):
        old.unlink()

    for year, part in aligned.groupby(aligned["trading_date"].str[:4]):
        part.to_parquet(out_path / f"articles_{year}.parquet", index=False, compression="zstd")

    cases = aligned["alignment_case"].value_counts().to_dict()
    report = {
        "articles_loaded": n_loaded,
        "articles_on_or_after_start": n_in_range,
        "articles_after_last_fixing_dropped": n_unmapped,
        "articles_aligned": len(aligned),
        "alignment_cases": cases,
        "cutoff_time_wib": config["alignment"]["cutoff_time_wib"],
    }
    with open(processed_dir / "alignment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[align_news] {len(aligned):,} articles aligned -> {out_path}")
    print(f"[align_news] Cases: {cases}")


if __name__ == "__main__":
    main()