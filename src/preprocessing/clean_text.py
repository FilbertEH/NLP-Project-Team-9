"""
This step unescapes HTML entities, removes empty headlines, and eliminates cross-site syndicated duplicate titles. It also handles scraped body sanitisation in scrape articles later
"""
import glob
import html
import json
import re
from pathlib import Path
import pandas as pd

from src.common.config import load_config
from src.common.gdelt_utils import gdelt_timestamp_to_wib, normalize_url

# Known non-content boilerplate injected by Indonesian portals
BOILERPLATE_PATTERNS = [
    re.compile(r"^Reporter:.*\|\s*Editor:.*$"),          # Kontan bylines
    re.compile(r"^Cek Berita dan Artikel yang lain di"),  # Bisnis follow-us prompt
    re.compile(r"^Baca Juga$"),                          # Embedded link titles
    re.compile(r"^Nyaman tanpa iklan\."),                 # Bisnis subscription prompt
]
MIN_BODY_CHARS = 200


def clean_title(raw_title: str):
    if not raw_title:
        return None
    title = html.unescape(raw_title).strip()
    return title or None


def normalize_title_key(title: str) -> str:
    return re.sub(r"\s+", " ", title.lower()).strip()


def clean_id_headlines(interim_dir: Path):
    path = interim_dir / "id_headlines_filtered.parquet"
    df = pd.read_parquet(path)
    before = len(df)

    df["title"] = df["title"].apply(clean_title)
    df = df[df["title"].notna()].reset_index(drop=True)
    after_empty = len(df)

    df["_title_key"] = df["title"].apply(normalize_title_key)
    df = df.sort_values("wib_datetime").drop_duplicates(subset="_title_key", keep="first")
    df = df.drop(columns=["_title_key"]).reset_index(drop=True)
    after_dedupe = len(df)

    return df, [
        {"stage": "id_headlines_raw", "count": before},
        {"stage": "id_headlines_after_empty_title_drop", "count": after_empty},
        {"stage": "id_headlines_after_cross_site_title_dedupe", "count": after_dedupe},
    ]


def strip_boilerplate(body: str) -> str:
    lines = [line.strip() for line in body.split("\n")]
    kept = [
        line for line in lines
        if line and not any(pat.search(line) for pat in BOILERPLATE_PATTERNS)
    ]
    return "\n".join(kept)


def clean_bodies(raw_dir: Path):
    articles_dir = raw_dir / "articles"
    rows = []
    if articles_dir.exists():
        for jsonl_path in sorted(articles_dir.glob("*.jsonl")):
            if jsonl_path.name.startswith("_"):
                continue
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

    if not rows:
        return pd.DataFrame(columns=["url", "body"]), [{"stage": "bodies_scraped", "count": 0}]

    df = pd.DataFrame(rows)
    before = len(df)

    df["url"] = df["url"].apply(normalize_url)
    df["body"] = df["body"].apply(lambda b: strip_boilerplate(b) if b else None)
    df = df[df["body"].notna() & (df["body"].str.len() >= MIN_BODY_CHARS)]
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    after = len(df)

    return df[["url", "body"]], [
        {"stage": "bodies_scraped", "count": before},
        {"stage": "bodies_after_boilerplate_and_length_filter", "count": after},
    ]


def append_report(processed_dir: Path, new_stages: list):
    report_path = processed_dir / "filter_report.json"
    report = {"stages": []}
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    report["stages"].extend(new_stages)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])

    id_df, id_stages = clean_id_headlines(interim_dir)
    bodies_df, body_stages = clean_bodies(raw_dir)

    id_df.to_parquet(interim_dir / "id_headlines_clean.parquet", index=False)
    bodies_df.to_parquet(interim_dir / "bodies_clean.parquet", index=False)

    append_report(processed_dir, id_stages + body_stages)
    print(f"[clean_text] id_headlines_clean: {len(id_df):,} rows")
    print(f"[clean_text] bodies_clean: {len(bodies_df):,} rows")


if __name__ == "__main__":
    main()