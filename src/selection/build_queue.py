"""
This script loads all raw Parquets, eliminates duplicate URLs, removes non-economic desks, filters for geopolitical themes, localises dates to WIB, and caps articles deterministically
"""

import glob
import hashlib
import json
from pathlib import Path
import pandas as pd

from src.common.config import load_config
from src.common.gdelt_utils import (
    gdelt_timestamp_to_wib,
    matches_theme_prefixes,
    normalize_url,
    section_for_url,
    site_for_url,
)

def load_raw_id_headlines(raw_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(raw_dir / 'gdelt_id_*.parquet')))
    if not paths:
        raise SystemExit(f'No gdelt_id_*.parquet files found in {raw_dir}')
    print(f'Loading {len(paths)} raw parquet files...')
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

def sort_key(url: str) -> str:
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def build(config: dict):
    raw_dir = Path(config['paths']['raw_dir'])
    interim_dir = Path(config['paths']['interim_dir'])
    processed_dir = Path(config['paths']['processed_dir'])
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_id_headlines(raw_dir)
    report = {'stages': []}
    report['stages'].append({'stage': 'raw_pulled', 'count': len(df)})

    # 1. URL normalisation and deduplication
    df['url'] = df['url'].apply(normalize_url)
    df = df.drop_duplicates(subset='url').reset_index(drop=True)
    report['stages'].append({'stage': 'after_url_dedupe', 'count': len(df)})

    # 2. Section filtering
    df['site'] = df['url'].apply(site_for_url)
    df = df[df['site'].notna()].reset_index(drop=True)
    df['section'] = df.apply(lambda r: section_for_url(r['url'], r['site']), axis=1)

    sections_cfg = config['sections']
    df['section_ok'] = df.apply(
        lambda r: r['section'] in sections_cfg.get(r['site'], []), axis=1
    )
    dropped_sections = df[~df['section_ok']]['section'].value_counts().to_dict()
    df = df[df['section_ok']].drop(columns=['section_ok']).reset_index(drop=True)
    report['stages'].append({
        'stage': 'after_section_filter',
        'count': len(df),
        'dropped_sections_sample': dict(list(dropped_sections.items())[:20]),
    })

    # 3. Thematic prefix matching
    theme_prefixes = config['relevant_theme_prefixes']
    df['theme_ok'] = df['V2Themes'].apply(lambda t: matches_theme_prefixes(t, theme_prefixes))
    df = df[df['theme_ok']].drop(columns=['theme_ok']).reset_index(drop=True)
    report['stages'].append({'stage': 'after_theme_filter', 'count': len(df)})

    # 4. WIB localisation
    df['wib_datetime'] = df['gdelt_timestamp'].apply(gdelt_timestamp_to_wib)
    df['wib_date'] = df['wib_datetime'].apply(lambda dt: dt.date().isoformat())

    filtered_path = interim_dir / 'id_headlines_filtered.parquet'
    df.to_parquet(filtered_path, index=False)

    # 5. Deterministic daily cap for scraping queue
    cap = config['daily_cap_per_site']
    df['_sort_key'] = df['url'].apply(sort_key)
    df = df.sort_values(['site', 'wib_date', '_sort_key'])
    queue = df.groupby(['site', 'wib_date'], group_keys=False).head(cap).drop(columns=['_sort_key'])
    report['stages'].append({
        'stage': 'scrape_queue_after_daily_cap',
        'count': len(queue),
        'daily_cap_per_site': cap,
    })

    queue_path = interim_dir / 'scrape_queue.parquet'
    queue.to_parquet(queue_path, index=False)

    report_path = processed_dir / 'filter_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'Wrote {len(df):,} filtered headlines to {filtered_path}')
    print(f'Wrote {len(queue):,} queued scrape targets to {queue_path}')
    print(f'Wrote filter report to {report_path}')

if __name__ == '__main__':
    build(load_config())