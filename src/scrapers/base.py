"""
This file implements the polite HTTP session with backoff, the Wayback Machine snapshot fallback, and the append-only resumable JSONL store.
"""


import json
import random
import time
from pathlib import Path
import pandas as pd
import requests

from src.common.config import load_config

WAYBACK_SNAPSHOT_URL = "https://web.archive.org/web/{timestamp}id_/{url}"


def load_queue_for_site(site_name: str, config: dict = None) -> pd.DataFrame:
    """Load scrape queue for a single outlet sorted chronologically by WIB date."""
    config = config or load_config()
    queue_path = Path(config["paths"]["interim_dir"]) / "scrape_queue.parquet"
    if not queue_path.exists():
        raise SystemExit(f"{queue_path} not found. Run src.selection.build_queue first.")
    df = pd.read_parquet(queue_path)
    df = df[df["site"] == site_name].sort_values("wib_date")
    return df


class PoliteSession:
    """Requests session wrapper with random jitter delay and exponential backoff."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        scraping_cfg = self.config["scraping"]
        self.min_delay = scraping_cfg["min_delay_seconds"]
        self.max_delay = scraping_cfg["max_delay_seconds"]
        self.max_retries = scraping_cfg["max_retries"]
        self.timeout = scraping_cfg["timeout_seconds"]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": scraping_cfg["user_agent"]})

    def get(self, url: str, **kwargs):
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(random.uniform(self.min_delay, self.max_delay))
                resp = self.session.get(url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                backoff = self.min_delay * (2 ** attempt)
                time.sleep(backoff)
        raise last_exc


def fetch_article_html(session: PoliteSession, url: str, wayback_timestamp: str = None):
    """Fetch live HTML, falling back to Internet Archive Wayback snapshot on failure."""
    try:
        return session.get(url).text
    except requests.RequestException:
        pass

    if not wayback_timestamp:
        return None

    snapshot_url = WAYBACK_SNAPSHOT_URL.format(timestamp=wayback_timestamp, url=url)
    try:
        return session.get(snapshot_url).text
    except requests.RequestException:
        return None


class JsonlStore:
    """Append-only JSONL storage that skips already-scraped URLs on resume."""

    def __init__(self, site_name: str, config: dict = None):
        config = config or load_config()
        self.dir = Path(config["paths"]["raw_dir"]) / "articles"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{site_name}.jsonl"
        self.failed_path = self.dir / "_failed.jsonl"

    def load_existing_urls(self) -> set:
        if not self.path.exists():
            return set()
        urls = set()
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        urls.add(json.loads(line)["url"])
                    except json.JSONDecodeError:
                        continue
        return urls

    def append(self, article: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    def append_failed(self, url: str, reason: str):
        with open(self.failed_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"url": url, "reason": reason}, ensure_ascii=False) + "\n")