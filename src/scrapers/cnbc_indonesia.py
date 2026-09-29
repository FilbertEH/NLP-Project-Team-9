import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.scrapers.base import JsonlStore, PoliteSession, fetch_article_html, load_queue_for_site
from src.common.config import load_config


def parse_article(html: str):
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else None

    candidates = soup.find_all("div", class_="detail-text")
    body_el = max(candidates, key=lambda d: len(d.find_all("p", recursive=False)), default=None)
    body = None
    if body_el:
        paragraphs = [p.get_text(" ", strip=True) for p in body_el.find_all("p", recursive=False)]
        body = "\n".join(p for p in paragraphs if p)

    return title, body


def main():
    config = load_config()
    session = PoliteSession(config)
    store = JsonlStore("cnbc_indonesia", config=config)

    queue = load_queue_for_site("cnbc_indonesia", config)
    existing = store.load_existing_urls()
    queue = queue[~queue["url"].isin(existing)]
    print(f"[cnbc_indonesia] {len(queue):,} URLs to fetch ({len(existing):,} already processed)")

    if len(queue) == 0:
        print("[cnbc_indonesia] Queue is empty (domain unindexed in translingual GDELT). Skipping.")
        return

    for _, row in tqdm(queue.iterrows(), total=len(queue), desc="cnbc_indonesia"):
        url = row["url"]
        wb_ts = str(int(row["gdelt_timestamp"])) if pd.notna(row["gdelt_timestamp"]) else None
        html = fetch_article_html(session, url, wayback_timestamp=wb_ts)
        if html is None:
            store.append_failed(url, "fetch_failed")
            continue

        title, body = parse_article(html)
        if not title or not body:
            store.append_failed(url, "missing_title_or_body")
            continue

        store.append({
            "url": url,
            "title": title,
            "published_at": row["wib_date"],
            "body": body,
            "source": "cnbc_indonesia",
        })


if __name__ == "__main__":
    main()