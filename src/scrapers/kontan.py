import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.scrapers.base import JsonlStore, PoliteSession, fetch_article_html, load_queue_for_site
from src.common.config import load_config


def parse_article(html: str):
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("meta", property="og:title")
    title = title_tag["content"].strip() if title_tag and title_tag.get("content") else None

    date_tag = soup.find("meta", property="article:published_time")
    published_at = date_tag["content"] if date_tag and date_tag.get("content") else None

    body_div = soup.find("div", class_="tmpt-desk-kon")
    body = None
    if body_div:
        paragraphs = [p.get_text(" ", strip=True) for p in body_div.find_all("p", recursive=False)]
        body = "\n".join(p for p in paragraphs if p)

    return title, published_at, body


def main():
    config = load_config()
    session = PoliteSession(config)
    store = JsonlStore("kontan", config=config)

    queue = load_queue_for_site("kontan", config)
    existing = store.load_existing_urls()
    queue = queue[~queue["url"].isin(existing)]
    print(f"[kontan] {len(queue):,} URLs to fetch ({len(existing):,} already processed)")

    for _, row in tqdm(queue.iterrows(), total=len(queue), desc="kontan"):
        url = row["url"]
        wb_ts = str(int(row["gdelt_timestamp"])) if pd.notna(row["gdelt_timestamp"]) else None
        html = fetch_article_html(session, url, wayback_timestamp=wb_ts)
        if html is None:
            store.append_failed(url, "fetch_failed")
            continue

        title, published_at, body = parse_article(html)
        if not title or not body:
            store.append_failed(url, "missing_title_or_body")
            continue

        store.append({
            "url": url,
            "title": title,
            "published_at": published_at or row["wib_date"],
            "body": body,
            "source": "kontan",
        })


if __name__ == "__main__":
    main()