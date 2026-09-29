"""
This module handles URL canonicalisation, section/domain detection, timezone shifting from UTC to WIB (UTC+7), and theme prefix evaluation
"""

import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

WIB = timezone(timedelta(hours=7))

DOMAIN_TO_SITE = {
    'kontan.co.id': 'kontan',
    'bisnis.com': 'bisnis',
    'cnbcindonesia.com': 'cnbc_indonesia',
}

def normalize_url(url: str) -> str:
    """Drop query parameters, fragments, trailing slashes, and /amp/ suffixes."""
    parsed = urlparse(url)
    path = re.sub(r'/amp/?$', '/', parsed.path).rstrip('/')
    return f"{parsed.scheme}://{parsed.netloc}{path}"

def site_for_url(url: str):
    host = urlparse(url).netloc.lower()
    for domain, site in DOMAIN_TO_SITE.items():
        if host == domain or host.endswith('.' + domain):
            return site
    return None

def section_for_url(url: str, site: str):
    """Extract section from subdomain (Kontan/Bisnis) or first path segment (CNBC)."""
    parsed = urlparse(url)
    host_parts = parsed.netloc.lower().split('.')

    if site in ('kontan', 'bisnis'):
        return host_parts[0] if len(host_parts) > 2 else None

    if site == 'cnbc_indonesia':
        segments = [s for s in parsed.path.split('/') if s]
        return segments[0] if segments else None

    return None

def gdelt_timestamp_to_wib(gdelt_timestamp):
    """Convert GDELT UTC integer (YYYYMMDDHHMMSS) to timezone-aware WIB datetime."""
    ts = str(int(gdelt_timestamp))
    dt_utc = datetime.strptime(ts, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(WIB)

def matches_theme_prefixes(v2themes: str, prefixes: list) -> bool:
    """Check if any GDELT theme token starts with a relevant prefix."""
    if not isinstance(v2themes, str) or not v2themes:
        return False
    tokens = [t.split(',')[0] for t in v2themes.split(';') if t]
    return any(token.startswith(prefix) for token in tokens for prefix in prefixes)