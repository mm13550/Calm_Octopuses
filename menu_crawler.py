#!/usr/bin/env python3
"""
NYC Michelin restaurant menu crawler (restaurant-site crawler, not Michelin-site crawler).

What it does:
- Reads a CSV of restaurant seeds: name, homepage, borough, michelin_category, notes
- Respects robots.txt on each restaurant domain
- Crawls each restaurant site for likely menu pages and menu PDFs
- Extracts menu text from HTML and PDF files
- Saves structured output as JSONL and CSV

Recommended use:
1) Build data/seeds.csv from a lawful/public source or a manually exported Michelin list.
2) Crawl restaurant-owned sites only.

Example:
    python menu_crawler.py \
        --input data/seeds.csv \
        --output-dir out \
        --per-domain-delay 2.0 \
        --max-pages-per-site 25

CSV columns expected:
    name,homepage,borough,michelin_category,notes
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

MENU_KEYWORDS = [
    "menu", "menus", "dining", "eat", "food", "drink", "wine", "cocktail",
    "tasting", "prix fixe", "a la carte", "bar menu", "dessert", "lunch",
    "dinner", "brunch", "omakase", "beverage", "sake", "cellar", "pdf",
]

NEGATIVE_KEYWORDS = [
    "reservation", "reserve", "book", "gift card", "private dining", "careers",
    "press", "gallery", "instagram", "facebook", "twitter", "privacy", "terms",
    "contact", "about", "hotel", "events", "news", "shop",
]

TEXT_HINTS = [
    "appetizer", "appetizers", "starter", "starters", "entree", "entrées", "main course",
    "dessert", "tasting menu", "prix fixe", "chef's tasting", "vegetarian", "wine pairing",
    "$", "usd", "course", "courses",
]

TIMEOUT = 25


@dataclass
class Seed:
    name: str
    homepage: str
    borough: str = ""
    michelin_category: str = ""
    notes: str = ""


@dataclass
class MenuRecord:
    restaurant_name: str
    homepage: str
    borough: str
    michelin_category: str
    source_url: str
    source_type: str  # html_menu | pdf_menu | discovered_pdf | discovered_html
    title: str
    extracted_text: str
    http_status: int
    content_type: str


class PoliteSession:
    def __init__(self, per_domain_delay: float = 2.0):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.per_domain_delay = per_domain_delay
        self.last_hit: Dict[str, float] = defaultdict(float)
        self.robots: Dict[str, RobotFileParser] = {}

    def _base(self, url: str) -> str:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"

    def _sleep_if_needed(self, url: str) -> None:
        host = urlparse(url).netloc
        elapsed = time.time() - self.last_hit[host]
        wait = self.per_domain_delay - elapsed
        if wait > 0:
            time.sleep(wait + random.uniform(0, 0.4))
        self.last_hit[host] = time.time()

    def _load_robots(self, url: str) -> RobotFileParser:
        base = self._base(url)
        if base in self.robots:
            return self.robots[base]
        rp = RobotFileParser()
        rp.set_url(urljoin(base, "/robots.txt"))
        try:
            self._sleep_if_needed(rp.url)
            rp.read()
        except Exception:
            pass
        self.robots[base] = rp
        return rp

    def allowed(self, url: str, ua: str = USER_AGENT) -> bool:
        rp = self._load_robots(url)
        try:
            return rp.can_fetch(ua, url)
        except Exception:
            return True

    def get(self, url: str, **kwargs) -> requests.Response:
        self._sleep_if_needed(url)
        return self.session.get(url, timeout=TIMEOUT, allow_redirects=True, **kwargs)


# ---------- helpers ----------

def normalize_url(url: str) -> str:
    p = urlparse(url.strip())
    scheme = p.scheme or "https"
    netloc = p.netloc
    path = p.path or "/"
    clean = p._replace(scheme=scheme, netloc=netloc, path=path, params="", query="", fragment="")
    return urlunparse(clean)


def same_domain(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc


def looks_like_pdf(url: str, content_type: str = "") -> bool:
    return url.lower().endswith(".pdf") or "pdf" in content_type.lower()


def menu_score(text: str, url: str) -> int:
    blob = f"{text} {url}".lower()
    score = 0
    for k in MENU_KEYWORDS:
        if k in blob:
            score += 2
    for k in NEGATIVE_KEYWORDS:
        if k in blob:
            score -= 2
    return score


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def html_to_text(html: str) -> Tuple[str, str]:
    """
    Parses an HTML string to extract its title and the main readable text.
    It targets common semantic tags (like <main>, <article>) and class names indicating menus.
    
    Args:
        html (str): The raw HTML content from the website.
        
    Returns:
        Tuple[str, str]: A tuple containing the parsed document title and the extracted text.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "form"]):
        tag.decompose()

    title = clean_text(soup.title.get_text(" ")) if soup.title else ""

    texts = []
    selectors = [
        "main", "article", ".menu", "#menu", "[class*='menu']", "[id*='menu']",
        ".content", "section",
    ]
    seen_nodes = set()
    for sel in selectors:
        for node in soup.select(sel):
            node_id = id(node)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            txt = clean_text(node.get_text(" ", strip=True))
            if len(txt) >= 120:
                texts.append(txt)

    if not texts:
        body = soup.body.get_text(" ", strip=True) if soup.body else soup.get_text(" ", strip=True)
        texts = [clean_text(body)]

    merged = "\n\n".join(t for t in texts if t)
    return title, merged


def pdf_to_text(data: bytes) -> str:
    """
    Extracts readable text from a raw PDF byte stream using PyPDF.
    
    Args:
        data (bytes): The raw bytes of the downloaded PDF file.
        
    Returns:
        str: The extracted and cleaned text from all pages of the PDF.
    """
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return clean_text("\n\n".join(pages))


def discover_links(base_url: str, html: str) -> List[str]:
    """
    Extracts and filters links from an HTML page that are likely to point to a menu.
    It resolves relative links and ensures the links belong to the same domain.
    
    Args:
        base_url (str): The URL of the current page.
        html (str): The HTML content of the page.
        
    Returns:
        List[str]: A deduplicated list of candidate URLs to crawl.
    """
    soup = BeautifulSoup(html, "html.parser")
    found = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        text = clean_text(a.get_text(" "))
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        if not same_domain(base_url, abs_url):
            continue
        score = menu_score(text, abs_url)
        if score > 0:
            found.append(abs_url)
    # keep order, dedupe
    out = []
    seen = set()
    for u in found:
        nu = normalize_url(u)
        if nu not in seen:
            seen.add(nu)
            out.append(nu)
    return out


def has_menu_like_content(text: str) -> bool:
    blob = text.lower()
    hits = sum(1 for h in TEXT_HINTS if h in blob)
    return hits >= 3 or ("$" in blob and hits >= 1)


# ---------- crawler ----------

def fetch_html(session: PoliteSession, url: str) -> Tuple[int, str, str]:
    resp = session.get(url)
    content_type = resp.headers.get("Content-Type", "")
    body = resp.text if "text/html" in content_type or "application/xhtml+xml" in content_type else ""
    return resp.status_code, content_type, body


def crawl_site_for_menus(
    session: PoliteSession,
    seed: Seed,
    max_pages_per_site: int = 25,
) -> List[MenuRecord]:
    """
    Crawls a restaurant's website to discover and extract its menus (HTML or PDF).
    It initiates a BFS crawl starting from the seed's homepage and follows relevant links.
    
    Args:
        session (PoliteSession): The requests session configured for politeness.
        seed (Seed): The restaurant seed data.
        max_pages_per_site (int): Stop crawling after checking this many pages.
        
    Returns:
        List[MenuRecord]: A list of structured menu records found on the site.
    """
    homepage = normalize_url(seed.homepage)
    results: List[MenuRecord] = []
    visited: Set[str] = set()
    queue = deque([homepage])

    while queue and len(visited) < max_pages_per_site:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        if not session.allowed(url):
            continue

        try:
            resp = session.get(url, stream=True)
        except Exception as e:
            print(f"[WARN] GET failed: {url} :: {e}", file=sys.stderr)
            continue

        status = resp.status_code
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.content

        if looks_like_pdf(url, content_type):
            try:
                text = pdf_to_text(raw)
            except Exception as e:
                print(f"[WARN] PDF parse failed: {url} :: {e}", file=sys.stderr)
                text = ""
            if text and len(text) > 80:
                results.append(MenuRecord(
                    restaurant_name=seed.name,
                    homepage=homepage,
                    borough=seed.borough,
                    michelin_category=seed.michelin_category,
                    source_url=url,
                    source_type="pdf_menu",
                    title=os.path.basename(urlparse(url).path) or "menu.pdf",
                    extracted_text=text,
                    http_status=status,
                    content_type=content_type,
                ))
            continue

        if "html" not in content_type.lower():
            continue

        html = raw.decode(resp.encoding or "utf-8", errors="ignore")
        title, text = html_to_text(html)

        # Strong signal: current page itself is a menu page.
        if menu_score(title, url) > 1 or "/menu" in url.lower() or "menus" in url.lower():
            if text and has_menu_like_content(text):
                results.append(MenuRecord(
                    restaurant_name=seed.name,
                    homepage=homepage,
                    borough=seed.borough,
                    michelin_category=seed.michelin_category,
                    source_url=url,
                    source_type="html_menu",
                    title=title,
                    extracted_text=text,
                    http_status=status,
                    content_type=content_type,
                ))

        # Discover more candidate menu URLs.
        for child in discover_links(url, html):
            if child not in visited and child not in queue:
                queue.append(child)

    # De-duplicate by URL and text prefix
    deduped: List[MenuRecord] = []
    seen_keys = set()
    for r in results:
        key = (r.source_url, r.extracted_text[:300])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(r)
    return deduped


# ---------- io ----------

def load_seeds(csv_path: str) -> List[Seed]:
    seeds: List[Seed] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            homepage = (row.get("homepage") or "").strip()
            name = (row.get("name") or "").strip()
            if not homepage or not name:
                continue
            seeds.append(Seed(
                name=name,
                homepage=homepage,
                borough=(row.get("borough") or "").strip(),
                michelin_category=(row.get("michelin_category") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            ))
    return seeds


def save_results(records: List[MenuRecord], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    jsonl_path = out / "menus.jsonl"
    csv_path = out / "menus.csv"

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for r in records:
            jf.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    fieldnames = [
        "restaurant_name", "homepage", "borough", "michelin_category", "source_url",
        "source_type", "title", "http_status", "content_type", "extracted_text",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def main() -> int:
    parser = argparse.ArgumentParser(description="Restaurant menu crawler for NYC Michelin restaurant sites.")
    parser.add_argument("--input", required=True, help="Path to seeds CSV.")
    parser.add_argument("--output-dir", default="out", help="Directory for outputs.")
    parser.add_argument("--per-domain-delay", type=float, default=2.0, help="Delay between requests to same domain.")
    parser.add_argument("--max-pages-per-site", type=int, default=25, help="Max pages to crawl per restaurant site.")
    args = parser.parse_args()

    seeds = load_seeds(args.input)
    if not seeds:
        print("No valid seeds found.", file=sys.stderr)
        return 1

    session = PoliteSession(per_domain_delay=args.per_domain_delay)
    all_records: List[MenuRecord] = []

    for idx, seed in enumerate(seeds, 1):
        print(f"[{idx}/{len(seeds)}] Crawling {seed.name} -> {seed.homepage}")
        try:
            recs = crawl_site_for_menus(
                session=session,
                seed=seed,
                max_pages_per_site=args.max_pages_per_site,
            )
            all_records.extend(recs)
            print(f"    Found {len(recs)} menu documents/pages")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[ERROR] {seed.name}: {e}", file=sys.stderr)

    save_results(all_records, args.output_dir)
    print(f"Saved {len(all_records)} menu records to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
