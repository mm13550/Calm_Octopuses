#!/usr/bin/env python3
"""
Resolve restaurant homepages from a CSV of restaurant names using SerpAPI.

Input CSV columns expected:
- name (required)
- borough (optional)
- michelin_category (optional)
- notes (optional)

Output CSV columns:
- name
- homepage
- borough
- michelin_category
- notes
- resolver_status
- candidate_title
- candidate_snippet
- search_query

Options:
- Resume runs from existing CSV smoothly `--resume`
- Safely pauses on HTTP 429 quota exhaustion.

Usage:
  python resolve_homepages.py \
    --input data/nyc_michelin_names_cleaned.csv \
    --output data/seeds_resolved.csv \
    --api-key YOUR_SERPAPI_KEY

Or set environment variable SERPAPI_API_KEY and omit --api-key.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urlparse

import requests

BLOCKED_HOST_KEYWORDS = {
    "guide.michelin.com",
    "michelin.com",
    "resy.com",
    "opentable.com",
    "yelp.com",
    "instagram.com",
    "facebook.com",
    "tripadvisor.com",
    "doordash.com",
    "ubereats.com",
    "grubhub.com",
    "seamless.com",
    "theinfatuation.com",
    "timeout.com",
    "ny.eater.com",
    "eater.com",
    "foursquare.com",
    "mapquest.com",
    "linkedin.com",
    "wikipedia.org",
    "tock.com",
    "toasttab.com",
    "square.site",
    "apple.com",
    "google.com",
    "maps.apple.com",
    "googleusercontent.com",
}

PREFERRED_PATHS = (
    "",
    "/",
    "/home",
    "/welcome",
    "/nyc",
    "/new-york",
)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"


class RateLimitError(Exception):
    """Raised when SerpAPI returns a 429 status code."""
    pass


@dataclass
class Row:
    name: str
    borough: str = ""
    michelin_category: str = ""
    notes: str = ""


@dataclass
class Candidate:
    homepage: str
    title: str
    snippet: str
    score: int


class SerpAPIClient:
    endpoint = "https://serpapi.com/search.json"

    def __init__(self, api_key: str, connect_timeout: float, read_timeout: float) -> None:
        self.api_key = api_key
        self.timeout = (connect_timeout, read_timeout)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def search(self, query: str) -> dict:
        """
        Executes a Google search via SerpAPI and returns the parsed JSON response.
        
        Args:
            query (str): The search query string.
            
        Returns:
            dict: The JSON payload returned from SerpAPI.
        """
        params = {
            "engine": "google",
            "q": query,
            "hl": "en",
            "gl": "us",
            "num": 10,
            "api_key": self.api_key,
        }
        resp = self.session.get(self.endpoint, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            try:
                payload = resp.json()
                msg = payload.get("error") or "HTTP 429"
            except Exception:
                msg = "HTTP 429"
            raise RateLimitError(msg)
        resp.raise_for_status()
        return resp.json()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--api-key", default=os.environ.get("SERPAPI_API_KEY", ""))
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--connect-timeout", type=float, default=8.0)
    p.add_argument("--read-timeout", type=float, default=20.0)
    p.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    p.add_argument("--resume", action="store_true", help="Resume from existing output CSV if present")
    p.add_argument("--stop-after-429", type=int, default=3, help="Stop cleanly after this many consecutive 429s")
    p.add_argument("--backoff-seconds", type=float, default=30.0, help="Initial wait time after a 429")
    return p.parse_args()


def read_rows(path: str, max_rows: int = 0) -> list[Row]:
    """
    Reads the input CSV and converts it into a list of Row data objects.
    
    Args:
        path (str): Filepath to the CSV.
        max_rows (int): Maximum limit of rows to read. 0 means all rows.
        
    Returns:
        list[Row]: The parsed restaurant rows.
    """
    rows: list[Row] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader, start=1):
            name = (r.get("name") or "").strip()
            if not name:
                continue
            rows.append(
                Row(
                    name=name,
                    borough=(r.get("borough") or "").strip(),
                    michelin_category=(r.get("michelin_category") or "").strip(),
                    notes=(r.get("notes") or "").strip(),
                )
            )
            if max_rows and i >= max_rows:
                break
    return rows


def load_existing_output(path: str) -> list[dict]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    FIELDNAMES = [
        "name", "homepage", "borough", "michelin_category", "notes",
        "resolver_status", "candidate_title", "candidate_snippet", "search_query"
    ]
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = {key: (r.get(key) or "") for key in FIELDNAMES}
            if (row.get("name") or "").strip():
                rows.append(row)
        return rows


def build_queries(row: Row) -> list[str]:
    """
    Constructs a list of search queries to robustly find the homepage.
    
    Args:
        row (Row): A Restaurant Row item containing name and borough.
        
    Returns:
        list[str]: Variations of search queries optimized for finding official websites.
    """
    borough = f" {row.borough}" if row.borough else " NYC"
    base = row.name.strip()
    return [
        f'{base}{borough} restaurant official site',
        f'{base}{borough} official website',
        f'{base}{borough} menu official site',
    ]


def normalize_homepage(url: str) -> str:
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme or "https"
        host = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        if path in ("/index.html", "/index.htm"):
            path = ""
        if path.lower() in PREFERRED_PATHS:
            path = ""
        return f"{scheme}://{host}{path or '/'}"
    except Exception:
        return url


def blocked(url: str) -> bool:
    u = url.lower()
    return any(bad in u for bad in BLOCKED_HOST_KEYWORDS)


def score_candidate(row: Row, link: str, title: str, snippet: str) -> int:
    """
    Evaluates how likely a given search result is the official restaurant homepage.
    It penalizes platforms (e.g. Yelp, OpenTable) and rewards strict name matching.
    
    Args:
        row (Row): The restaurant row data.
        link (str): The search result URL.
        title (str): The HTML title of the search result.
        snippet (str): The search result description snippet.
        
    Returns:
        int: A heuristic score; higher means a higher likelihood of it being the official site.
    """
    score = 0
    link_l = link.lower()
    title_l = title.lower()
    snippet_l = snippet.lower()
    name_l = row.name.lower()

    if blocked(link_l):
        return -999

    parsed = urlparse(link_l)
    host = parsed.netloc
    path = parsed.path or "/"

    if host and name_l.split()[0] in host:
        score += 20

    if any(ch.isdigit() for ch in host):
        score -= 3

    if path in ("", "/"):
        score += 18
    elif path.rstrip("/") in PREFERRED_PATHS:
        score += 12
    elif any(k in path for k in ["menu", "location", "about"]):
        score += 5
    else:
        score -= 3

    name_tokens = [t for t in name_l.replace("&", " ").replace("'", " ").split() if len(t) >= 3]
    matched = sum(1 for t in name_tokens if t in title_l or t in host)
    score += matched * 8

    if row.borough and row.borough.lower() in title_l + " " + snippet_l:
        score += 3

    if any(x in title_l + " " + snippet_l for x in ["official", "restaurant", "nyc", "new york"]):
        score += 2

    if any(x in link_l for x in ["/reservations", "/book", "/menu"]):
        score -= 2

    return score


def iter_organic_results(payload: dict) -> Iterable[dict]:
    for item in payload.get("organic_results", []) or []:
        yield item
    # Some responses also include knowledge graph website
    kg = payload.get("knowledge_graph") or {}
    if kg.get("website"):
        yield {
            "link": kg.get("website"),
            "title": kg.get("title") or "",
            "snippet": kg.get("description") or "",
        }


def find_best_candidate(client: SerpAPIClient, row: Row) -> tuple[Optional[Candidate], str, str]:
    """
    Executes search queries and scores the results, attempting to resolve the best homepage link.
    
    Args:
        client (SerpAPIClient): The configured SerpAPI client.
        row (Row): The restaurant to resolve.
        
    Returns:
        tuple[Optional[Candidate], str, str]: 
            The best Candidate found (or None), the resolution status string, and the query used.
    """
    last_status = "not_found"
    for query in build_queries(row):
        try:
            payload = client.search(query)
        except RateLimitError:
            raise
        except requests.Timeout:
            last_status = "error:Timeout"
            continue
        except requests.HTTPError as e:
            last_status = f"error:HTTP{getattr(e.response, 'status_code', 'X')}"
            continue
        except requests.RequestException:
            last_status = "error:RequestException"
            continue

        best: Optional[Candidate] = None
        for item in iter_organic_results(payload):
            link = (item.get("link") or "").strip()
            title = (item.get("title") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            if not link:
                continue
            score = score_candidate(row, link, title, snippet)
            if score < 0:
                continue
            cand = Candidate(
                homepage=normalize_homepage(link),
                title=title,
                snippet=snippet,
                score=score,
            )
            if best is None or cand.score > best.score:
                best = cand

        if best is not None:
            status = "resolved" if best.score >= 18 else "review_needed"
            return best, status, query

    return None, last_status, build_queries(row)[0]


def write_rows(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "name",
        "homepage",
        "borough",
        "michelin_category",
        "notes",
        "resolver_status",
        "candidate_title",
        "candidate_snippet",
        "search_query",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("Missing SerpAPI key. Pass --api-key or set SERPAPI_API_KEY.", file=sys.stderr)
        return 2

    rows = read_rows(args.input, args.max_rows)
    print(f"Loaded {len(rows)} rows from {args.input}")
    
    out_rows: list[dict] = []
    completed_names: set[str] = set()
    if args.resume:
        existing_rows = load_existing_output(args.output)
        retryable_prefixes = ("error:",)
        retryable_names = {r["name"] for r in existing_rows if (r.get("resolver_status") or "").startswith(retryable_prefixes)}
        out_rows = [r for r in existing_rows if r.get("name") not in retryable_names]
        completed_names = {r["name"] for r in out_rows if r.get("name")}
        print(f"Loaded {len(existing_rows)} existing rows from {args.output}")
        if retryable_names:
            print(f"Will retry {len(retryable_names)} rows with retryable error statuses")

    pending_rows = [r for r in rows if r.name not in completed_names]
    print(f"Pending rows to process: {len(pending_rows)}")
    print(f"Writing results to {args.output}")

    client = SerpAPIClient(args.api_key, args.connect_timeout, args.read_timeout)
    consecutive_429 = 0

    try:
        for idx, row in enumerate(pending_rows, start=1):
            global_pos = len(out_rows) + 1
            try:
                cand, status, query = find_best_candidate(client, row)
                consecutive_429 = 0
            except RateLimitError as e:
                consecutive_429 += 1
                wait_seconds = args.backoff_seconds * (2 ** (consecutive_429 - 1))
                print(f"[{global_pos}/{len(rows)}] {row.name} -> error:HTTP429 ({e})")
                if consecutive_429 >= args.stop_after_429:
                    print(
                        f"Stopped after {consecutive_429} consecutive HTTP429 responses. "
                        f"Progress has been saved to {args.output}. "
                        f"Run again later with --resume.",
                        file=sys.stderr,
                    )
                    write_rows(args.output, out_rows)
                    return 3
                print(f"Waiting {wait_seconds:.0f}s before retrying later...")
                time.sleep(wait_seconds)
                continue

            result = {
                "name": row.name,
                "homepage": cand.homepage if cand else "",
                "borough": row.borough,
                "michelin_category": row.michelin_category,
                "notes": row.notes,
                "resolver_status": status,
                "candidate_title": cand.title if cand else "",
                "candidate_snippet": cand.snippet if cand else "",
                "search_query": query,
            }
            out_rows.append(result)
            write_rows(args.output, out_rows)
            print(f"[{global_pos}/{len(rows)}] {row.name} -> {status} {result['homepage']}")
            time.sleep(max(0.0, args.delay + random.uniform(0, 0.3)))
            
    except KeyboardInterrupt:
        write_rows(args.output, out_rows)
        print(f"\nInterrupted. Progress saved to {args.output}", file=sys.stderr)
        return 130

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
