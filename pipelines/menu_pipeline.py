"""
pipelines/menu_pipeline.py
==========================
Unified menu management pipeline for the Calm Octopuses project.

This script consolidates menu-related maintenance tasks into a single tool:
1. Merge: Consolidates parsed, retried, and final menus into a single ordered catalog.
2. Retry: Identifies restaurants with zero dishes and triggers a targeted re-crawl.

Usage::

    python pipelines/menu_pipeline.py merge
    python pipelines/menu_pipeline.py retry --start 120 --end 180
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Append current dir to allow importing from menu_crawler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from menu_crawler import scrape_menu_text, parse_text_to_json_with_llm, get_place_id
except ImportError:
    # Fallbacks if called from different CWD
    scrape_menu_text = parse_text_to_json_with_llm = get_place_id = None


BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "pipelines" else Path.cwd()
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted_menus"
SEEDS_CSV = DATA_DIR / "csv" / "seeds_resolved.csv"


def run_merge(args: argparse.Namespace) -> None:
    print(f"Reading restaurant order from {SEEDS_CSV}...")
    df = pd.read_csv(SEEDS_CSV)
    ordered_names = df["name"].tolist()

    all_dishes = []
    for filename in ["parsed_menus.json", "final_parsed_menus.json", "retried_menus.json"]:
        path = EXTRACTED_DIR / filename
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    all_dishes.extend(data)
                    print(f"Loaded {len(data)} dishes from {filename}.")
                except json.JSONDecodeError: pass

    grouped = {}
    for dish in all_dishes:
        name = dish.get("restaurant_name", "Unknown")
        grouped.setdefault(name, [])
        if dish.get("dish_name") not in [d.get("dish_name") for d in grouped[name]]:
            grouped[name].append(dish)

    final_menus = []
    found, missing = 0, 0
    for name in ordered_names:
        if name in grouped and grouped[name]:
            final_menus.extend(grouped[name])
            found += 1
        else: missing += 1

    out_path = EXTRACTED_DIR / "final_parsed_menus.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_menus, f, indent=4, ensure_ascii=False)

    print(f"\nSummary: {len(final_menus)} dishes, {found} restaurants found, {missing} missing.")
    print(f"Exported to: {out_path}")


def run_retry(args: argparse.Namespace) -> None:
    if scrape_menu_text is None:
        print("Error: Could not import menu_crawler functions. Ensure menu_crawler.py is in the same directory.")
        return

    df = pd.read_csv(SEEDS_CSV)
    batch = df[args.start:args.end]
    print(f"Checking missing dishes for batch: {args.start} to {args.end}")

    final_path = EXTRACTED_DIR / "final_parsed_menus.json"
    existing_names = set()
    if final_path.exists():
        with final_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                existing_names = {d.get("restaurant_name") for d in data if d.get("restaurant_name")}
            except json.JSONDecodeError: pass

    failed_rows = [(i, r) for i, r in batch.iterrows() if r.get("name") not in existing_names]
    if not failed_rows:
        print("No zero-dish restaurants found in this batch.")
        return

    print(f"\nFound {len(failed_rows)} restaurants to retry.")
    retried_menus = []
    for i, row in failed_rows:
        name, url = row.get("name"), row.get("homepage")
        print(f"Retrying [{i+1}]: {name} ({url})")
        if pd.isna(url) or not str(url).startswith("http"): continue
        
        raw_text, image_urls = scrape_menu_text(url)
        if raw_text or image_urls:
            rid = get_place_id(name, homepage=url) or f"retry_{i}"
            structured = parse_text_to_json_with_llm(name, rid, raw_text, image_urls)
            print(f"  -> Structured {len(structured)} dishes.")
            retried_menus.extend(structured)

    out_path = EXTRACTED_DIR / "retried_menus.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(retried_menus, f, indent=4, ensure_ascii=False)
    print(f"\nSaved {len(retried_menus)} new dishes to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Menu Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Merge
    subparsers.add_parser("merge", help="Merge all menu sources into final_parsed_menus.json")

    # Retry
    retry_parser = subparsers.add_parser("retry", help="Retry zero-dish restaurants.")
    retry_parser.add_argument("--start", type=int, default=120)
    retry_parser.add_argument("--end", type=int, default=180)

    args = parser.parse_args()
    if args.command == "merge": run_merge(args)
    elif args.command == "retry": run_retry(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
