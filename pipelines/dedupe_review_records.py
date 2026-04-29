"""
pipelines/dedupe_review_records.py
====================================
Deduplicates social review records and their corresponding embeddings.

This script ensures consistency between the raw ``social_reviews.csv`` and 
the vector ``review_embeddings_latest.jsonl`` by removing duplicate entries
based on (rest_id, text, rating) collisions.

Usage::

    python pipelines/dedupe_review_records.py [--restaurant-id ID]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

RAW_REVIEWS_CSV = DATA_DIR / "social_reviews.csv"
REVIEW_JSONL = EMBEDDINGS_DIR / "review_embeddings_latest.jsonl"


def should_process_restaurant(restaurant_id: str, only_restaurant_id: str | None) -> bool:
    return only_restaurant_id is None or restaurant_id == only_restaurant_id


def replace_file(temp_path: Path, target_path: Path, *, backup: bool) -> None:
    if backup:
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup_path)
    temp_path.replace(target_path)


def dedupe_social_reviews_csv(path: Path, *, only_restaurant_id: str | None, backup: bool) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    seen: set[tuple[str, str, str, str]] = set()
    kept_rows: list[dict[str, str]] = []
    removed = 0

    for row in rows:
        restaurant_id = str(row.get("rest_id") or "").strip()
        if not should_process_restaurant(restaurant_id, only_restaurant_id):
            kept_rows.append(row)
            continue

        key = (
            str(row.get("uid") or "").strip(),
            restaurant_id,
            str(row.get("text") or "").strip(),
            str(row.get("rating") or "").strip(),
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept_rows.append(row)

    with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as temp_file:
        temp_path = Path(temp_file.name)
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    replace_file(temp_path, path, backup=backup)
    return len(rows), removed


def dedupe_review_jsonl(path: Path, *, only_restaurant_id: str | None, backup: bool) -> tuple[int, int]:
    seen: set[tuple[str, str, str, str]] = set()
    total = 0
    removed = 0

    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as temp_file:
        temp_path = Path(temp_file.name)

        with path.open("r", encoding="utf-8") as src:
            for line in src:
                text = line.strip()
                if not text:
                    continue

                total += 1
                row = json.loads(text)
                restaurant_id = str(row.get("restaurant_id") or "").strip()

                if not should_process_restaurant(restaurant_id, only_restaurant_id):
                    temp_file.write(text + "\n")
                    continue

                key = (
                    str(row.get("doc_id") or "").strip(),
                    restaurant_id,
                    str(row.get("text") or "").strip(),
                    str(row.get("rating") or "").strip(),
                )
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
                temp_file.write(text + "\n")

    replace_file(temp_path, path, backup=backup)
    return total, removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate raw and embedded review records in place.")
    parser.add_argument("--raw-reviews-csv", default=str(RAW_REVIEWS_CSV))
    parser.add_argument("--review-jsonl", default=str(REVIEW_JSONL))
    parser.add_argument("--restaurant-id", default=None, help="Optional restaurant_id for targeted cleanup.")
    parser.add_argument("--no-backup", action="store_true", help="Do not write .bak copies before replacing files.")
    args = parser.parse_args()

    backup = not args.no_backup

    raw_path = Path(args.raw_reviews_csv)
    review_path = Path(args.review_jsonl)

    raw_total, raw_removed = dedupe_social_reviews_csv(
        raw_path,
        only_restaurant_id=args.restaurant_id,
        backup=backup,
    )
    review_total, review_removed = dedupe_review_jsonl(
        review_path,
        only_restaurant_id=args.restaurant_id,
        backup=backup,
    )

    print(f"social_reviews.csv: total={raw_total} removed={raw_removed}")
    print(f"review_embeddings_latest.jsonl: total={review_total} removed={review_removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

