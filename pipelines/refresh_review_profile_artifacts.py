"""
pipelines/refresh_review_profile_artifacts.py
==============================================
Synchronizes review-derived artifacts with the canonical lookup table.

This script ensures that all review-based metadata and counts are aligned
with the current ``restaurant_lookup.csv``, fixing any stale assignments
without requiring a full re-embedding.

Usage::

    python pipelines/refresh_review_profile_artifacts.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LOOKUP_CSV = DATA_DIR / "csv" / "restaurant_lookup.csv"

REVIEW_EMBEDDINGS = EMBEDDINGS_DIR / "review_embeddings_latest.jsonl"
REVIEW_IDS_LATEST = EMBEDDINGS_DIR / "review_restaurant_ids_latest.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_lookup_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row]


def replace_file(temp_path: Path, target_path: Path, *, backup: bool) -> None:
    if backup and target_path.exists():
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup_path)
    temp_path.replace(target_path)


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, backup: bool) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as temp_file:
        temp_path = Path(temp_file.name)
        for row in rows:
            temp_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    replace_file(temp_path, path, backup=backup)


def rebuild_review_restaurant_ids(
    review_path: Path,
    lookup_path: Path,
    out_path: Path,
    *,
    backup: bool,
) -> int:
    rows = read_jsonl(review_path)
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"restaurant_name": "", "count": 0})

    for row in rows:
        restaurant_id = str(row.get("restaurant_id") or "").strip()
        if not restaurant_id:
            continue
        bucket = grouped[restaurant_id]
        bucket["restaurant_name"] = str(row.get("restaurant_name") or restaurant_id).strip()
        bucket["count"] += 1

    out_rows: list[dict[str, Any]] = []
    for lookup_row in read_lookup_rows(lookup_path):
        restaurant_id = str(
            lookup_row.get("rest_id")
            or lookup_row.get("restaurant_id")
            or lookup_row.get("place_id")
            or ""
        ).strip()
        restaurant_name = str(lookup_row.get("name") or lookup_row.get("restaurant_name") or "").strip()
        if not restaurant_id:
            continue

        bucket = grouped.get(restaurant_id, {"restaurant_name": restaurant_name, "count": 0})
        out_rows.append(
            {
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant_name or bucket["restaurant_name"] or restaurant_id,
                "review_embedding_count": bucket["count"],
            }
        )

    write_jsonl(out_path, out_rows, backup=backup)
    return len(out_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh latest review-derived JSONL artifacts from cleaned base embeddings.")
    parser.add_argument("--review-embeddings", default=str(REVIEW_EMBEDDINGS))
    parser.add_argument("--lookup-csv", default=str(LOOKUP_CSV))
    parser.add_argument("--review-ids-output", default=str(REVIEW_IDS_LATEST))
    parser.add_argument("--no-backup", action="store_true", help="Do not write .bak copies before replacing files.")
    args = parser.parse_args()

    backup = not args.no_backup

    review_count = rebuild_review_restaurant_ids(
        Path(args.review_embeddings),
        Path(args.lookup_csv),
        Path(args.review_ids_output),
        backup=backup,
    )

    output_name = Path(args.review_ids_output).name
    print(f"{output_name}: rows={review_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

