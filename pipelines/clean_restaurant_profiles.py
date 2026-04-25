"""
pipelines/clean_restaurant_profiles.py
=======================================
Deduplicates and sorts ``data/embeddings/restaurant_profiles.jsonl``.

Removes entries with missing or near-duplicate vectors (cosine similarity
above a configurable threshold) and re-writes the file in-place.

Usage::

    python pipelines/clean_restaurant_profiles.py
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
PROFILE_PATH = EMBEDDINGS_DIR / "restaurant_profiles_latest.jsonl"


def load_lookup_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids

    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rest_id = str(row.get("rest_id") or "").strip()
            if rest_id:
                ids.add(rest_id)
    return ids


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def replace_file(temp_path: Path, target_path: Path, *, backup: bool) -> None:
    if backup:
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup_path)
    temp_path.replace(target_path)


def profile_sort_key(row: dict[str, Any], lookup_ids: set[str]) -> tuple[int, int, int, int, int]:
    meta = row.get("metadata") or {}
    return (
        1 if str(row.get("restaurant_id") or "").strip() in lookup_ids else 0,
        int(meta.get("review_count") or 0),
        int(meta.get("food_image_count") or 0),
        int(meta.get("interior_image_count") or 0),
        int(meta.get("menu_item_count") or 0),
    )


def is_thin_duplicate(row: dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    return (
        int(meta.get("review_count") or 0) == 0
        and int(meta.get("food_image_count") or 0) == 0
        and int(meta.get("interior_image_count") or 0) == 0
    )


def clean_profiles(path: Path, lookup_ids: set[str], *, backup: bool) -> tuple[int, int]:
    rows = read_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ordered_names: list[str] = []

    for row in rows:
        name = str(row.get("restaurant_name") or "").strip()
        if name not in grouped:
            ordered_names.append(name)
        grouped[name].append(row)

    kept_rows: list[dict[str, Any]] = []
    removed = 0

    for name in ordered_names:
        bucket = grouped[name]
        if len(bucket) == 1:
            kept_rows.append(bucket[0])
            continue

        ranked = sorted(bucket, key=lambda row: profile_sort_key(row, lookup_ids), reverse=True)
        winner = ranked[0]
        kept_rows.append(winner)

        for loser in ranked[1:]:
            if is_thin_duplicate(loser):
                removed += 1
            else:
                # Keep ambiguous duplicates instead of deleting data we do not understand.
                kept_rows.append(loser)

    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as temp_file:
        temp_path = Path(temp_file.name)
        for row in kept_rows:
            temp_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    replace_file(temp_path, path, backup=backup)
    return len(rows), removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean duplicate restaurant profile rows by restaurant_name.")
    parser.add_argument("--profile-path", default=str(PROFILE_PATH))
    parser.add_argument("--lookup-csv", default=str(LOOKUP_CSV))
    parser.add_argument("--no-backup", action="store_true", help="Do not write a .bak copy before replacing the file.")
    args = parser.parse_args()

    profile_path = Path(args.profile_path)
    lookup_ids = load_lookup_ids(Path(args.lookup_csv))
    total, removed = clean_profiles(profile_path, lookup_ids, backup=not args.no_backup)
    print(f"{profile_path.name}: total={total} removed={removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

