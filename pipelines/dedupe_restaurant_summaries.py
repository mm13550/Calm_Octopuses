#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = BASE_DIR / "data" / "embeddings" / "restaurant_summary_latest.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def replace_file(temp_path: Path, target_path: Path, *, backup: bool) -> None:
    if backup and target_path.exists():
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup_path)
    temp_path.replace(target_path)


def sort_key(row: dict[str, Any]) -> tuple[int, int]:
    metadata = row.get("metadata") or {}
    return (
        int(metadata.get("menu_item_count") or 0),
        len(str(row.get("text") or "")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate restaurant summaries by restaurant_id.")
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH))
    parser.add_argument("--no-backup", action="store_true", help="Do not write a .bak copy before replacing the file.")
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    rows = read_jsonl(summary_path)

    best_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    removed = 0

    for row in rows:
        restaurant_id = str(row.get("restaurant_id") or "").strip()
        key = restaurant_id or str(row.get("doc_id") or "").strip()
        if key not in best_by_id:
            best_by_id[key] = row
            order.append(key)
            continue
        if sort_key(row) > sort_key(best_by_id[key]):
            best_by_id[key] = row
        removed += 1

    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(summary_path.parent)) as temp_file:
        temp_path = Path(temp_file.name)
        for key in order:
            temp_file.write(json.dumps(best_by_id[key], ensure_ascii=False) + "\n")

    replace_file(temp_path, summary_path, backup=not args.no_backup)
    print(f"{summary_path.name}: total={len(rows)} removed={removed} kept={len(best_by_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
