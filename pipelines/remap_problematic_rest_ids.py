#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

DEFAULT_MAPPING_CSV = DATA_DIR / "csv" / "problematic_rest_id_mapping.csv"
DEFAULT_FILES = [
    DATA_DIR / "extracted_menus" / "parsed_menus.json",
    DATA_DIR / "extracted_menus" / "final_parsed_menus.json",
    DATA_DIR / "extracted_bios" / "restaurant_bios_joinable.json",
    EMBEDDINGS_DIR / "menu_embeddings_latest.jsonl",
    EMBEDDINGS_DIR / "restaurant_summary_latest.jsonl",
    EMBEDDINGS_DIR / "review_embeddings_latest.jsonl",
    EMBEDDINGS_DIR / "image_embeddings_food_latest.jsonl",
    EMBEDDINGS_DIR / "image_embeddings_interior_latest.jsonl",
]

ID_KEYS = {"rest_id", "restaurant_id"}
NAME_KEYS = {"restaurant_name"}
ID_EMBEDDED_KEYS = {"doc_id", "uid", "image_uid"}


def load_mapping(path: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            source_rest_id = str(row.get("source_rest_id") or "").strip()
            action = str(row.get("action") or "").strip().lower()
            if not source_rest_id or action not in {"remap", "drop"}:
                continue
            mapping[source_rest_id] = {
                "action": action,
                "target_rest_id": str(row.get("target_rest_id") or "").strip(),
                "target_restaurant_name": str(row.get("target_restaurant_name") or "").strip(),
                "scope": str(row.get("scope") or "").strip(),
                "confidence": str(row.get("confidence") or "").strip(),
                "notes": str(row.get("notes") or "").strip(),
            }
    return mapping


def replace_file(temp_path: Path, target_path: Path, *, backup: bool) -> None:
    if backup and target_path.exists():
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup_path)
    temp_path.replace(target_path)


def transform_row(row: dict[str, Any], mapping: dict[str, dict[str, str]]) -> tuple[dict[str, Any] | None, Counter[str]]:
    stats: Counter[str] = Counter()
    updated = dict(row)

    matched_source_ids = []
    for key in ID_KEYS:
        rest_id = str(updated.get(key) or "").strip()
        if rest_id and rest_id in mapping:
            matched_source_ids.append((key, rest_id, mapping[rest_id]))

    if not matched_source_ids:
        return updated, stats

    if any(rule["action"] == "drop" for _, _, rule in matched_source_ids):
        for _, rest_id, rule in matched_source_ids:
            stats[f"{rule['action']}:{rest_id}"] += 1
        return None, stats

    for key, rest_id, rule in matched_source_ids:
        target_rest_id = rule["target_rest_id"]
        if target_rest_id and updated.get(key) != target_rest_id:
            updated[key] = target_rest_id
            stats[f"remap:{rest_id}->{target_rest_id}"] += 1
            for embedded_key in ID_EMBEDDED_KEYS:
                embedded_value = updated.get(embedded_key)
                if isinstance(embedded_value, str) and rest_id in embedded_value:
                    updated[embedded_key] = embedded_value.replace(rest_id, target_rest_id)

            target_name = rule["target_restaurant_name"]
            if target_name:
                for name_key in NAME_KEYS:
                    current_name = str(updated.get(name_key) or "").strip()
                    if not current_name or current_name == rest_id:
                        updated[name_key] = target_name

    return updated, stats


def apply_to_json_file(path: Path, mapping: dict[str, dict[str, str]], *, backup: bool) -> Counter[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")

    out_rows = []
    stats: Counter[str] = Counter()
    for row in data:
        if not isinstance(row, dict):
            out_rows.append(row)
            continue
        updated, row_stats = transform_row(row, mapping)
        stats.update(row_stats)
        if updated is not None:
            out_rows.append(updated)

    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as temp_file:
        temp_path = Path(temp_file.name)
        json.dump(out_rows, temp_file, indent=4, ensure_ascii=False)
    replace_file(temp_path, path, backup=backup)
    return stats


def apply_to_jsonl_file(path: Path, mapping: dict[str, dict[str, str]], *, backup: bool) -> Counter[str]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc

    stats: Counter[str] = Counter()
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as temp_file:
        temp_path = Path(temp_file.name)
        for row in rows:
            if not isinstance(row, dict):
                temp_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            updated, row_stats = transform_row(row, mapping)
            stats.update(row_stats)
            if updated is not None:
                temp_file.write(json.dumps(updated, ensure_ascii=False) + "\n")

    replace_file(temp_path, path, backup=backup)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply rest_id remaps/drops to JSON and JSONL derived files.")
    parser.add_argument("--mapping-csv", default=str(DEFAULT_MAPPING_CSV))
    parser.add_argument("--file", action="append", dest="files", default=[], help="Optional file(s) to update.")
    parser.add_argument("--no-backup", action="store_true", help="Do not write .bak copies before replacing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapping = load_mapping(Path(args.mapping_csv))
    files = [Path(p) for p in args.files] if args.files else list(DEFAULT_FILES)
    backup = not args.no_backup

    for path in files:
        if not path.exists():
            print(f"[skip] {path}: file not found")
            continue

        if path.suffix.lower() == ".json":
            stats = apply_to_json_file(path, mapping, backup=backup)
        elif path.suffix.lower() == ".jsonl":
            stats = apply_to_jsonl_file(path, mapping, backup=backup)
        else:
            raise ValueError(f"Unsupported file type for {path}")

        rel_path = path.relative_to(BASE_DIR)
        if stats:
            summary = ", ".join(f"{key} x{value}" for key, value in sorted(stats.items()))
        else:
            summary = "no matching ids"
        print(f"[updated] {rel_path}: {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
