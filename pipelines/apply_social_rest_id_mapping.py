"""
pipelines/apply_social_rest_id_mapping.py
==========================================
Propagates restaurant ID rewrites across the data pipeline.

This script applies a set of remap/drop rules to social reviews and images.
Use this to fix ID collisions or canonicalize restaurant IDs without 
re-running expensive scrapers or embedding generation.

Usage::

    python pipelines/apply_social_rest_id_mapping.py --apply
"""
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOOKUP_CSV = DATA_DIR / "csv" / "restaurant_lookup.csv"
MAPPING_CSV = DATA_DIR / "csv" / "social_rest_id_mapping.csv"
RAW_REVIEWS_CSV = DATA_DIR / "social_reviews.csv"
RAW_IMAGES_CSV = DATA_DIR / "social_images.csv"

VALID_ACTIONS = {"remap", "drop"}


@dataclass(frozen=True)
class MappingRule:
    source_rest_id: str
    action: str
    target_rest_id: str
    target_restaurant_name: str
    confidence: str
    notes: str


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def read_lookup_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {
            clean_text(row.get("rest_id") or row.get("restaurant_id") or row.get("place_id"))
            for row in reader
            if clean_text(row.get("rest_id") or row.get("restaurant_id") or row.get("place_id"))
        }


def read_mapping_rules(path: Path, lookup_ids: set[str]) -> list[MappingRule]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rules: list[MappingRule] = []
    seen_sources: set[str] = set()
    for row in rows:
        source_rest_id = clean_text(row.get("source_rest_id"))
        action = clean_text(row.get("action")).lower()
        target_rest_id = clean_text(row.get("target_rest_id"))
        target_restaurant_name = clean_text(row.get("target_restaurant_name"))
        confidence = clean_text(row.get("confidence"))
        notes = clean_text(row.get("notes"))

        if not source_rest_id:
            raise ValueError("Mapping CSV contains an empty source_rest_id.")
        if source_rest_id in seen_sources:
            raise ValueError(f"Duplicate source_rest_id in mapping CSV: {source_rest_id}")
        seen_sources.add(source_rest_id)

        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}' for {source_rest_id}. Expected one of {sorted(VALID_ACTIONS)}.")

        if action == "remap":
            if not target_rest_id:
                raise ValueError(f"Missing target_rest_id for remap rule {source_rest_id}.")
            if target_rest_id not in lookup_ids:
                raise ValueError(f"target_rest_id '{target_rest_id}' is not present in restaurant_lookup.csv.")
        else:
            if target_rest_id:
                raise ValueError(f"Drop rule {source_rest_id} should not specify target_rest_id.")

        rules.append(
            MappingRule(
                source_rest_id=source_rest_id,
                action=action,
                target_rest_id=target_rest_id,
                target_restaurant_name=target_restaurant_name,
                confidence=confidence,
                notes=notes,
            )
        )

    return rules


def rewrite_review_uid(uid: str, source_rest_id: str, target_rest_id: str) -> str:
    for prefix in ("g", "y"):
        source_prefix = f"{prefix}_{source_rest_id}_"
        if uid.startswith(source_prefix):
            return f"{prefix}_{target_rest_id}_{uid[len(source_prefix):]}"
    return uid


def rewrite_image_uid(image_uid: str, source_rest_id: str, target_rest_id: str) -> str:
    for prefix in ("img_g", "img_y"):
        source_prefix = f"{prefix}_{source_rest_id}_"
        if image_uid.startswith(source_prefix):
            return f"{prefix}_{target_rest_id}_{image_uid[len(source_prefix):]}"
    return image_uid


def replace_file(temp_path: Path, target_path: Path, *, backup: bool) -> None:
    if backup and target_path.exists():
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup_path)
    temp_path.replace(target_path)


def apply_to_reviews(
    path: Path,
    rules_by_source: dict[str, MappingRule],
    *,
    apply_changes: bool,
    backup: bool,
) -> tuple[int, int, int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    remapped = 0
    dropped = 0
    out_rows: list[dict[str, Any]] = []

    for row in rows:
        source_rest_id = clean_text(row.get("rest_id"))
        rule = rules_by_source.get(source_rest_id)
        if rule is None:
            out_rows.append(row)
            continue

        if rule.action == "drop":
            dropped += 1
            continue

        updated = dict(row)
        updated["rest_id"] = rule.target_rest_id
        updated["uid"] = rewrite_review_uid(clean_text(row.get("uid")), rule.source_rest_id, rule.target_rest_id)
        out_rows.append(updated)
        remapped += 1

    if apply_changes:
        with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as temp_file:
            temp_path = Path(temp_file.name)
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        replace_file(temp_path, path, backup=backup)

    return len(rows), remapped, dropped


def apply_to_images(
    path: Path,
    rules_by_source: dict[str, MappingRule],
    *,
    apply_changes: bool,
    backup: bool,
) -> tuple[int, int, int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    remapped = 0
    dropped = 0
    out_rows: list[dict[str, Any]] = []

    for row in rows:
        source_rest_id = clean_text(row.get("rest_id"))
        rule = rules_by_source.get(source_rest_id)
        if rule is None:
            out_rows.append(row)
            continue

        if rule.action == "drop":
            dropped += 1
            continue

        updated = dict(row)
        updated["rest_id"] = rule.target_rest_id
        updated["image_uid"] = rewrite_image_uid(
            clean_text(row.get("image_uid")),
            rule.source_rest_id,
            rule.target_rest_id,
        )
        out_rows.append(updated)
        remapped += 1

    if apply_changes:
        with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as temp_file:
            temp_path = Path(temp_file.name)
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        replace_file(temp_path, path, backup=backup)

    return len(rows), remapped, dropped


def summarize_current_counts(
    reviews_path: Path,
    images_path: Path,
    rules: list[MappingRule],
) -> list[str]:
    review_counts: dict[str, int] = {}
    image_counts: dict[str, int] = {}

    with reviews_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rest_id = clean_text(row.get("rest_id"))
            review_counts[rest_id] = review_counts.get(rest_id, 0) + 1

    with images_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rest_id = clean_text(row.get("rest_id"))
            image_counts[rest_id] = image_counts.get(rest_id, 0) + 1

    lines: list[str] = []
    for rule in rules:
        lines.append(
            f"{rule.source_rest_id}: action={rule.action} target={rule.target_rest_id or '-'} "
            f"reviews={review_counts.get(rule.source_rest_id, 0)} images={image_counts.get(rule.source_rest_id, 0)}"
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply canonical social rest_id remap/drop rules to raw social CSVs."
    )
    parser.add_argument("--lookup-csv", default=str(LOOKUP_CSV))
    parser.add_argument("--mapping-csv", default=str(MAPPING_CSV))
    parser.add_argument("--raw-reviews-csv", default=str(RAW_REVIEWS_CSV))
    parser.add_argument("--raw-images-csv", default=str(RAW_IMAGES_CSV))
    parser.add_argument("--apply", action="store_true", help="Rewrite raw CSVs in place. Default is report-only.")
    parser.add_argument("--no-backup", action="store_true", help="Do not write .bak files before replacing raw outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backup = not args.no_backup

    lookup_ids = read_lookup_ids(Path(args.lookup_csv))
    rules = read_mapping_rules(Path(args.mapping_csv), lookup_ids)
    rules_by_source = {rule.source_rest_id: rule for rule in rules}

    print(f"Loaded {len(rules)} mapping rules from {args.mapping_csv}")
    for line in summarize_current_counts(Path(args.raw_reviews_csv), Path(args.raw_images_csv), rules):
        print(line)

    review_total, review_remapped, review_dropped = apply_to_reviews(
        Path(args.raw_reviews_csv),
        rules_by_source,
        apply_changes=args.apply,
        backup=backup,
    )
    image_total, image_remapped, image_dropped = apply_to_images(
        Path(args.raw_images_csv),
        rules_by_source,
        apply_changes=args.apply,
        backup=backup,
    )

    mode = "APPLIED" if args.apply else "REPORT"
    print(f"\n[{mode}] social_reviews.csv total={review_total} remapped={review_remapped} dropped={review_dropped}")
    print(f"[{mode}] social_images.csv total={image_total} remapped={image_remapped} dropped={image_dropped}")

    if args.apply:
        print("Raw social mapping changes have been written. Derived embeddings/profiles now need refresh.")
    else:
        print("No files were modified. Re-run with --apply to write the mapping results.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

