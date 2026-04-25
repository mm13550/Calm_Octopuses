"""
pipelines/refresh_social_derived_artifacts.py
==============================================
Refreshes all derived artefacts (merged JSONL files, filtered CSVs, mode counts)
that depend on the raw social scrape outputs.

Useful after a re-scrape or mapping-rule change to ensure all downstream files
are consistent without re-running the full pipeline from scratch.

Usage::

    python pipelines/refresh_social_derived_artifacts.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

RAW_REVIEWS_CSV = DATA_DIR / "social_reviews.csv"
RAW_IMAGES_CSV = DATA_DIR / "social_images.csv"
LOOKUP_CSV = DATA_DIR / "csv" / "restaurant_lookup.csv"

REVIEW_JSONL = EMBEDDINGS_DIR / "review_embeddings_latest.jsonl"
FOOD_JSONL = EMBEDDINGS_DIR / "image_embeddings_food_latest.jsonl"
INTERIOR_JSONL = EMBEDDINGS_DIR / "image_embeddings_interior_latest.jsonl"


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).lower() in {"1", "true", "t", "yes", "y"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def replace_file(temp_path: Path, target_path: Path, *, backup: bool) -> None:
    if backup and target_path.exists():
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup_path)
    temp_path.replace(target_path)


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, backup: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as temp_file:
        temp_path = Path(temp_file.name)
        for row in rows:
            temp_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    replace_file(temp_path, path, backup=backup)


def merge_jsonl_rows(
    target_path: Path,
    *,
    id_field: str,
    restaurant_ids: set[str],
    new_rows: list[dict[str, Any]],
    backup: bool,
) -> tuple[int, int]:
    existing_rows = read_jsonl(target_path)
    merged_rows = [
        row
        for row in existing_rows
        if clean_text(row.get(id_field)) not in restaurant_ids
    ]
    removed = len(existing_rows) - len(merged_rows)
    merged_rows.extend(new_rows)
    write_jsonl(target_path, merged_rows, backup=backup)
    return removed, len(merged_rows)


def filter_csv_rows(
    input_path: Path,
    output_path: Path,
    *,
    id_field: str,
    restaurant_ids: set[str],
) -> int:
    with input_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            row
            for row in reader
            if clean_text(row.get(id_field)) in restaurant_ids
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def cleaned_image_mode_counts(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    food_count = sum(
        1
        for row in rows
        if clean_text(row.get("image_category")).lower() == "food"
        and as_bool(row.get("keep_for_food_embedding"))
    )
    interior_count = sum(
        1
        for row in rows
        if clean_text(row.get("image_category")).lower() == "interior"
        and as_bool(row.get("keep_for_ambiance_embedding"))
    )
    return food_count, interior_count


def run_python_script(script_path: Path, *args: str) -> None:
    cmd = [sys.executable, str(script_path), *args]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild only the social-derived embeddings/profiles for selected restaurants."
    )
    parser.add_argument(
        "--restaurant-id",
        action="append",
        required=True,
        help="Canonical restaurant_id to refresh. Repeatable.",
    )
    parser.add_argument("--raw-reviews-csv", default=str(RAW_REVIEWS_CSV))
    parser.add_argument("--raw-images-csv", default=str(RAW_IMAGES_CSV))
    parser.add_argument("--lookup-csv", default=str(LOOKUP_CSV))
    parser.add_argument("--review-jsonl", default=str(REVIEW_JSONL))
    parser.add_argument("--food-jsonl", default=str(FOOD_JSONL))
    parser.add_argument("--interior-jsonl", default=str(INTERIOR_JSONL))
    parser.add_argument("--skip-reviews", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--no-backup", action="store_true", help="Do not write .bak copies before replacing merged artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backup = not args.no_backup
    restaurant_ids = {clean_text(rest_id) for rest_id in args.restaurant_id if clean_text(rest_id)}
    if not restaurant_ids:
        raise SystemExit("At least one non-empty --restaurant-id is required.")

    review_jsonl_path = Path(args.review_jsonl)
    food_jsonl_path = Path(args.food_jsonl)
    interior_jsonl_path = Path(args.interior_jsonl)

    with TemporaryDirectory(dir=str(DATA_DIR)) as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        if not args.skip_reviews:
            temp_reviews_csv = temp_dir / "social_reviews_target.csv"
            review_row_count = filter_csv_rows(
                Path(args.raw_reviews_csv),
                temp_reviews_csv,
                id_field="rest_id",
                restaurant_ids=restaurant_ids,
            )
            if review_row_count > 0:
                run_python_script(
                    BASE_DIR / "pipelines" / "generate_embeddings_reviews.py",
                    "--input",
                    str(temp_reviews_csv),
                    "--lookup-csv",
                    str(args.lookup_csv),
                    "--output-dir",
                    str(temp_dir),
                )
                new_review_rows = read_jsonl(temp_dir / "review_embeddings_latest.jsonl")
            else:
                new_review_rows = []

            removed, total = merge_jsonl_rows(
                review_jsonl_path,
                id_field="restaurant_id",
                restaurant_ids=restaurant_ids,
                new_rows=new_review_rows,
                backup=backup,
            )
            print(f"review_embeddings_latest.jsonl merged: removed={removed} total={total}")

        if not args.skip_images:
            temp_images_csv = temp_dir / "social_images_target.csv"
            image_row_count = filter_csv_rows(
                Path(args.raw_images_csv),
                temp_images_csv,
                id_field="rest_id",
                restaurant_ids=restaurant_ids,
            )
            new_food_rows: list[dict[str, Any]] = []
            new_interior_rows: list[dict[str, Any]] = []

            if image_row_count > 0:
                cleaned_images_csv = temp_dir / "social_images_cleaned.csv"
                run_python_script(
                    BASE_DIR / "pipelines" / "clean_social_images.py",
                    "--input",
                    str(temp_images_csv),
                    "--lookup-csv",
                    str(args.lookup_csv),
                    "--output",
                    str(cleaned_images_csv),
                )
                food_count, interior_count = cleaned_image_mode_counts(cleaned_images_csv)

                if food_count > 0:
                    run_python_script(
                        BASE_DIR / "pipelines" / "generate_embeddings_images.py",
                        "--input",
                        str(cleaned_images_csv),
                        "--mode",
                        "food",
                        "--output-dir",
                        str(temp_dir),
                    )
                    new_food_rows = read_jsonl(temp_dir / "image_embeddings_food_latest.jsonl")

                if interior_count > 0:
                    run_python_script(
                        BASE_DIR / "pipelines" / "generate_embeddings_images.py",
                        "--input",
                        str(cleaned_images_csv),
                        "--mode",
                        "interior",
                        "--output-dir",
                        str(temp_dir),
                    )
                    new_interior_rows = read_jsonl(temp_dir / "image_embeddings_interior_latest.jsonl")

            removed_food, total_food = merge_jsonl_rows(
                food_jsonl_path,
                id_field="restaurant_id",
                restaurant_ids=restaurant_ids,
                new_rows=new_food_rows,
                backup=backup,
            )
            removed_interior, total_interior = merge_jsonl_rows(
                interior_jsonl_path,
                id_field="restaurant_id",
                restaurant_ids=restaurant_ids,
                new_rows=new_interior_rows,
                backup=backup,
            )
            print(f"image_embeddings_food_latest.jsonl merged: removed={removed_food} total={total_food}")
            print(f"image_embeddings_interior_latest.jsonl merged: removed={removed_interior} total={total_interior}")

    if not args.skip_profiles:
        run_python_script(BASE_DIR / "pipelines" / "build_restaurant_profiles.py")
        run_python_script(BASE_DIR / "pipelines" / "clean_restaurant_profiles.py")
        refresh_args = [str(BASE_DIR / "pipelines" / "refresh_review_profile_artifacts.py"), "--lookup-csv", str(args.lookup_csv)]
        if args.no_backup:
            refresh_args.append("--no-backup")
        run_python_script(Path(refresh_args[0]), *refresh_args[1:])

    print("Finished targeted social-derived artifact refresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

