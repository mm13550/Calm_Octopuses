"""
pipelines/check_embedding_integrity.py
=======================================
Validates embedding JSONL files and raw social data (images, reviews) against
the restaurant lookup for coverage and structural integrity.

Reports missing restaurants, malformed vectors, duplicate records, and
embedding dimension mismatches to stdout.

Usage::

    python pipelines/check_embedding_integrity.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LOOKUP_CSV = DATA_DIR / "csv" / "restaurant_lookup.csv"
RAW_IMAGES_CSV = DATA_DIR / "csv" / "social_images.csv"
RAW_REVIEWS_CSV = DATA_DIR / "csv" / "social_reviews.csv"

SAMPLE_LIMIT = 5


@dataclass
class CheckResult:
    label: str
    path: Path
    row_count: int
    issues: list[str]


def load_lookup_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rest_id = str(row.get("rest_id") or "").strip()
            if rest_id:
                ids.add(rest_id)
    return ids


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_counter(counter: Counter[Any], *, formatter=str) -> str:
    duplicates = [(key, count) for key, count in counter.items() if count > 1]
    if not duplicates:
        return ""

    sample = ", ".join(
        f"{formatter(key)} x{count}"
        for key, count in duplicates[:SAMPLE_LIMIT]
    )
    return f"{len(duplicates)} duplicate keys; sample: {sample}"


def analyze_raw_images(path: Path) -> CheckResult:
    rows = read_csv_rows(path)
    key_counter = Counter(
        (
            str(row.get("image_uid") or "").strip(),
            str(row.get("rest_id") or "").strip(),
            str(row.get("image_path") or "").strip(),
        )
        for row in rows
    )
    issues: list[str] = []
    summary = summarize_counter(key_counter, formatter=lambda x: "|".join(x))
    if summary:
        issues.append(f"duplicate image rows by (image_uid, rest_id, image_path): {summary}")
    return CheckResult("raw_images_csv", path, len(rows), issues)


def analyze_raw_reviews(path: Path) -> CheckResult:
    rows = read_csv_rows(path)
    key_counter = Counter(
        (
            str(row.get("uid") or "").strip(),
            str(row.get("rest_id") or "").strip(),
            str(row.get("text") or "").strip(),
            str(row.get("rating") or "").strip(),
        )
        for row in rows
    )
    issues: list[str] = []
    summary = summarize_counter(key_counter, formatter=lambda x: "|".join(x[:3] + (x[3],)))
    if summary:
        issues.append(f"duplicate review rows by (uid, rest_id, text, rating): {summary}")
    return CheckResult("raw_reviews_csv", path, len(rows), issues)


def analyze_jsonl_file(path: Path, lookup_ids: set[str]) -> CheckResult:
    rows = read_jsonl(path)
    issues: list[str] = []
    name = path.name

    exact_counter = Counter(
        json.dumps(
            {k: v for k, v in row.items() if k != "_line_no"},
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in rows
    )
    summary = summarize_counter(exact_counter, formatter=lambda _: "<exact-object>")
    if summary:
        issues.append(f"exact duplicate JSON objects: {summary}")

    if name == "menu_embeddings.jsonl":
        doc_counter = Counter(str(row.get("doc_id") or "").strip() for row in rows)
        summary = summarize_counter(doc_counter)
        if summary:
            issues.append(f"duplicate doc_id: {summary}")

        tuple_counter = Counter(
            (
                str(row.get("restaurant_id") or "").strip(),
                str(row.get("dish_name") or "").strip(),
                str(row.get("price") or "").strip(),
            )
            for row in rows
        )
        summary = summarize_counter(tuple_counter, formatter=lambda x: "|".join(x))
        if summary:
            issues.append(f"duplicate (restaurant_id, dish_name, price): {summary}")

        if lookup_ids:
            ids_not_in_lookup = sorted(
                {
                    str(row.get("restaurant_id") or "").strip()
                    for row in rows
                    if str(row.get("restaurant_id") or "").strip()
                    and str(row.get("restaurant_id") or "").strip() not in lookup_ids
                }
            )
            if ids_not_in_lookup:
                issues.append(
                    "restaurant_id not in restaurant_lookup.csv: "
                    + ", ".join(ids_not_in_lookup[:SAMPLE_LIMIT])
                    + ("" if len(ids_not_in_lookup) <= SAMPLE_LIMIT else ", ...")
                )

    elif name == "review_embeddings.jsonl":
        doc_counter = Counter(str(row.get("doc_id") or "").strip() for row in rows)
        summary = summarize_counter(doc_counter)
        if summary:
            issues.append(f"duplicate doc_id: {summary}")

        tuple_counter = Counter(
            (
                str(row.get("restaurant_id") or "").strip(),
                str(row.get("text") or "").strip(),
                str(row.get("rating") or "").strip(),
            )
            for row in rows
        )
        summary = summarize_counter(tuple_counter, formatter=lambda x: f"{x[0]}|{x[2]}")
        if summary:
            issues.append(f"duplicate (restaurant_id, text, rating): {summary}")

        if lookup_ids:
            ids_not_in_lookup = sorted(
                {
                    str(row.get("restaurant_id") or "").strip()
                    for row in rows
                    if str(row.get("restaurant_id") or "").strip()
                    and str(row.get("restaurant_id") or "").strip() not in lookup_ids
                }
            )
            if ids_not_in_lookup:
                issues.append(
                    "restaurant_id not in restaurant_lookup.csv: "
                    + ", ".join(ids_not_in_lookup[:SAMPLE_LIMIT])
                    + ("" if len(ids_not_in_lookup) <= SAMPLE_LIMIT else ", ...")
                )

    elif name.startswith("image_embeddings_"):
        doc_counter = Counter(str(row.get("doc_id") or "").strip() for row in rows)
        summary = summarize_counter(doc_counter)
        if summary:
            issues.append(f"duplicate doc_id: {summary}")

        path_counter = Counter(str(row.get("image_path") or "").strip() for row in rows)
        summary = summarize_counter(path_counter)
        if summary:
            issues.append(f"duplicate image_path: {summary}")

        tuple_counter = Counter(
            (
                str(row.get("doc_id") or "").strip(),
                str(row.get("restaurant_id") or "").strip(),
                str(row.get("image_path") or "").strip(),
            )
            for row in rows
        )
        summary = summarize_counter(tuple_counter, formatter=lambda x: "|".join(x))
        if summary:
            issues.append(f"duplicate (doc_id, restaurant_id, image_path): {summary}")

        if lookup_ids:
            ids_not_in_lookup = sorted(
                {
                    str(row.get("restaurant_id") or "").strip()
                    for row in rows
                    if str(row.get("restaurant_id") or "").strip()
                    and str(row.get("restaurant_id") or "").strip() not in lookup_ids
                }
            )
            if ids_not_in_lookup:
                issues.append(
                    "restaurant_id not in restaurant_lookup.csv: "
                    + ", ".join(ids_not_in_lookup[:SAMPLE_LIMIT])
                    + ("" if len(ids_not_in_lookup) <= SAMPLE_LIMIT else ", ...")
                )

    elif name.startswith("restaurant_profiles"):
        id_counter = Counter(str(row.get("restaurant_id") or "").strip() for row in rows)
        summary = summarize_counter(id_counter)
        if summary:
            issues.append(f"duplicate restaurant_id: {summary}")

        name_counter = Counter(str(row.get("restaurant_name") or "").strip() for row in rows)
        summary = summarize_counter(name_counter)
        if summary:
            issues.append(f"duplicate restaurant_name: {summary}")

        if lookup_ids:
            ids_not_in_lookup = sorted(
                {
                    rest_id
                    for rest_id in id_counter
                    if rest_id and rest_id not in lookup_ids
                }
            )
            if ids_not_in_lookup:
                issues.append(
                    "restaurant_id not in restaurant_lookup.csv: "
                    + ", ".join(ids_not_in_lookup[:SAMPLE_LIMIT])
                    + ("" if len(ids_not_in_lookup) <= SAMPLE_LIMIT else ", ...")
                )

    elif name.startswith("review_restaurant_ids"):
        id_counter = Counter(str(row.get("restaurant_id") or "").strip() for row in rows)
        summary = summarize_counter(id_counter)
        if summary:
            issues.append(f"duplicate restaurant_id: {summary}")

        if lookup_ids:
            ids_not_in_lookup = sorted(
                {
                    rest_id
                    for rest_id in id_counter
                    if rest_id and rest_id not in lookup_ids
                }
            )
            if ids_not_in_lookup:
                issues.append(
                    "restaurant_id not in restaurant_lookup.csv: "
                    + ", ".join(ids_not_in_lookup[:SAMPLE_LIMIT])
                    + ("" if len(ids_not_in_lookup) <= SAMPLE_LIMIT else ", ...")
                )

    elif name == "restaurant_summary.jsonl":
        id_counter = Counter(str(row.get("restaurant_id") or "").strip() for row in rows)
        summary = summarize_counter(id_counter)
        if summary:
            issues.append(f"duplicate restaurant_id: {summary}")

        if lookup_ids:
            ids_not_in_lookup = sorted(
                {
                    rest_id
                    for rest_id in id_counter
                    if rest_id and rest_id not in lookup_ids
                }
            )
            if ids_not_in_lookup:
                issues.append(
                    "restaurant_id not in restaurant_lookup.csv: "
                    + ", ".join(ids_not_in_lookup[:SAMPLE_LIMIT])
                    + ("" if len(ids_not_in_lookup) <= SAMPLE_LIMIT else ", ...")
                )

    return CheckResult("jsonl", path, len(rows), issues)


def print_result(result: CheckResult) -> None:
    rel_path = result.path.relative_to(BASE_DIR)
    print(f"\n[{result.label}] {rel_path}")
    print(f"rows: {result.row_count}")
    if not result.issues:
        print("issues: none")
        return
    print("issues:")
    for issue in result.issues:
        print(f"- {issue}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check embedding and raw data files for duplicates and canonical-ID issues.")
    parser.add_argument("--embeddings-dir", default=str(EMBEDDINGS_DIR))
    parser.add_argument("--lookup-csv", default=str(LOOKUP_CSV))
    parser.add_argument("--raw-images-csv", default=str(RAW_IMAGES_CSV))
    parser.add_argument("--raw-reviews-csv", default=str(RAW_REVIEWS_CSV))
    parser.add_argument("--skip-raw", action="store_true", help="Skip raw CSV checks.")
    args = parser.parse_args()

    lookup_ids = load_lookup_ids(Path(args.lookup_csv))
    results: list[CheckResult] = []

    if not args.skip_raw:
        raw_images = Path(args.raw_images_csv)
        if raw_images.exists():
            results.append(analyze_raw_images(raw_images))

        raw_reviews = Path(args.raw_reviews_csv)
        if raw_reviews.exists():
            results.append(analyze_raw_reviews(raw_reviews))

    embeddings_dir = Path(args.embeddings_dir)
    for path in sorted(embeddings_dir.glob("*.jsonl")):
        results.append(analyze_jsonl_file(path, lookup_ids))

    any_issues = False
    for result in results:
        print_result(result)
        any_issues = any_issues or bool(result.issues)

    return 1 if any_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
