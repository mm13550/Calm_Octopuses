from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

LOOKUP_CSV = DATA_DIR / "csv" / "restaurant_lookup.csv"
MAPPING_CSV = DATA_DIR / "csv" / "social_rest_id_mapping.csv"
RAW_REVIEWS_CSV = DATA_DIR / "social_reviews.csv"
RAW_IMAGES_CSV = DATA_DIR / "social_images.csv"
MENU_JSONL = EMBEDDINGS_DIR / "menu_embeddings_latest.jsonl"
DEFAULT_OUTPUT_CSV = DATA_DIR / "csv" / "likely_problematic_social_restaurants.csv"


@dataclass
class Candidate:
    rest_id: str
    restaurant_name: str
    menu_count: int = 0
    review_count: int = 0
    image_count: int = 0
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def read_lookup(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {
            clean_text(row.get("rest_id") or row.get("restaurant_id") or row.get("place_id")):
            clean_text(row.get("name") or row.get("restaurant_name"))
            for row in reader
            if clean_text(row.get("rest_id") or row.get("restaurant_id") or row.get("place_id"))
        }


def read_mapping_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_counter_csv(path: Path, field: str) -> Counter[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return Counter(clean_text(row.get(field)) for row in csv.DictReader(f) if clean_text(row.get(field)))


def read_menu_counts(path: Path, valid_rest_ids: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            restaurant_id = clean_text(row.get("restaurant_id"))
            if restaurant_id in valid_rest_ids:
                counts[restaurant_id] += 1
    return counts


def add_reason(candidate: Candidate, reason: str, note: str) -> None:
    if reason not in candidate.reasons:
        candidate.reasons.append(reason)
    if note not in candidate.notes:
        candidate.notes.append(note)


def build_candidates(
    lookup: dict[str, str],
    mapping_rows: list[dict[str, str]],
    menu_counts: Counter[str],
    review_counts: Counter[str],
    image_counts: Counter[str],
) -> list[Candidate]:
    candidates: dict[str, Candidate] = {}

    for row in mapping_rows:
        action = clean_text(row.get("action")).lower()
        target_rest_id = clean_text(row.get("target_rest_id"))
        if action != "remap" or not target_rest_id or target_rest_id not in lookup:
            continue

        review_count = review_counts.get(target_rest_id, 0)
        image_count = image_counts.get(target_rest_id, 0)
        if review_count > 0 and image_count > 0:
            continue

        missing_parts: list[str] = []
        if review_count == 0:
            missing_parts.append("reviews")
        if image_count == 0:
            missing_parts.append("images")

        candidate = candidates.setdefault(
            target_rest_id,
            Candidate(
                rest_id=target_rest_id,
                restaurant_name=lookup[target_rest_id],
                menu_count=menu_counts.get(target_rest_id, 0),
                review_count=review_count,
                image_count=image_count,
            ),
        )
        add_reason(
            candidate,
            "mapped_target",
            "Was repaired from source id "
            f"{clean_text(row.get('source_rest_id'))}, but canonical raw "
            f"{' and '.join(missing_parts)} are still missing.",
        )

    for rest_id, restaurant_name in lookup.items():
        menu_count = menu_counts.get(rest_id, 0)
        review_count = review_counts.get(rest_id, 0)
        image_count = image_counts.get(rest_id, 0)
        if menu_count <= 0:
            continue
        if review_count == 0 and image_count == 0:
            candidate = candidates.setdefault(
                rest_id,
                Candidate(
                    rest_id=rest_id,
                    restaurant_name=restaurant_name,
                    menu_count=menu_count,
                    review_count=review_count,
                    image_count=image_count,
                ),
            )
            add_reason(
                candidate,
                "menu_only_raw",
                "Canonical restaurant has menu data but no raw social reviews or images.",
            )

    for candidate in candidates.values():
        candidate.menu_count = menu_counts.get(candidate.rest_id, candidate.menu_count)
        candidate.review_count = review_counts.get(candidate.rest_id, candidate.review_count)
        candidate.image_count = image_counts.get(candidate.rest_id, candidate.image_count)

    return sorted(candidates.values(), key=lambda item: (item.restaurant_name.lower(), item.rest_id))


def write_candidate_csv(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rest_id",
                "restaurant_name",
                "menu_count",
                "review_count",
                "image_count",
                "reasons",
                "notes",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "rest_id": candidate.rest_id,
                    "restaurant_name": candidate.restaurant_name,
                    "menu_count": candidate.menu_count,
                    "review_count": candidate.review_count,
                    "image_count": candidate.image_count,
                    "reasons": "; ".join(candidate.reasons),
                    "notes": " | ".join(candidate.notes),
                }
            )


def run_python_script(script_path: Path, *args: str) -> None:
    cmd = [sys.executable, str(script_path), *args]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify and optionally rerun canonical restaurants whose social scrape is likely problematic."
    )
    parser.add_argument("--lookup-csv", default=str(LOOKUP_CSV))
    parser.add_argument("--mapping-csv", default=str(MAPPING_CSV))
    parser.add_argument("--raw-reviews-csv", default=str(RAW_REVIEWS_CSV))
    parser.add_argument("--raw-images-csv", default=str(RAW_IMAGES_CSV))
    parser.add_argument("--menu-jsonl", default=str(MENU_JSONL))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--execute", action="store_true", help="Run the canonical social scraper and targeted artifact refresh for all candidates.")
    parser.add_argument("--skip-scrape", action="store_true", help="When --execute is set, skip social scraping and only refresh derived artifacts.")
    parser.add_argument("--skip-refresh", action="store_true", help="When --execute is set, skip derived artifact refresh.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Delay passed through to social_scraper.py when executing.")
    parser.add_argument("--no-backup", action="store_true", help="Pass through no-backup mode to downstream rewrite scripts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    lookup = read_lookup(Path(args.lookup_csv))
    mapping_rows = read_mapping_rows(Path(args.mapping_csv))
    review_counts = read_counter_csv(Path(args.raw_reviews_csv), "rest_id")
    image_counts = read_counter_csv(Path(args.raw_images_csv), "rest_id")
    menu_counts = read_menu_counts(Path(args.menu_jsonl), set(lookup))

    candidates = build_candidates(lookup, mapping_rows, menu_counts, review_counts, image_counts)
    write_candidate_csv(Path(args.output_csv), candidates)

    print(f"Wrote {len(candidates)} candidates to {args.output_csv}")
    for candidate in candidates:
        print(
            f"{candidate.restaurant_name} ({candidate.rest_id}) | "
            f"menu={candidate.menu_count} reviews={candidate.review_count} images={candidate.image_count} | "
            f"reasons={', '.join(candidate.reasons)}"
        )

    if not args.execute:
        print("Report-only mode. Re-run with --execute to scrape and refresh these restaurants.")
        return 0

    rest_ids = [candidate.rest_id for candidate in candidates]
    if not rest_ids:
        print("No candidates selected; nothing to rerun.")
        return 0

    if not args.skip_scrape:
        scraper_args = [
            BASE_DIR / "pipelines" / "social_scraper.py",
            "--sleep-seconds",
            str(args.sleep_seconds),
        ]
        for rest_id in rest_ids:
            scraper_args.extend(["--rest-id", rest_id])
        if args.no_backup:
            scraper_args.append("--no-backup")
        run_python_script(scraper_args[0], *map(str, scraper_args[1:]))

    if not args.skip_refresh:
        refresh_args = [BASE_DIR / "pipelines" / "refresh_social_derived_artifacts.py"]
        for rest_id in rest_ids:
            refresh_args.extend(["--restaurant-id", rest_id])
        if args.no_backup:
            refresh_args.append("--no-backup")
        run_python_script(refresh_args[0], *map(str, refresh_args[1:]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
