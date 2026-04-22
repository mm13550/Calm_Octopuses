from __future__ import annotations

import argparse
import csv
import difflib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

try:
    from apify_client import ApifyClient
except Exception:  # pragma: no cover - optional local dependency
    ApifyClient = None


load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
LOOKUP_CSV = DATA_DIR / "csv" / "restaurant_lookup.csv"
MAPPING_CSV = DATA_DIR / "csv" / "social_rest_id_mapping.csv"
REVIEWS_OUTPUT = DATA_DIR / "social_reviews.csv"
IMAGES_OUTPUT = DATA_DIR / "social_images.csv"

REQUEST_TIMEOUT = 20
NAME_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
NAME_STOPWORDS = {
    "restaurant",
    "restaurants",
    "nyc",
    "new",
    "york",
    "city",
    "the",
    "and",
    "of",
}
REVIEW_FIELDS = ["uid", "rest_id", "source", "text", "rating"]
IMAGE_FIELDS = ["image_uid", "rest_id", "source", "image_path"]

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RestaurantTarget:
    restaurant_name: str
    rest_id: str
    homepage: str
    borough: str
    michelin_category: str


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def to_project_relative_path(file_path: str | Path) -> str:
    return os.path.relpath(str(file_path), start=BASE_DIR).replace(os.sep, "/")


def normalize_domain(raw_url: str) -> str:
    value = clean_text(raw_url)
    if not value:
        return ""

    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = (parsed.netloc or parsed.path).split("/", 1)[0].split(":", 1)[0].lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def normalized_name_tokens(value: str) -> list[str]:
    return [
        token
        for token in NAME_TOKEN_PATTERN.findall(clean_text(value).lower())
        if token and token not in NAME_STOPWORDS
    ]


def name_match_score(expected_name: str, candidate_name: str) -> float:
    expected_tokens = normalized_name_tokens(expected_name)
    candidate_tokens = normalized_name_tokens(candidate_name)
    if not expected_tokens or not candidate_tokens:
        return 0.0

    expected_set = set(expected_tokens)
    candidate_set = set(candidate_tokens)
    overlap = len(expected_set & candidate_set)
    if overlap == 0:
        return 0.0

    recall = overlap / len(expected_set)
    precision = overlap / len(candidate_set)
    f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) else 0.0
    ratio = difflib.SequenceMatcher(
        None,
        " ".join(expected_tokens),
        " ".join(candidate_tokens),
    ).ratio()

    expected_compact = " ".join(expected_tokens)
    candidate_compact = " ".join(candidate_tokens)
    prefix_bonus = 1.0 if (
        expected_compact.startswith(candidate_compact)
        or candidate_compact.startswith(expected_compact)
    ) else 0.0

    return round(0.55 * f1 + 0.35 * ratio + 0.10 * prefix_bonus, 4)


def load_lookup_targets(path: Path) -> list[RestaurantTarget]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    targets: list[RestaurantTarget] = []
    for row in rows:
        restaurant_name = clean_text(row.get("name") or row.get("restaurant_name"))
        rest_id = clean_text(row.get("rest_id") or row.get("restaurant_id") or row.get("place_id"))
        if not restaurant_name or not rest_id:
            continue
        targets.append(
            RestaurantTarget(
                restaurant_name=restaurant_name,
                rest_id=rest_id,
                homepage=clean_text(row.get("homepage")),
                borough=clean_text(row.get("borough")),
                michelin_category=clean_text(row.get("michelin_category")),
            )
        )
    return targets


def load_google_override_map(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}

    overrides: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            action = clean_text(row.get("action")).lower()
            source_rest_id = clean_text(row.get("source_rest_id"))
            target_rest_id = clean_text(row.get("target_rest_id"))
            if action != "remap" or not source_rest_id or not target_rest_id:
                continue
            overrides.setdefault(target_rest_id, []).append(source_rest_id)
    return overrides


def select_targets(
    targets: list[RestaurantTarget],
    *,
    rest_ids: list[str],
    names: list[str],
    start: int | None,
    end: int | None,
    limit: int | None,
) -> list[RestaurantTarget]:
    by_rest_id = {target.rest_id: target for target in targets}
    by_name = {target.restaurant_name.casefold(): target for target in targets}

    requested_rest_ids = {clean_text(rest_id) for rest_id in rest_ids if clean_text(rest_id)}
    requested_names = {clean_text(name).casefold() for name in names if clean_text(name)}

    missing_rest_ids = sorted(rest_id for rest_id in requested_rest_ids if rest_id not in by_rest_id)
    missing_names = sorted(name for name in requested_names if name not in by_name)
    if missing_rest_ids:
        raise ValueError(f"Unknown rest_id values: {missing_rest_ids}")
    if missing_names:
        raise ValueError(f"Unknown restaurant names: {missing_names}")

    if requested_rest_ids or requested_names:
        selected = [
            target
            for target in targets
            if target.rest_id in requested_rest_ids or target.restaurant_name.casefold() in requested_names
        ]
    else:
        selected = list(targets)

    start_idx = 0 if start is None else max(start, 0)
    end_idx = len(selected) if end is None else min(end, len(selected))
    selected = selected[start_idx:end_idx]

    if limit is not None:
        selected = selected[: max(limit, 0)]

    return selected


def download_google_photo(photo_reference: str, filename_prefix: str) -> str | None:
    photo_url = (
        f"https://places.googleapis.com/v1/{photo_reference}/media"
        f"?maxHeightPx=800&maxWidthPx=800&key={GOOGLE_PLACES_API_KEY}"
    )
    response = requests.get(photo_url, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        return None

    file_path = IMAGES_DIR / f"{filename_prefix}.jpg"
    file_path.write_bytes(response.content)
    return str(file_path)


def extract_text(value: Any) -> str:
    if isinstance(value, dict):
        return clean_text(value.get("text") or value.get("displayName"))
    return clean_text(value)


def extract_location_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("address", "formattedAddress", "neighborhood", "city", "state", "zipCode"):
        value = item.get(key)
        if isinstance(value, dict):
            parts.extend(clean_text(v) for v in value.values() if clean_text(v))
        elif isinstance(value, list):
            parts.extend(clean_text(v) for v in value if clean_text(v))
        else:
            text = clean_text(value)
            if text:
                parts.append(text)

    location = item.get("location")
    if isinstance(location, dict):
        parts.extend(clean_text(v) for v in location.values() if clean_text(v))
    elif isinstance(location, str):
        parts.append(clean_text(location))

    return " ".join(part for part in parts if part).strip()


def extract_candidate_name(item: dict[str, Any]) -> str:
    for key in ("name", "businessName", "title", "displayName"):
        text = extract_text(item.get(key))
        if text:
            return text
    return ""


def extract_candidate_urls(item: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in (
        "website",
        "websiteUrl",
        "websiteUri",
        "homepage",
        "homePage",
        "businessWebsite",
        "businessWebsiteUrl",
    ):
        value = item.get(key)
        if isinstance(value, dict):
            for nested in value.values():
                text = clean_text(nested)
                if text:
                    urls.append(text)
        else:
            text = clean_text(value)
            if text:
                urls.append(text)
    return urls


def score_yelp_candidate(item: dict[str, Any], target: RestaurantTarget) -> dict[str, Any]:
    matched_name = extract_candidate_name(item)
    homepage_domain = normalize_domain(target.homepage)
    candidate_domains = {
        normalize_domain(url)
        for url in extract_candidate_urls(item)
        if normalize_domain(url)
    }
    homepage_match = bool(homepage_domain and homepage_domain in candidate_domains)
    name_score = name_match_score(target.restaurant_name, matched_name)

    location_text = extract_location_text(item).lower()
    borough_match = bool(target.borough and target.borough.lower() in location_text)
    new_york_match = "new york" in location_text or "ny" in location_text or "nyc" in location_text
    location_match = borough_match or new_york_match

    accepted = homepage_match or name_score >= 0.78 or (name_score >= 0.68 and location_match)
    score = name_score + (0.35 if homepage_match else 0.0) + (0.05 if location_match else 0.0)

    return {
        "item": item,
        "matched_name": matched_name,
        "name_score": name_score,
        "homepage_match": homepage_match,
        "location_match": location_match,
        "score": round(score, 4),
        "accepted": accepted,
    }


def is_google_match_acceptable(target: RestaurantTarget, matched_name: str, matched_homepage: str) -> tuple[bool, float, bool]:
    name_score = name_match_score(target.restaurant_name, matched_name)
    homepage_match = bool(
        normalize_domain(target.homepage)
        and normalize_domain(target.homepage) == normalize_domain(matched_homepage)
    )
    accepted = homepage_match or name_score >= 0.72
    return accepted, name_score, homepage_match


def fetch_google_places_data(
    target: RestaurantTarget,
    *,
    google_lookup_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reviews_out: list[dict[str, Any]] = []
    images_out: list[dict[str, Any]] = []

    if not GOOGLE_PLACES_API_KEY:
        print("  [google] skipped: GOOGLE_PLACES_API_KEY missing")
        return reviews_out, images_out

    headers = {
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "id,displayName,formattedAddress,websiteUri,reviews,photos",
    }

    seen_lookup_ids: set[str] = set()
    for lookup_id in google_lookup_ids:
        lookup_id = clean_text(lookup_id)
        if not lookup_id or lookup_id in seen_lookup_ids:
            continue
        seen_lookup_ids.add(lookup_id)

        if lookup_id != target.rest_id:
            print(f"  [google] trying legacy lookup id override {lookup_id}")

        details_url = f"https://places.googleapis.com/v1/places/{lookup_id}"
        try:
            response = requests.get(details_url, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception as exc:
            print(f"  [google] request error for {lookup_id}: {exc}")
            continue

        if response.status_code != 200:
            print(f"  [google] details lookup failed for {lookup_id}: {response.status_code}")
            continue

        details = response.json()
        matched_name = extract_text(details.get("displayName"))
        matched_homepage = clean_text(details.get("websiteUri"))
        accepted, name_score, homepage_match = is_google_match_acceptable(
            target,
            matched_name,
            matched_homepage,
        )

        if not accepted:
            print(
                "  [google] rejected candidate "
                f"'{matched_name or lookup_id}' for {target.restaurant_name} "
                f"(score={name_score:.2f}, homepage_match={homepage_match})"
            )
            continue

        reviews = details.get("reviews", []) or []
        photos = details.get("photos", []) or []

        for index, review in enumerate(reviews):
            reviews_out.append(
                {
                    "uid": f"g_{target.rest_id}_{index}",
                    "rest_id": target.rest_id,
                    "source": "google_places",
                    "text": extract_text(review.get("text")),
                    "rating": review.get("rating"),
                }
            )

        for index, photo in enumerate(photos):
            photo_reference = clean_text(photo.get("name"))
            if not photo_reference:
                continue
            image_path = download_google_photo(photo_reference, f"{target.rest_id}_google_{index}")
            if image_path:
                images_out.append(
                    {
                        "image_uid": f"img_g_{target.rest_id}_{index}",
                        "rest_id": target.rest_id,
                        "source": "google_places",
                        "image_path": to_project_relative_path(image_path),
                    }
                )

        print(
            "  [google] matched "
            f"'{matched_name or target.restaurant_name}' via {lookup_id} "
            f"| reviews={len(reviews_out)} | images={len(images_out)}"
        )
        return reviews_out, images_out

    print(f"  [google] skipped: no trustworthy place match for {target.restaurant_name}")
    return reviews_out, images_out


def fetch_yelp_data_apify(
    target: RestaurantTarget,
    *,
    search_limit: int,
    reviews_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reviews_out: list[dict[str, Any]] = []
    images_out: list[dict[str, Any]] = []

    if not APIFY_API_TOKEN:
        print("  [yelp] skipped: APIFY_API_TOKEN missing")
        return reviews_out, images_out
    if ApifyClient is None:
        print("  [yelp] skipped: apify_client is not installed")
        return reviews_out, images_out

    client = ApifyClient(APIFY_API_TOKEN)
    location_hint = target.borough or "New York"
    run_input = {
        "searchTerms": [target.restaurant_name],
        "locations": [location_hint],
        "searchLimit": max(search_limit, 1),
        "reviewsCount": max(reviews_count, 1),
        "scrapeReview": True,
        "scrapeImages": True,
    }

    try:
        print(f"  [yelp] searching '{target.restaurant_name}' in '{location_hint}'...")
        run = client.actor("tri_angle/yelp-scraper").call(run_input=run_input)
    except Exception as exc:
        print(f"  [yelp] actor error: {exc}")
        return reviews_out, images_out

    scored_candidates: list[dict[str, Any]] = []
    try:
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            scored_candidates.append(score_yelp_candidate(item, target))
    except Exception as exc:
        print(f"  [yelp] dataset error: {exc}")
        return reviews_out, images_out

    if not scored_candidates:
        print("  [yelp] no candidates returned")
        return reviews_out, images_out

    scored_candidates.sort(
        key=lambda candidate: (candidate["accepted"], candidate["score"]),
        reverse=True,
    )
    best = next((candidate for candidate in scored_candidates if candidate["accepted"]), None)
    if best is None:
        top_names = [candidate["matched_name"] for candidate in scored_candidates[:3] if candidate["matched_name"]]
        print(
            "  [yelp] skipped: no trustworthy candidate "
            f"(top candidates: {top_names or ['<none>']})"
        )
        return reviews_out, images_out

    item = best["item"]
    matched_name = best["matched_name"] or target.restaurant_name
    reviews = item.get("reviews", []) or []
    photos = item.get("photos") or item.get("images") or item.get("imageUrls") or []

    for index, review in enumerate(reviews):
        content = clean_text(review.get("text") or review.get("body"))
        if not content:
            continue
        reviews_out.append(
            {
                "uid": f"y_{target.rest_id}_{index}",
                "rest_id": target.rest_id,
                "source": "yelp",
                "text": content,
                "rating": review.get("rating"),
            }
        )

    for index, photo in enumerate(photos):
        image_url = photo if isinstance(photo, str) else clean_text(photo.get("url") or photo.get("link"))
        if not image_url:
            continue
        if image_url.startswith("//"):
            image_url = "https:" + image_url

        try:
            response = requests.get(image_url, timeout=REQUEST_TIMEOUT)
            content_type = response.headers.get("content-type", "")
            if response.status_code != 200 or "image" not in content_type:
                continue
        except Exception:
            continue

        file_path = IMAGES_DIR / f"{target.rest_id}_yelp_{index}.jpg"
        file_path.write_bytes(response.content)
        images_out.append(
            {
                "image_uid": f"img_y_{target.rest_id}_{index}",
                "rest_id": target.rest_id,
                "source": "yelp",
                "image_path": to_project_relative_path(file_path),
            }
        )

    print(
        "  [yelp] matched "
        f"'{matched_name}' (score={best['score']:.2f}) | reviews={len(reviews_out)} | images={len(images_out)}"
    )
    return reviews_out, images_out


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.exists():
        return [], []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def replace_file(temp_path: Path, target_path: Path, *, backup: bool) -> None:
    if backup and target_path.exists():
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        backup_path.write_bytes(target_path.read_bytes())
    temp_path.replace(target_path)


def rewrite_csv(
    path: Path,
    *,
    required_fields: list[str],
    drop_rest_ids: set[str],
    replace_sources_by_rest_id: dict[str, set[str]],
    new_rows: list[dict[str, Any]],
    backup: bool,
) -> tuple[int, int]:
    existing_fieldnames, existing_rows = read_csv_rows(path)
    fieldnames = list(existing_fieldnames or required_fields)
    for field in required_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    kept_rows = [
        row
        for row in existing_rows
        if clean_text(row.get("rest_id")) not in drop_rest_ids
        and (
            clean_text(row.get("rest_id")) not in replace_sources_by_rest_id
            or clean_text(row.get("source")) not in replace_sources_by_rest_id[clean_text(row.get("rest_id"))]
        )
    ]
    removed = len(existing_rows) - len(kept_rows)

    rows_to_write = kept_rows + [
        {field: row.get(field, "") for field in fieldnames}
        for row in new_rows
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as temp_file:
        temp_path = Path(temp_file.name)
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)

    replace_file(temp_path, path, backup=backup)
    return removed, len(rows_to_write)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape social reviews/images using canonical restaurant_lookup.csv identities."
    )
    parser.add_argument("--lookup-csv", default=str(LOOKUP_CSV))
    parser.add_argument("--mapping-csv", default=str(MAPPING_CSV))
    parser.add_argument("--reviews-output", default=str(REVIEWS_OUTPUT))
    parser.add_argument("--images-output", default=str(IMAGES_OUTPUT))
    parser.add_argument("--rest-id", action="append", default=[], help="Canonical rest_id to scrape. Repeatable.")
    parser.add_argument("--restaurant-name", action="append", default=[], help="Canonical restaurant name to scrape. Repeatable.")
    parser.add_argument("--purge-rest-id", action="append", default=[], help="Stale rest_id to remove from raw social outputs before writing.")
    parser.add_argument("--start", type=int, default=None, help="Optional start index into restaurant_lookup.csv.")
    parser.add_argument("--end", type=int, default=None, help="Optional end index into restaurant_lookup.csv.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of selected restaurants to process.")
    parser.add_argument("--skip-google", action="store_true", help="Do not fetch Google Places data.")
    parser.add_argument("--skip-yelp", action="store_true", help="Do not fetch Yelp data.")
    parser.add_argument("--search-limit", type=int, default=5, help="How many Yelp candidates to inspect before scoring.")
    parser.add_argument("--reviews-count", type=int, default=40, help="Max Yelp reviews to request from Apify.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Delay between restaurants.")
    parser.add_argument("--no-backup", action="store_true", help="Do not write .bak files before replacing raw outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backup = not args.no_backup

    lookup_path = Path(args.lookup_csv)
    targets = load_lookup_targets(lookup_path)
    google_override_map = load_google_override_map(Path(args.mapping_csv))
    selected_targets = select_targets(
        targets,
        rest_ids=args.rest_id,
        names=args.restaurant_name,
        start=args.start,
        end=args.end,
        limit=args.limit,
    )

    purge_rest_ids = {clean_text(rest_id) for rest_id in args.purge_rest_id if clean_text(rest_id)}
    target_rest_ids = {target.rest_id for target in selected_targets} | purge_rest_ids

    print(f"Loaded {len(targets)} canonical restaurants from {lookup_path}")
    print(f"Selected {len(selected_targets)} restaurants for scraping")
    if purge_rest_ids:
        print(f"Will purge stale rest_id values: {sorted(purge_rest_ids)}")

    all_review_rows: list[dict[str, Any]] = []
    all_image_rows: list[dict[str, Any]] = []
    review_replace_sources_by_rest_id: dict[str, set[str]] = {}
    image_replace_sources_by_rest_id: dict[str, set[str]] = {}

    for index, target in enumerate(selected_targets, start=1):
        print(f"\n[{index}/{len(selected_targets)}] {target.restaurant_name} ({target.rest_id})")

        if not args.skip_google:
            google_lookup_ids = list(google_override_map.get(target.rest_id, [])) + [target.rest_id]
            google_reviews, google_images = fetch_google_places_data(
                target,
                google_lookup_ids=google_lookup_ids,
            )
            all_review_rows.extend(google_reviews)
            all_image_rows.extend(google_images)
            if google_reviews:
                review_replace_sources_by_rest_id.setdefault(target.rest_id, set()).add("google_places")
            if google_images:
                image_replace_sources_by_rest_id.setdefault(target.rest_id, set()).add("google_places")

        if not args.skip_yelp:
            yelp_reviews, yelp_images = fetch_yelp_data_apify(
                target,
                search_limit=args.search_limit,
                reviews_count=args.reviews_count,
            )
            all_review_rows.extend(yelp_reviews)
            all_image_rows.extend(yelp_images)
            if yelp_reviews:
                review_replace_sources_by_rest_id.setdefault(target.rest_id, set()).add("yelp")
            if yelp_images:
                image_replace_sources_by_rest_id.setdefault(target.rest_id, set()).add("yelp")

        if args.sleep_seconds > 0 and index < len(selected_targets):
            time.sleep(args.sleep_seconds)

    if not target_rest_ids:
        print("No target rest_id values selected and nothing to purge. No files changed.")
        return 0

    reviews_removed, review_total = rewrite_csv(
        Path(args.reviews_output),
        required_fields=REVIEW_FIELDS,
        drop_rest_ids=purge_rest_ids,
        replace_sources_by_rest_id=review_replace_sources_by_rest_id,
        new_rows=all_review_rows,
        backup=backup,
    )
    images_removed, image_total = rewrite_csv(
        Path(args.images_output),
        required_fields=IMAGE_FIELDS,
        drop_rest_ids=purge_rest_ids,
        replace_sources_by_rest_id=image_replace_sources_by_rest_id,
        new_rows=all_image_rows,
        backup=backup,
    )

    print("\nCompleted social scrape refresh.")
    print(f"Reviews updated for {len(selected_targets)} selected restaurants plus {len(purge_rest_ids)} purge ids; removed={reviews_removed}; total_rows={review_total}")
    print(f"Images updated for {len(selected_targets)} selected restaurants plus {len(purge_rest_ids)} purge ids; removed={images_removed}; total_rows={image_total}")
    print(f"New rows written: reviews={len(all_review_rows)} images={len(all_image_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
