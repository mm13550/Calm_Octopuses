from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOOKUP_PATH = BASE_DIR / "data" / "csv" / "restaurant_lookup.csv"


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().lower()


@lru_cache(maxsize=4)
def load_restaurant_lookup(path_str: str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    path = Path(path_str) if path_str else DEFAULT_LOOKUP_PATH
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}

    if not path.exists():
        return {"by_id": by_id, "by_name": by_name}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            restaurant_id = str(row.get("rest_id") or "").strip()
            restaurant_name = str(row.get("name") or "").strip()
            payload = {
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant_name,
                "homepage": str(row.get("homepage") or "").strip() or None,
                "borough": str(row.get("borough") or "").strip() or None,
                "michelin_category": str(row.get("michelin_category") or "").strip() or None,
            }

            if restaurant_id:
                by_id[restaurant_id] = payload
            if restaurant_name:
                by_name[_normalize_name(restaurant_name)] = payload

    return {"by_id": by_id, "by_name": by_name}


def get_restaurant_metadata(
    restaurant_id: str | None,
    restaurant_name: str | None,
    *,
    lookup_path: str | Path = DEFAULT_LOOKUP_PATH,
) -> dict[str, Any]:
    lookup = load_restaurant_lookup(str(lookup_path))
    by_id = lookup["by_id"]
    by_name = lookup["by_name"]

    normalized_id = str(restaurant_id or "").strip()
    normalized_name = str(restaurant_name or "").strip()

    payload = None
    if normalized_id:
        payload = by_id.get(normalized_id)
    if payload is None and normalized_name:
        payload = by_name.get(_normalize_name(normalized_name))

    return {
        "restaurant_id": normalized_id or (payload or {}).get("restaurant_id") or None,
        "restaurant_name": (payload or {}).get("restaurant_name") or normalized_name or None,
        "homepage": (payload or {}).get("homepage"),
        "borough": (payload or {}).get("borough"),
        "michelin_category": (payload or {}).get("michelin_category"),
    }
