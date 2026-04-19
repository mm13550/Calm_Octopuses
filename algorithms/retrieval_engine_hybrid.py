from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import retrieval_engine as menu_engine
from algorithms import retrieval_engine_reviews as review_engine


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_name(name: Any) -> str:
    return str(name or "").strip()


def _normalize_menu_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results", []) if isinstance(payload, dict) else []
    normalized: list[dict[str, Any]] = []

    for item in results:
        if not isinstance(item, dict):
            continue

        normalized.append({
            "restaurant_name": item.get("restaurant_name"),
            "dish_name": item.get("dish_name"),
            "price": item.get("price"),
            "score": round(_safe_float(item.get("score")), 4),
            "content_type": item.get("content_type"),
            "source": item.get("source"),
            "text": item.get("text"),
        })

    return normalized


def _normalize_review_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results", []) if isinstance(payload, dict) else []
    normalized: list[dict[str, Any]] = []

    for item in results:
        if not isinstance(item, dict):
            continue

        normalized.append({
            "restaurant_name": item.get("restaurant_name"),
            "rating": item.get("rating"),
            "score": round(_safe_float(item.get("score")), 4),
            "source": item.get("source"),
            "text": item.get("text"),
        })

    return normalized


def _dedupe_menu_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []

    for item in items:
        key = (
            str(item.get("dish_name") or "").strip().lower(),
            str(item.get("price") or "").strip().lower(),
            str(item.get("text") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    deduped.sort(key=lambda x: _safe_float(x.get("score")), reverse=True)
    return deduped


def _dedupe_review_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []

    for item in items:
        key = (
            str(item.get("text") or "").strip().lower(),
            str(item.get("rating") or "").strip().lower(),
            str(item.get("source") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    deduped.sort(key=lambda x: _safe_float(x.get("score")), reverse=True)
    return deduped


def _merge_by_restaurant(
    menu_results: list[dict[str, Any]],
    review_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    def ensure_card(restaurant_name: str) -> dict[str, Any]:
        key = _normalize_name(restaurant_name)
        if not key:
            key = "Unknown Restaurant"

        if key not in merged:
            merged[key] = {
                "restaurant_name": key,
                "menu_matches": [],
                "review_matches": [],
                "menu_max_score": 0.0,
                "review_max_score": 0.0,
                "combined_score": 0.0,
            }
        return merged[key]

    for item in menu_results:
        restaurant_name = _normalize_name(item.get("restaurant_name"))
        card = ensure_card(restaurant_name)
        card["menu_matches"].append(item)
        card["menu_max_score"] = max(card["menu_max_score"], _safe_float(item.get("score")))

    for item in review_results:
        restaurant_name = _normalize_name(item.get("restaurant_name"))
        card = ensure_card(restaurant_name)
        card["review_matches"].append(item)
        card["review_max_score"] = max(card["review_max_score"], _safe_float(item.get("score")))

    cards: list[dict[str, Any]] = []
    for card in merged.values():
        card["menu_matches"] = _dedupe_menu_items(card["menu_matches"])
        card["review_matches"] = _dedupe_review_items(card["review_matches"])

        card["combined_score"] = round(
            (card["menu_max_score"] * 0.45) +
            (card["review_max_score"] * 0.45) +
            (0.10 if card["menu_matches"] and card["review_matches"] else 0.0),
            4,
        )

        cards.append({
            "restaurant_name": card["restaurant_name"],
            "combined_score": card["combined_score"],
            "menu_max_score": round(card["menu_max_score"], 4),
            "review_max_score": round(card["review_max_score"], 4),
            "menu_matches": card["menu_matches"],
            "review_matches": card["review_matches"],
        })

    cards.sort(key=lambda x: x["combined_score"], reverse=True)
    return cards


def hybrid_search_api(
    query_text: str,
    *,
    menu_top_k: int = 5,
    review_top_k: int = 5,
    merged_top_k: int = 10,
) -> dict[str, Any]:
    menu_payload = menu_engine.search_api_simple(
        query_text,
        top_k=menu_top_k,
    )

    review_payload = review_engine.search_reviews_api(
        query_text,
        top_k=review_top_k,
    )

    menu_results = _normalize_menu_results(menu_payload)
    review_results = _normalize_review_results(review_payload)
    restaurant_cards = _merge_by_restaurant(menu_results, review_results)[:merged_top_k]

    return {
        "query": query_text,
        "menu_results": menu_results,
        "review_results": review_results,
        "restaurant_cards": restaurant_cards,
    }


if __name__ == "__main__":
    demo_query = "quiet omakase"
    result = hybrid_search_api(
        demo_query,
        menu_top_k=5,
        review_top_k=5,
        merged_top_k=10,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))