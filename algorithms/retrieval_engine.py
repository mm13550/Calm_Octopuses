from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import lancedb

BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "algorithms" else Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "vector_db"
DATA_DIR = BASE_DIR / "data"
TABLE_NAME = "restaurant_vectors"
UTC = timezone.utc


@dataclass
class RankedResult:
    doc_id: str
    restaurant_id: str
    restaurant_name: str
    content_type: str
    source: str
    dish_name: str | None
    price: str | None
    created_at: str
    age_days: float
    cosine_distance: float
    semantic_similarity: float
    decay_factor: float
    freshness_adjustment: float
    lexical_bonus: float
    final_score: float
    trending_badge: bool
    text: str


@dataclass
class QueryUnderstanding:
    parser: str
    original_query: str
    normalized_query: str
    semantic_query: str
    must_include: list[str]
    should_include: list[str]
    exclude: list[str]
    cuisine_or_style: list[str]
    content_type_preference: str | None
    strict_filter: bool


def utc_now() -> datetime:
    return datetime.now(UTC)


def normalize(vector: Iterable[float]) -> list[float]:
    values = [float(x) for x in vector]
    norm = math.sqrt(sum(x * x for x in values))
    if norm == 0:
        raise ValueError("Vector must not be all zeros.")
    return [x / norm for x in values]


def parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def compute_lambda(half_life_days: float) -> float:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0")
    return math.log(2) / half_life_days


def exponential_decay(age_days: float, lambda_: float) -> float:
    return math.exp(-lambda_ * max(age_days, 0.0))


def blend_similarity_with_decay(
    semantic_similarity: float,
    decay_factor: float,
    alpha: float = 0.85,
) -> tuple[float, float]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    freshness_adjustment = (1.0 - alpha) + alpha * decay_factor
    final_score = semantic_similarity * freshness_adjustment
    return freshness_adjustment, final_score


def connect_db():
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(DB_PATH))


def _open_table_if_exists(db, table_name: str):
    try:
        return db.open_table(table_name)
    except Exception:
        return None


def ensure_table(rows: list[dict[str, Any]] | None = None, *, reset: bool = False):
    db = connect_db()
    table = _open_table_if_exists(db, TABLE_NAME)

    if table is None:
        if not rows:
            raise ValueError("Table does not exist yet. Provide seed rows to create it.")
        return db.create_table(TABLE_NAME, data=rows)

    if reset:
        if not rows:
            raise ValueError("reset=True requires rows to recreate the table.")
        return db.create_table(TABLE_NAME, data=rows, mode="overwrite")

    return table


REQUIRED_FIELDS = {
    "doc_id",
    "restaurant_id",
    "restaurant_name",
    "content_type",
    "text",
    "vector",
    "created_at",
    "source",
}


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")

    for idx, row in enumerate(rows, start=1):
        missing = REQUIRED_FIELDS - set(row.keys())
        if missing:
            raise ValueError(f"Row {idx} is missing required fields: {sorted(missing)}")

        if not isinstance(row["vector"], list) or not row["vector"]:
            raise ValueError(f"Row {idx} must contain a non-empty vector list")

        row["vector"] = normalize(row["vector"])
        row["created_at"] = parse_timestamp(row["created_at"]).isoformat()

        row.setdefault("dish_name", None)
        row.setdefault("price", None)
        row.setdefault("image_url", None)
        row.setdefault("menu_url", None)
        row.setdefault("metadata", {})


def ingest_documents(rows: list[dict[str, Any]], *, overwrite: bool = False):
    validate_rows(rows)
    db = connect_db()
    existing = _open_table_if_exists(db, TABLE_NAME)

    if overwrite:
        return ensure_table(rows, reset=True)

    if existing is None:
        return ensure_table(rows, reset=False)

    existing.add(rows)
    return existing


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return rows


def make_dummy_documents() -> list[dict[str, Any]]:
    now = utc_now()
    return [
        {
            "doc_id": "doc_001",
            "restaurant_id": "rest_atomix",
            "restaurant_name": "Atomix",
            "content_type": "review",
            "text": "Fresh review about an inventive Korean tasting menu with strong dessert praise.",
            "vector": [0.97, 0.21, 0.05, 0.03],
            "created_at": (now - timedelta(hours=6)).isoformat(),
            "source": "reddit",
            "dish_name": "seasonal tasting menu",
            "price": "$395",
            "metadata": {"borough": "Manhattan", "style": "fine dining"},
        },
        {
            "doc_id": "doc_002",
            "restaurant_id": "rest_le_bernardin",
            "restaurant_name": "Le Bernardin",
            "content_type": "menu",
            "text": "Older but very relevant seafood tasting menu entry with excellent semantic match.",
            "vector": [0.99, 0.18, 0.04, 0.02],
            "created_at": (now - timedelta(days=10)).isoformat(),
            "source": "official_menu",
            "dish_name": "chef tasting menu",
            "price": "$340",
            "metadata": {"borough": "Manhattan", "style": "seafood tasting"},
        },
        {
            "doc_id": "doc_003",
            "restaurant_id": "rest_yoshino",
            "restaurant_name": "Yoshino",
            "content_type": "review",
            "text": "Recent omakase discussion mentioning progression, pacing, and premium ingredients.",
            "vector": [0.75, 0.45, 0.15, 0.06],
            "created_at": (now - timedelta(days=2)).isoformat(),
            "source": "google_maps",
            "dish_name": "omakase",
            "price": "$495",
            "metadata": {"borough": "Manhattan", "style": "omakase"},
        },
        {
            "doc_id": "doc_004",
            "restaurant_id": "rest_sushi_noz",
            "restaurant_name": "Sushi Noz",
            "content_type": "review",
            "text": "Historic but semantically close writeup of a premium sushi counter experience.",
            "vector": [0.96, 0.19, 0.08, 0.01],
            "created_at": (now - timedelta(days=45)).isoformat(),
            "source": "reddit",
            "dish_name": "omakase",
            "price": "$550",
            "metadata": {"borough": "Manhattan", "style": "omakase"},
        },
        {
            "doc_id": "doc_005",
            "restaurant_id": "rest_kono",
            "restaurant_name": "Kono",
            "content_type": "review",
            "text": "Very fresh but semantically weaker yakitori review used as a control example.",
            "vector": [0.10, 0.92, 0.21, 0.08],
            "created_at": (now - timedelta(hours=3)).isoformat(),
            "source": "reddit",
            "dish_name": "yakitori omakase",
            "price": "$285",
            "metadata": {"borough": "Manhattan", "style": "yakitori"},
        },
    ]


def initialize_dummy_db(*, reset: bool = True):
    rows = make_dummy_documents()
    validate_rows(rows)
    return ensure_table(rows, reset=reset)


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)?")

TOKEN_ALIASES: dict[str, list[str]] = {
    "chicken": ["chicken", "gà"],
    "beef": ["beef", "wagyu", "short rib", "tenderloin", "sirloin"],
    "pork": ["pork", "sausage", "ham", "bacon"],
    "duck": ["duck"],
    "lamb": ["lamb"],
    "lobster": ["lobster"],
    "shrimp": ["shrimp", "prawn"],
    "crab": ["crab"],
    "tuna": ["tuna", "bluefin", "yellowfin"],
    "salmon": ["salmon"],
    "mushroom": ["mushroom", "morels", "maitake", "enoki", "nameko", "shiitake"],
    "vegetarian": ["vegetarian", "vegetable", "vegan", "greens", "tofu"],
    "dessert": ["dessert", "ice cream", "sorbet", "pudding", "pie", "donut", "shaved ice", "cookie"],
    "omakase": ["omakase"],
    "noodle": ["noodle", "pho", "ramen", "pasta", "fettuccine", "tagliatelle", "lumache", "ravioli"],
}

STRICT_TERMS = {
    "chicken", "beef", "pork", "duck", "lamb", "lobster", "shrimp", "crab", "tuna", "salmon",
    "mushroom", "vegetarian", "omakase", "dessert", "noodle",
}

STYLE_TERMS = {"omakase", "dessert", "vegetarian", "yakitori", "seafood", "korean", "sushi"}


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text.lower())]


def canonicalize_token(token: str) -> str:
    lowered = token.lower().strip()
    for canonical, aliases in TOKEN_ALIASES.items():
        if lowered == canonical or lowered in aliases:
            return canonical
    return lowered


def get_row_text_blob(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("restaurant_name") or ""),
        str(row.get("dish_name") or ""),
        str(row.get("text") or ""),
    ]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for value in metadata.values():
            parts.append(str(value))
    return " ".join(parts).lower()


def parse_query_rule_based(query_text: str) -> QueryUnderstanding:
    tokens = [canonicalize_token(tok) for tok in tokenize(query_text)]
    normalized = " ".join(tokens)
    must_include: list[str] = []
    should_include: list[str] = []
    cuisine_or_style: list[str] = []

    for tok in tokens:
        if tok in STRICT_TERMS and tok not in must_include:
            must_include.append(tok)
        elif tok not in should_include:
            should_include.append(tok)
        if tok in STYLE_TERMS and tok not in cuisine_or_style:
            cuisine_or_style.append(tok)

    strict_filter = bool(must_include)
    content_type_preference = "menu" if strict_filter or any(tok in {"dish", "menu", "dessert", "omakase", "noodle"} for tok in tokens) else None

    excludes: list[str] = []
    if "vegetarian" in must_include:
        excludes.extend(["beef", "pork", "duck", "lamb", "chicken", "lobster", "shrimp", "crab", "tuna", "salmon"])

    return QueryUnderstanding(
        parser="rule_based",
        original_query=query_text,
        normalized_query=normalized or query_text.lower().strip(),
        semantic_query=" ".join(must_include + should_include).strip() or query_text.strip(),
        must_include=must_include,
        should_include=should_include,
        exclude=excludes,
        cuisine_or_style=cuisine_or_style,
        content_type_preference=content_type_preference,
        strict_filter=strict_filter,
    )


def understand_query(query_text: str) -> QueryUnderstanding:
    # Optional future API hook can replace this function; this fallback stays deterministic for local testing.
    return parse_query_rule_based(query_text)


def row_matches_understanding(row: dict[str, Any], understanding: QueryUnderstanding, *, default_content_type: str | None) -> bool:
    blob = get_row_text_blob(row)
    row_content_type = str(row.get("content_type") or "")
    expected_content_type = default_content_type or understanding.content_type_preference
    if expected_content_type and row_content_type != expected_content_type:
        return False

    for term in understanding.exclude:
        aliases = TOKEN_ALIASES.get(term, [term])
        if any(alias.lower() in blob for alias in aliases):
            return False

    if not understanding.strict_filter:
        return True

    for term in understanding.must_include:
        aliases = TOKEN_ALIASES.get(term, [term])
        if not any(alias.lower() in blob for alias in aliases):
            return False
    return True


def lexical_bonus_for_row(row: dict[str, Any], understanding: QueryUnderstanding) -> float:
    blob = get_row_text_blob(row)
    dish_name = str(row.get("dish_name") or "").lower()
    bonus = 0.0

    for term in understanding.must_include:
        aliases = TOKEN_ALIASES.get(term, [term])
        if any(alias.lower() in dish_name for alias in aliases):
            bonus += 0.08
        elif any(alias.lower() in blob for alias in aliases):
            bonus += 0.04

    for term in understanding.should_include:
        aliases = TOKEN_ALIASES.get(term, [term])
        if any(alias.lower() in dish_name for alias in aliases):
            bonus += 0.03
        elif any(alias.lower() in blob for alias in aliases):
            bonus += 0.015

    return bonus


def apply_filters(row: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if row.get(key) != expected:
            return False
    return True


def retrieve_with_time_decay(
    query_vector: Iterable[float],
    *,
    top_k: int = 5,
    candidate_pool: int = 20,
    half_life_days: float = 7.0,
    alpha: float = 0.85,
    filters: dict[str, Any] | None = None,
    understanding: QueryUnderstanding | None = None,
    default_content_type: str | None = None,
) -> list[RankedResult]:
    if candidate_pool < top_k:
        candidate_pool = top_k

    db = connect_db()
    table = _open_table_if_exists(db, TABLE_NAME)
    if table is None:
        raise ValueError("Table does not exist. Run initialize_dummy_db() or ingest_documents() first.")

    query = normalize(query_vector)
    lambda_ = compute_lambda(half_life_days)

    raw_results = (
        table.search(query)
        .distance_type("cosine")
        .limit(candidate_pool)
        .to_list()
    )

    now = utc_now()
    ranked: list[RankedResult] = []

    for row in raw_results:
        if not apply_filters(row, filters):
            continue
        if understanding and not row_matches_understanding(row, understanding, default_content_type=default_content_type):
            continue

        created_at_dt = parse_timestamp(row["created_at"])
        age_days = max((now - created_at_dt).total_seconds() / 86400.0, 0.0)
        cosine_distance = float(row["_distance"])
        semantic_similarity = max(0.0, 1.0 - cosine_distance)
        decay_factor = exponential_decay(age_days, lambda_)
        freshness_adjustment, base_score = blend_similarity_with_decay(
            semantic_similarity=semantic_similarity,
            decay_factor=decay_factor,
            alpha=alpha,
        )

        lexical_bonus = lexical_bonus_for_row(row, understanding) if understanding else 0.0
        final_score = min(base_score + lexical_bonus, 1.5)
        trending_badge = age_days <= 7 and decay_factor >= 0.5

        ranked.append(
            RankedResult(
                doc_id=row["doc_id"],
                restaurant_id=row["restaurant_id"],
                restaurant_name=row["restaurant_name"],
                content_type=row["content_type"],
                source=row["source"],
                dish_name=row.get("dish_name"),
                price=row.get("price"),
                created_at=row["created_at"],
                age_days=age_days,
                cosine_distance=cosine_distance,
                semantic_similarity=semantic_similarity,
                decay_factor=decay_factor,
                freshness_adjustment=freshness_adjustment,
                lexical_bonus=lexical_bonus,
                final_score=final_score,
                trending_badge=trending_badge,
                text=row["text"],
            )
        )

    ranked.sort(key=lambda item: item.final_score, reverse=True)
    return ranked[:top_k]


def dedupe_by_restaurant(results: list[RankedResult], limit: int) -> list[RankedResult]:
    seen: set[str] = set()
    unique: list[RankedResult] = []
    for item in results:
        if item.restaurant_name in seen:
            continue
        seen.add(item.restaurant_name)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def get_default_query_vector() -> list[float]:
    db = connect_db()
    table = _open_table_if_exists(db, TABLE_NAME)
    if table is None:
        raise ValueError("Table does not exist.")
    sample = table.search().limit(1).to_list()
    if not sample:
        raise ValueError("Table is empty.")
    return normalize([0.2] * len(sample[0]["vector"]))


def search_by_text(
    query_text: str,
    *,
    top_k: int = 10,
    candidate_pool: int = 100,
    half_life_days: float = 7.0,
    alpha: float = 0.85,
    dedupe_restaurants: bool = True,
    default_content_type: str | None = "menu",
) -> tuple[QueryUnderstanding, list[RankedResult]]:
    understanding = understand_query(query_text)
    query_vector = get_default_query_vector()
    results = retrieve_with_time_decay(
        query_vector,
        top_k=max(top_k, 50 if dedupe_restaurants else top_k),
        candidate_pool=max(candidate_pool, 200 if dedupe_restaurants else candidate_pool),
        half_life_days=half_life_days,
        alpha=alpha,
        understanding=understanding,
        default_content_type=default_content_type,
    )
    if dedupe_restaurants:
        results = dedupe_by_restaurant(results, limit=top_k)
    else:
        results = results[:top_k]
    return understanding, results


def search_api(
    query_text: str,
    *,
    top_k: int = 10,
    half_life_days: float = 7.0,
    alpha: float = 0.85,
    dedupe_restaurants: bool = True,
    default_content_type: str | None = "menu",
) -> dict[str, Any]:
    understanding, results = search_by_text(
        query_text,
        top_k=top_k,
        candidate_pool=max(100, top_k * 10),
        half_life_days=half_life_days,
        alpha=alpha,
        dedupe_restaurants=dedupe_restaurants,
        default_content_type=default_content_type,
    )
    return {
        "query_understanding": asdict(understanding),
        "results": results_to_dicts(results),
    }


def results_to_dicts(results: list[RankedResult]) -> list[dict[str, Any]]:
    return [asdict(item) for item in results]


def print_understanding(understanding: QueryUnderstanding) -> None:
    print("\nQuery understanding")
    print("=" * 80)
    print(f"parser        : {understanding.parser}")
    print(f"normalized    : {understanding.normalized_query}")
    print(f"semantic_query: {understanding.semantic_query}")
    print(f"must_include  : {understanding.must_include}")
    print(f"should_include: {understanding.should_include}")
    print(f"exclude       : {understanding.exclude}")
    print(f"style         : {understanding.cuisine_or_style}")
    print(f"strict_filter : {understanding.strict_filter}")
    print(f"content_pref  : {understanding.content_type_preference}")


def print_results(results: list[RankedResult], *, half_life_days: float, alpha: float) -> None:
    print(f"\nRetrieval results (half_life={half_life_days}d, alpha={alpha})")
    print("=" * 154)
    header = (
        f"{'rank':<5} {'restaurant':<20} {'type':<8} {'age_days':>9} {'sem_sim':>10} {'decay':>10} "
        f"{'fresh_adj':>10} {'lexical':>9} {'final':>10} {'trend':>7}  dish"
    )
    print(header)
    print("-" * 154)

    for idx, item in enumerate(results, start=1):
        print(
            f"{idx:<5} {item.restaurant_name:<20} {item.content_type:<8} {item.age_days:>9.2f} {item.semantic_similarity:>10.4f} "
            f"{item.decay_factor:>10.4f} {item.freshness_adjustment:>10.4f} {item.lexical_bonus:>9.4f} {item.final_score:>10.4f} "
            f"{str(item.trending_badge):>7}  {item.dish_name or '-'}"
        )


def demo_from_dummy_data() -> None:
    initialize_dummy_db(reset=True)
    query_vector = normalize([1.0, 0.2, 0.05, 0.01])

    print(f"Initialized local LanceDB at: {DB_PATH}")
    print(f"Table name: {TABLE_NAME}")

    for half_life in (1.0, 7.0, 30.0):
        results = retrieve_with_time_decay(
            query_vector,
            top_k=5,
            candidate_pool=10,
            half_life_days=half_life,
            alpha=0.85,
        )
        print_results(results, half_life_days=half_life, alpha=0.85)


def demo_from_jsonl(path: str | Path) -> None:
    rows = load_jsonl(path)
    ingest_documents(rows, overwrite=True)
    understanding, results = search_by_text(
        "chicken",
        top_k=10,
        candidate_pool=100,
        half_life_days=7.0,
        alpha=0.85,
        dedupe_restaurants=True,
        default_content_type="menu",
    )
    print_understanding(understanding)
    print_results(results, half_life_days=7.0, alpha=0.85)


if __name__ == "__main__":
    week2_path = DATA_DIR / "week2_embeddings.jsonl"

    if week2_path.exists():
        print(f"Using Week 2 embeddings at: {week2_path}")
        demo_from_jsonl(week2_path)
    else:
        print("No Week 2 embeddings found. Falling back to dummy documents.")
        demo_from_dummy_data()
