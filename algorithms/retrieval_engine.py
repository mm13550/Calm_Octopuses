from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import lancedb

from algorithms.retrieval_metadata import get_restaurant_metadata

try:
    import torch
    from transformers import CLIPModel, CLIPTokenizer
except Exception:  # pragma: no cover - dependency availability varies by machine
    torch = None
    CLIPModel = None
    CLIPTokenizer = None


BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "algorithms" else Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "vector_db"
TABLE_NAME = "restaurant_vectors"
MENU_EMBEDDINGS_PATH = DATA_DIR / "embeddings" / "menu_embeddings_latest.jsonl"
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
UTC = timezone.utc


@dataclass
class RankedResult:
    doc_id: str
    restaurant_id: str | None
    restaurant_name: str
    homepage: str | None
    borough: str | None
    michelin_category: str | None
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
    dt = datetime.fromisoformat(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


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


def initialize_menu_table(path: str | Path = MENU_EMBEDDINGS_PATH):
    rows = load_jsonl(path)
    ingest_documents(rows, overwrite=True)
    return rows


def ensure_menu_table_initialized(path: str | Path = MENU_EMBEDDINGS_PATH):
    db = connect_db()
    table = _open_table_if_exists(db, TABLE_NAME)
    if table is not None:
        return table
    if not Path(path).exists():
        raise ValueError("Menu table does not exist and default embeddings file is missing.")
    initialize_menu_table(path)
    return connect_db().open_table(TABLE_NAME)


def make_dummy_documents() -> list[dict[str, Any]]:
    now = utc_now()
    return [
        {
            "doc_id": "doc_001",
            "restaurant_id": "rest_atomix",
            "restaurant_name": "Atomix",
            "content_type": "menu",
            "text": "Inventive Korean tasting menu with a dessert finish.",
            "vector": [0.97, 0.21, 0.05, 0.03],
            "created_at": (now - timedelta(hours=6)).isoformat(),
            "source": "official_menu",
            "dish_name": "seasonal tasting menu",
            "price": "$395",
            "metadata": {"borough": "Manhattan", "style": "fine dining"},
        },
        {
            "doc_id": "doc_002",
            "restaurant_id": "rest_le_bernardin",
            "restaurant_name": "Le Bernardin",
            "content_type": "menu",
            "text": "Seafood tasting menu with elegant fine dining structure.",
            "vector": [0.99, 0.18, 0.04, 0.02],
            "created_at": (now - timedelta(days=10)).isoformat(),
            "source": "official_menu",
            "dish_name": "chef tasting menu",
            "price": "$340",
            "metadata": {"borough": "Manhattan", "style": "seafood tasting"},
        },
    ]


def initialize_dummy_db(*, reset: bool = True):
    rows = make_dummy_documents()
    validate_rows(rows)
    return ensure_table(rows, reset=reset)


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)?")

TOKEN_ALIASES: dict[str, list[str]] = {
    "chicken": ["chicken", "ga"],
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
    "dessert": ["dessert", "ice cream", "sorbet", "pudding", "pie", "donut", "cookie", "cake", "tart"],
    "omakase": ["omakase"],
    "noodle": ["noodle", "pho", "ramen", "pasta", "fettuccine", "tagliatelle", "lumache", "ravioli"],
    "pasta": ["pasta", "fettuccine", "tagliatelle", "lumache", "ravioli", "spaghetti", "rigatoni"],
    "quiet": ["quiet", "calm", "intimate", "cozy"],
}

STRICT_TERMS = {
    "chicken", "beef", "pork", "duck", "lamb", "lobster", "shrimp", "crab", "tuna", "salmon",
    "mushroom", "vegetarian", "omakase", "dessert", "noodle", "pasta",
}

STYLE_TERMS = {"omakase", "dessert", "vegetarian", "yakitori", "seafood", "korean", "sushi", "pasta"}


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text.lower())]


def canonicalize_token(token: str) -> str:
    lowered = token.lower().strip()
    for canonical, aliases in TOKEN_ALIASES.items():
        if lowered == canonical or lowered in aliases:
            return canonical
    return lowered


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
    content_type_preference = "menu"

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
    return parse_query_rule_based(query_text)


def get_row_text_blob(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("restaurant_name") or ""),
        str(row.get("dish_name") or ""),
        str(row.get("text") or ""),
        str(row.get("price") or ""),
    ]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        parts.extend(str(v) for v in metadata.values())
    return " ".join(parts).lower()


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
            bonus += 0.10
        elif any(alias.lower() in blob for alias in aliases):
            bonus += 0.05

    for term in understanding.should_include:
        aliases = TOKEN_ALIASES.get(term, [term])
        if any(alias.lower() in dish_name for alias in aliases):
            bonus += 0.04
        elif any(alias.lower() in blob for alias in aliases):
            bonus += 0.02

    return bonus


_CLIP_CACHE: dict[str, Any] = {}


def _clip_backend_available() -> bool:
    return all(x is not None for x in (torch, CLIPModel, CLIPTokenizer))


def _load_clip_backend(model_id: str = DEFAULT_CLIP_MODEL_ID):
    if not _clip_backend_available():
        raise RuntimeError("CLIP query embedding dependencies are unavailable.")

    cached = _CLIP_CACHE.get(model_id)
    if cached is not None:
        return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()

    _CLIP_CACHE[model_id] = (tokenizer, model, device)
    return tokenizer, model, device


def embed_query_text_clip(query_text: str, *, model_id: str = DEFAULT_CLIP_MODEL_ID) -> list[float]:
    tokenizer, model, device = _load_clip_backend(model_id=model_id)

    with torch.no_grad():
        inputs = tokenizer([query_text], return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.get_text_features(**inputs)

        if isinstance(outputs, torch.Tensor):
            features = outputs
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
            if hasattr(model, "text_projection") and model.text_projection is not None:
                features = model.text_projection(features)
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            features = outputs[0]
        else:
            raise TypeError(f"Unsupported CLIP text output type: {type(outputs)!r}")

        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Failed to extract tensor from CLIP output: {type(features)!r}")

        features = features / features.norm(p=2, dim=-1, keepdim=True)

    return normalize(features[0].detach().cpu().tolist())


def embed_query_text_hash(text: str, dim: int = 512) -> list[float]:
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        tokens = ["empty"]

    for tok in tokens:
        h = hash(tok)
        idx = abs(h) % dim
        sign = 1.0 if h % 2 == 0 else -1.0
        vec[idx] += sign

    return normalize(vec)


def get_vector_dim_from_file(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError("Embedding file is empty.")
    row = json.loads(line)
    return len(row["vector"])


def build_query_vector_from_text(
    query_text: str,
    *,
    backend: str,
    expected_dim: int,
) -> list[float] | None:
    if backend == "auto":
        backend = "clip" if _clip_backend_available() else "lexical"

    if backend == "clip":
        return embed_query_text_clip(query_text)
    if backend == "hash":
        return embed_query_text_hash(query_text, dim=expected_dim)
    if backend == "lexical":
        return None

    raise ValueError(f"Unsupported backend: {backend}")


def _rank_row(
    row: dict[str, Any],
    *,
    cosine_distance: float,
    understanding: QueryUnderstanding | None,
) -> RankedResult:
    restaurant_meta = get_restaurant_metadata(row.get("restaurant_id"), row.get("restaurant_name"))
    created_at_dt = parse_timestamp(row["created_at"])
    age_days = max((utc_now() - created_at_dt).total_seconds() / 86400.0, 0.0)
    semantic_similarity = max(0.0, 1.0 - cosine_distance)
    lexical_bonus = lexical_bonus_for_row(row, understanding) if understanding else 0.0
    final_score = min(semantic_similarity + lexical_bonus, 1.5)

    return RankedResult(
        doc_id=str(row["doc_id"]),
        restaurant_id=restaurant_meta["restaurant_id"],
        restaurant_name=restaurant_meta["restaurant_name"] or str(row.get("restaurant_name") or ""),
        homepage=restaurant_meta["homepage"],
        borough=restaurant_meta["borough"],
        michelin_category=restaurant_meta["michelin_category"],
        content_type=str(row.get("content_type") or ""),
        source=str(row.get("source") or ""),
        dish_name=row.get("dish_name"),
        price=row.get("price"),
        created_at=parse_timestamp(row["created_at"]).isoformat(),
        age_days=age_days,
        cosine_distance=cosine_distance,
        semantic_similarity=semantic_similarity,
        decay_factor=1.0,
        freshness_adjustment=1.0,
        lexical_bonus=lexical_bonus,
        final_score=final_score,
        trending_badge=False,
        text=str(row.get("text") or ""),
    )


def retrieve_ranked(
    query_vector: Iterable[float],
    *,
    top_k: int = 10,
    candidate_pool: int = 100,
    understanding: QueryUnderstanding | None = None,
    default_content_type: str | None = "menu",
) -> list[RankedResult]:
    if candidate_pool < top_k:
        candidate_pool = top_k

    table = ensure_menu_table_initialized()
    query = normalize(query_vector)

    raw_results = (
        table.search(query)
        .distance_type("cosine")
        .limit(candidate_pool)
        .to_list()
    )

    ranked: list[RankedResult] = []
    for row in raw_results:
        if understanding and not row_matches_understanding(row, understanding, default_content_type=default_content_type):
            continue
        ranked.append(
            _rank_row(
                row,
                cosine_distance=float(row["_distance"]),
                understanding=understanding,
            )
        )

    ranked.sort(key=lambda item: item.final_score, reverse=True)
    return ranked[:top_k]


def dedupe_by_restaurant(results: list[RankedResult], limit: int) -> list[RankedResult]:
    seen: set[str] = set()
    unique: list[RankedResult] = []
    for item in results:
        dedupe_key = item.restaurant_id or item.restaurant_name
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def lexical_fallback_search(
    query_text: str,
    *,
    top_k: int = 10,
    dedupe_restaurants: bool = True,
    default_content_type: str | None = "menu",
) -> tuple[QueryUnderstanding, list[RankedResult]]:
    understanding = understand_query(query_text)
    if not MENU_EMBEDDINGS_PATH.exists():
        return understanding, []

    rows = load_jsonl(MENU_EMBEDDINGS_PATH)
    ranked: list[RankedResult] = []

    for row in rows:
        if not row_matches_understanding(row, understanding, default_content_type=default_content_type):
            continue

        lexical_bonus = lexical_bonus_for_row(row, understanding)
        if lexical_bonus <= 0 and understanding.must_include:
            continue

        ranked.append(
            _rank_row(
                row,
                cosine_distance=0.5,
                understanding=understanding,
            )
        )

    ranked.sort(key=lambda item: (item.lexical_bonus, item.final_score), reverse=True)
    if dedupe_restaurants:
        ranked = dedupe_by_restaurant(ranked, limit=top_k)
    else:
        ranked = ranked[:top_k]

    return understanding, ranked


def search_by_text(
    query_text: str,
    *,
    top_k: int = 10,
    candidate_pool: int = 100,
    half_life_days: float = 7.0,
    alpha: float = 0.85,
    dedupe_restaurants: bool = True,
    default_content_type: str | None = "menu",
    embedding_backend: str = "auto",
) -> tuple[QueryUnderstanding, list[RankedResult], str]:
    del half_life_days, alpha
    understanding = understand_query(query_text)

    try:
        expected_dim = get_vector_dim_from_file(MENU_EMBEDDINGS_PATH)
    except Exception:
        expected_dim = 512

    query_vector = build_query_vector_from_text(
        understanding.semantic_query or query_text,
        backend=embedding_backend,
        expected_dim=expected_dim,
    )

    backend_used = embedding_backend
    results: list[RankedResult] = []

    if query_vector is not None:
        backend_used = "clip" if embedding_backend == "auto" and _clip_backend_available() else embedding_backend
        try:
            results = retrieve_ranked(
                query_vector,
                top_k=max(top_k, 50 if dedupe_restaurants else top_k),
                candidate_pool=max(candidate_pool, 200 if dedupe_restaurants else candidate_pool),
                understanding=understanding,
                default_content_type=default_content_type,
            )
        except Exception:
            results = []

    if not results:
        understanding, results = lexical_fallback_search(
            query_text,
            top_k=top_k,
            dedupe_restaurants=dedupe_restaurants,
            default_content_type=default_content_type,
        )
        backend_used = "lexical_fallback"

    if dedupe_restaurants:
        results = dedupe_by_restaurant(results, limit=top_k)
    else:
        results = results[:top_k]

    return understanding, results, backend_used


def results_to_dicts(results: list[RankedResult]) -> list[dict[str, Any]]:
    return [asdict(item) for item in results]


def results_to_simple_dicts(results: list[RankedResult]) -> list[dict[str, Any]]:
    simple_results: list[dict[str, Any]] = []
    for item in results:
        simple_results.append(
            {
                "doc_id": item.doc_id,
                "restaurant_id": item.restaurant_id,
                "restaurant_name": item.restaurant_name,
                "homepage": item.homepage,
                "borough": item.borough,
                "michelin_category": item.michelin_category,
                "dish_name": item.dish_name,
                "price": item.price,
                "score": round(item.final_score, 4),
                "semantic_similarity": round(item.semantic_similarity, 4),
                "lexical_bonus": round(item.lexical_bonus, 4),
                "content_type": item.content_type,
                "source": item.source,
                "text": item.text,
            }
        )
    return simple_results


def search_api(
    query_text: str,
    *,
    top_k: int = 10,
    half_life_days: float = 7.0,
    alpha: float = 0.85,
    dedupe_restaurants: bool = True,
    default_content_type: str | None = "menu",
    embedding_backend: str = "auto",
) -> dict[str, Any]:
    understanding, results, backend_used = search_by_text(
        query_text,
        top_k=top_k,
        candidate_pool=max(100, top_k * 10),
        half_life_days=half_life_days,
        alpha=alpha,
        dedupe_restaurants=dedupe_restaurants,
        default_content_type=default_content_type,
        embedding_backend=embedding_backend,
    )
    return {
        "query": query_text,
        "backend_used": backend_used,
        "query_understanding": asdict(understanding),
        "results": results_to_dicts(results),
    }


def search_api_simple(
    query_text: str,
    *,
    top_k: int = 10,
    half_life_days: float = 7.0,
    alpha: float = 0.85,
    dedupe_restaurants: bool = True,
    default_content_type: str | None = "menu",
    embedding_backend: str = "auto",
) -> dict[str, Any]:
    payload = search_api(
        query_text,
        top_k=top_k,
        half_life_days=half_life_days,
        alpha=alpha,
        dedupe_restaurants=dedupe_restaurants,
        default_content_type=default_content_type,
        embedding_backend=embedding_backend,
    )
    understanding = payload.get("query_understanding", {})
    return {
        "query": query_text,
        "backend_used": payload.get("backend_used"),
        "must_include": understanding.get("must_include", []),
        "results": results_to_simple_dicts([RankedResult(**row) for row in payload.get("results", [])]),
    }


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
    del half_life_days, alpha
    print("\nMenu retrieval results")
    print("=" * 140)
    header = (
        f"{'rank':<5} {'restaurant':<24} {'sem_sim':>10} {'lexical':>9} {'final':>10} "
        f"{'borough':<12} {'dish':<40}"
    )
    print(header)
    print("-" * 140)

    for idx, item in enumerate(results, start=1):
        print(
            f"{idx:<5} {item.restaurant_name:<24} {item.semantic_similarity:>10.4f} "
            f"{item.lexical_bonus:>9.4f} {item.final_score:>10.4f} "
            f"{str(item.borough or '-'): <12} {(item.dish_name or '-'): <40}"
        )


def demo_from_dummy_data() -> None:
    initialize_dummy_db(reset=True)
    query_vector = normalize([1.0, 0.2, 0.05, 0.01])
    results = retrieve_ranked(query_vector, top_k=5, candidate_pool=10)
    print_results(results, half_life_days=7.0, alpha=0.85)


def demo_from_jsonl(path: str | Path) -> None:
    rows = load_jsonl(path)
    ingest_documents(rows, overwrite=True)
    understanding, results, backend_used = search_by_text(
        "chicken",
        top_k=10,
        candidate_pool=100,
        dedupe_restaurants=True,
        default_content_type="menu",
        embedding_backend="auto",
    )
    print(f"backend_used  : {backend_used}")
    print_understanding(understanding)
    print_results(results, half_life_days=7.0, alpha=0.85)


if __name__ == "__main__":
    if MENU_EMBEDDINGS_PATH.exists():
        demo_from_jsonl(MENU_EMBEDDINGS_PATH)
    else:
        demo_from_dummy_data()
