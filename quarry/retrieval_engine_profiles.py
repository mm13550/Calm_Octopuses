from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import lancedb

try:
    import torch
    from transformers import CLIPModel, CLIPTokenizer
except Exception:  # pragma: no cover - dependency availability varies by machine
    torch = None
    CLIPModel = None
    CLIPTokenizer = None


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quarry.retrieval_metadata import get_restaurant_metadata


BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DB_PATH = DATA_DIR / "vector_db"

PROFILE_PATH = EMBEDDINGS_DIR / "restaurant_profiles_latest.jsonl"
TABLE_NAME = "restaurant_profiles"
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
UTC = timezone.utc

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)?")
STRICT_TERMS = {
    "omakase", "dessert", "quiet", "romantic", "cozy", "intimate",
    "staff", "service", "seafood", "sushi", "yakitori", "vegetarian",
    "pasta", "cocktail", "wine", "tasting",
}


@dataclass
class RankedProfile:
    doc_id: str
    restaurant_id: str | None
    restaurant_name: str
    homepage: str | None
    borough: str | None
    michelin_category: str | None
    content_type: str
    source: str
    created_at: str
    cosine_distance: float
    semantic_similarity: float
    lexical_bonus: float
    final_score: float
    text: str
    menu_item_count: int
    review_count: int
    food_image_count: int
    interior_image_count: int
    food_image_paths: list[str]
    interior_image_paths: list[str]
    top_menu_items: list[str]
    top_review_snippets: list[str]


@dataclass
class QueryUnderstanding:
    original_query: str
    normalized_query: str
    semantic_query: str
    must_include: list[str]
    should_include: list[str]


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


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_PATTERN.findall(str(text).lower())]


def understand_query(query_text: str) -> QueryUnderstanding:
    tokens = tokenize(query_text)
    must_include: list[str] = []
    should_include: list[str] = []
    for tok in tokens:
        if tok in STRICT_TERMS and tok not in must_include:
            must_include.append(tok)
        elif tok not in should_include:
            should_include.append(tok)

    semantic_query = " ".join(tokens).strip() or str(query_text).strip()
    normalized_query = " ".join(tokens).strip() or str(query_text).strip().lower()

    return QueryUnderstanding(
        original_query=query_text,
        normalized_query=normalized_query,
        semantic_query=semantic_query,
        must_include=must_include,
        should_include=should_include,
    )


_CLIP_CACHE: dict[str, Any] = {}


def _clip_backend_available() -> bool:
    return torch is not None and CLIPModel is not None and CLIPTokenizer is not None


def _load_clip_backend(model_id: str = DEFAULT_CLIP_MODEL_ID):
    if not _clip_backend_available():
        raise RuntimeError("CLIP dependencies are not available in this environment.")

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
    if torch is None:
        raise RuntimeError("torch is required for CLIP query embeddings.")

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
            raise TypeError(f"Failed to extract tensor from CLIP text output: {type(features)!r}")

        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return normalize(features[0].detach().cpu().tolist())


def connect_db():
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(DB_PATH))


def _open_table_if_exists(db, table_name: str):
    try:
        return db.open_table(table_name)
    except Exception:
        return None


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
            raise ValueError(f"Row {idx} missing required fields: {sorted(missing)}")

        if not isinstance(row["vector"], list) or not row["vector"]:
            raise ValueError(f"Row {idx} must contain a non-empty vector list")

        row["vector"] = normalize(row["vector"])
        row["created_at"] = parse_timestamp(row["created_at"]).isoformat()
        row.setdefault("metadata", {})


def ensure_table(rows: list[dict[str, Any]] | None = None, *, reset: bool = False):
    db = connect_db()
    table = _open_table_if_exists(db, TABLE_NAME)

    if table is None:
        if not rows:
            raise ValueError("Table does not exist yet. Provide rows to create it.")
        return db.create_table(TABLE_NAME, data=rows)

    if reset:
        if not rows:
            raise ValueError("reset=True requires rows.")
        return db.create_table(TABLE_NAME, data=rows, mode="overwrite")

    return table


def ingest_profiles(rows: list[dict[str, Any]], *, overwrite: bool = True):
    validate_rows(rows)
    db = connect_db()
    existing = _open_table_if_exists(db, TABLE_NAME)

    if overwrite or existing is None:
        return ensure_table(rows, reset=True if existing is not None else False)

    existing.add(rows)
    return existing


def initialize_profiles_table(path: str | Path = PROFILE_PATH):
    rows = load_jsonl(path)
    ingest_profiles(rows, overwrite=True)
    return rows


def ensure_profiles_table_initialized(path: str | Path = PROFILE_PATH):
    db = connect_db()
    table = _open_table_if_exists(db, TABLE_NAME)
    if table is not None:
        return table
    if not Path(path).exists():
        raise ValueError("Profile table does not exist and default embeddings file is missing.")
    initialize_profiles_table(path)
    return connect_db().open_table(TABLE_NAME)


def get_row_blob(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    parts = [
        str(row.get("restaurant_name") or ""),
        str(row.get("text") or ""),
        " ".join(str(x) for x in metadata.get("top_menu_items", []) if x),
        " ".join(str(x) for x in metadata.get("top_review_snippets", []) if x),
    ]
    return " ".join(parts).lower()


def row_matches_query(row: dict[str, Any], understanding: QueryUnderstanding) -> bool:
    if not understanding.must_include:
        return True
    blob = get_row_blob(row)
    return all(term in blob for term in understanding.must_include)


def lexical_bonus_for_row(row: dict[str, Any], understanding: QueryUnderstanding) -> float:
    blob = get_row_blob(row)
    bonus = 0.0
    for term in understanding.must_include:
        if term in blob:
            bonus += 0.08
    for term in understanding.should_include:
        if term in blob:
            bonus += 0.03
    return bonus


def _rank_row(
    row: dict[str, Any],
    *,
    cosine_distance: float,
    understanding: QueryUnderstanding | None,
) -> RankedProfile:
    metadata = row.get("metadata") or {}
    restaurant_meta = get_restaurant_metadata(row.get("restaurant_id"), row.get("restaurant_name"))
    semantic_similarity = max(0.0, 1.0 - cosine_distance)
    lexical_bonus = lexical_bonus_for_row(row, understanding) if understanding else 0.0
    final_score = min(semantic_similarity + lexical_bonus, 1.5)

    return RankedProfile(
        doc_id=str(row["doc_id"]),
        restaurant_id=restaurant_meta["restaurant_id"],
        restaurant_name=restaurant_meta["restaurant_name"] or str(row.get("restaurant_name") or ""),
        homepage=restaurant_meta["homepage"],
        borough=restaurant_meta["borough"],
        michelin_category=restaurant_meta["michelin_category"],
        content_type=str(row.get("content_type") or ""),
        source=str(row.get("source") or ""),
        created_at=parse_timestamp(row["created_at"]).isoformat(),
        cosine_distance=cosine_distance,
        semantic_similarity=semantic_similarity,
        lexical_bonus=lexical_bonus,
        final_score=final_score,
        text=str(row.get("text") or ""),
        menu_item_count=int(metadata.get("menu_item_count", 0)),
        review_count=int(metadata.get("review_count", 0)),
        food_image_count=int(metadata.get("food_image_count", 0)),
        interior_image_count=int(metadata.get("interior_image_count", 0)),
        food_image_paths=[str(x) for x in metadata.get("food_image_paths", []) if x],
        interior_image_paths=[str(x) for x in metadata.get("interior_image_paths", []) if x],
        top_menu_items=[str(x) for x in metadata.get("top_menu_items", []) if x],
        top_review_snippets=[str(x) for x in metadata.get("top_review_snippets", []) if x],
    )


def retrieve_profiles(
    query_vector: Iterable[float],
    *,
    top_k: int = 10,
    candidate_pool: int = 100,
    understanding: QueryUnderstanding | None = None,
) -> list[RankedProfile]:
    if candidate_pool < top_k:
        candidate_pool = top_k

    table = ensure_profiles_table_initialized()
    query = normalize(query_vector)
    raw_results = table.search(query).distance_type("cosine").limit(candidate_pool).to_list()

    ranked: list[RankedProfile] = []
    for row in raw_results:
        if understanding and not row_matches_query(row, understanding):
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


def lexical_fallback_profiles(
    query_text: str,
    *,
    top_k: int = 10,
) -> tuple[QueryUnderstanding, list[RankedProfile]]:
    understanding = understand_query(query_text)
    if not PROFILE_PATH.exists():
        return understanding, []

    rows = load_jsonl(PROFILE_PATH)
    ranked: list[RankedProfile] = []
    for row in rows:
        if not row_matches_query(row, understanding):
            continue

        lexical_bonus = lexical_bonus_for_row(row, understanding)
        if lexical_bonus <= 0 and (understanding.must_include or understanding.should_include):
            continue

        ranked.append(
            _rank_row(
                row,
                cosine_distance=0.5,
                understanding=understanding,
            )
        )

    ranked.sort(key=lambda item: (item.lexical_bonus, item.final_score), reverse=True)
    return understanding, ranked[:top_k]


def results_to_dicts(results: list[RankedProfile]) -> list[dict[str, Any]]:
    return [asdict(item) for item in results]


def results_to_simple_dicts(results: list[RankedProfile]) -> list[dict[str, Any]]:
    simple: list[dict[str, Any]] = []
    for item in results:
        simple.append(
            {
                "doc_id": item.doc_id,
                "restaurant_id": item.restaurant_id,
                "restaurant_name": item.restaurant_name,
                "homepage": item.homepage,
                "borough": item.borough,
                "michelin_category": item.michelin_category,
                "score": round(item.final_score, 4),
                "semantic_similarity": round(item.semantic_similarity, 4),
                "lexical_bonus": round(item.lexical_bonus, 4),
                "menu_item_count": item.menu_item_count,
                "review_count": item.review_count,
                "food_image_count": item.food_image_count,
                "interior_image_count": item.interior_image_count,
                "food_image_paths": item.food_image_paths,
                "interior_image_paths": item.interior_image_paths,
                "top_menu_items": item.top_menu_items,
                "top_review_snippets": item.top_review_snippets,
                "content_type": item.content_type,
                "source": item.source,
                "text": item.text,
            }
        )
    return simple


def search_profiles_api(
    query_text: str,
    *,
    top_k: int = 10,
    candidate_pool: int = 100,
    embedding_backend: str = "auto",
) -> dict[str, Any]:
    understanding = understand_query(query_text)

    backend_used = embedding_backend
    results: list[RankedProfile] = []

    if embedding_backend == "auto":
        backend_used = "clip" if _clip_backend_available() else "lexical_fallback"

    if backend_used == "clip":
        try:
            query_vector = embed_query_text_clip(understanding.semantic_query or query_text)
            results = retrieve_profiles(
                query_vector,
                top_k=top_k,
                candidate_pool=max(candidate_pool, top_k * 10),
                understanding=understanding,
            )
        except Exception:
            results = []
            backend_used = "lexical_fallback"

    if not results:
        understanding, results = lexical_fallback_profiles(query_text, top_k=top_k)
        backend_used = "lexical_fallback"

    return {
        "query": query_text,
        "backend_used": backend_used,
        "must_include": understanding.must_include,
        "results": results_to_simple_dicts(results),
    }


if __name__ == "__main__":
    if PROFILE_PATH.exists():
        print(f"Using restaurant profiles at: {PROFILE_PATH}")
        initialize_profiles_table(PROFILE_PATH)
        demo = search_profiles_api("quiet omakase", top_k=10)
        print(json.dumps(demo, ensure_ascii=False, indent=2))
    else:
        print(f"Restaurant profiles file not found: {PROFILE_PATH}")
