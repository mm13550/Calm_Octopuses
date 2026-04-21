from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
EMBEDDINGS_PATH = DATA_DIR / "embeddings" / "review_embeddings_latest.jsonl"
TABLE_NAME = "review_vectors"
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
UTC = timezone.utc


@dataclass
class RankedReviewResult:
    doc_id: str
    restaurant_id: str | None
    restaurant_name: str
    homepage: str | None
    borough: str | None
    michelin_category: str | None
    content_type: str
    source: str
    rating: float | None
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
class ReviewQueryUnderstanding:
    original_query: str
    normalized_query: str
    must_include: list[str]
    should_include: list[str]
    exclude: list[str]


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)?")
POSITIVE_STYLE_TERMS = {
    "friendly", "service", "quiet", "romantic", "cozy", "atmosphere", "staff",
    "omakase", "tasting", "sushi", "dessert", "cocktails", "wine", "beef",
    "seafood", "spicy", "vegetarian", "intimate"
}
NEGATION_TERMS = {"not", "without", "no"}


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
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def connect_db():
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(DB_PATH))


def _open_table_if_exists(db, table_name: str):
    try:
        return db.open_table(table_name)
    except Exception:
        return None


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text.lower())]


def understand_review_query(query_text: str) -> ReviewQueryUnderstanding:
    toks = tokenize(query_text)
    must_include: list[str] = []
    should_include: list[str] = []
    exclude: list[str] = []

    prev = None
    for tok in toks:
        if prev in NEGATION_TERMS:
            exclude.append(tok)
        elif tok in POSITIVE_STYLE_TERMS or len(tok) > 3:
            must_include.append(tok)
        else:
            should_include.append(tok)
        prev = tok

    must_include = list(dict.fromkeys(must_include))
    should_include = [t for t in dict.fromkeys(should_include) if t not in must_include]
    exclude = list(dict.fromkeys(exclude))

    return ReviewQueryUnderstanding(
        original_query=query_text,
        normalized_query=" ".join(toks),
        must_include=must_include,
        should_include=should_include,
        exclude=exclude,
    )


def get_row_text_blob(row: dict[str, Any]) -> str:
    return " ".join([
        str(row.get("restaurant_name") or ""),
        str(row.get("text") or ""),
        str(row.get("source") or ""),
    ]).lower()


def row_matches_understanding(row: dict[str, Any], understanding: ReviewQueryUnderstanding) -> bool:
    blob = get_row_text_blob(row)
    for term in understanding.exclude:
        if term in blob:
            return False
    return True


def lexical_bonus_for_row(row: dict[str, Any], understanding: ReviewQueryUnderstanding) -> float:
    blob = get_row_text_blob(row)
    bonus = 0.0
    for term in understanding.must_include:
        if term in blob:
            bonus += 0.04
    for term in understanding.should_include:
        if term in blob:
            bonus += 0.015
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
    tokenizer, model, device = _load_clip_backend(model_id)

    with torch.no_grad():
        inputs = tokenizer([query_text], return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.get_text_features(**inputs)

        if isinstance(outputs, torch.Tensor):
            text_features = outputs
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            text_features = outputs.pooler_output
            if hasattr(model, "text_projection") and model.text_projection is not None:
                text_features = model.text_projection(text_features)
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            text_features = outputs[0]
        else:
            raise TypeError(f"Unsupported CLIP text output type: {type(outputs)!r}")

        if not isinstance(text_features, torch.Tensor):
            raise TypeError(f"Failed to extract tensor from CLIP output: {type(text_features)!r}")

        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    return normalize(text_features[0].detach().cpu().tolist())


def get_vector_dim_from_file(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError("Embedding file is empty.")
    row = json.loads(line)
    return len(row["vector"])


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
            raise ValueError(f"Row {idx} is missing required fields: {sorted(missing)}")
        row["vector"] = normalize(row["vector"])
        row["created_at"] = parse_timestamp(row["created_at"]).isoformat()
        if "rating" in row and row["rating"] not in (None, ""):
            try:
                row["rating"] = float(row["rating"])
            except Exception:
                row["rating"] = None
        else:
            row["rating"] = None


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


def initialize_review_table(path: str | Path = EMBEDDINGS_PATH):
    rows = load_jsonl(path)
    ingest_documents(rows, overwrite=True)
    return rows


def ensure_review_table_initialized(path: str | Path = EMBEDDINGS_PATH):
    db = connect_db()
    table = _open_table_if_exists(db, TABLE_NAME)
    if table is not None:
        return table
    if not Path(path).exists():
        raise ValueError("Review table does not exist and default embeddings file is missing.")
    initialize_review_table(path)
    return connect_db().open_table(TABLE_NAME)


def _rank_row(
    row: dict[str, Any],
    *,
    cosine_distance: float,
    understanding: ReviewQueryUnderstanding | None,
) -> RankedReviewResult:
    restaurant_meta = get_restaurant_metadata(row.get("restaurant_id"), row.get("restaurant_name"))
    created_at_dt = parse_timestamp(row["created_at"])
    age_days = max((utc_now() - created_at_dt).total_seconds() / 86400.0, 0.0)
    semantic_similarity = max(0.0, 1.0 - cosine_distance)
    lexical_bonus = lexical_bonus_for_row(row, understanding) if understanding else 0.0
    final_score = min(semantic_similarity + lexical_bonus, 1.5)

    return RankedReviewResult(
        doc_id=str(row["doc_id"]),
        restaurant_id=restaurant_meta["restaurant_id"],
        restaurant_name=restaurant_meta["restaurant_name"] or str(row.get("restaurant_name") or ""),
        homepage=restaurant_meta["homepage"],
        borough=restaurant_meta["borough"],
        michelin_category=restaurant_meta["michelin_category"],
        content_type=str(row.get("content_type") or ""),
        source=str(row.get("source") or ""),
        rating=row.get("rating"),
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


def retrieve_reviews(
    query_vector: Iterable[float],
    *,
    top_k: int = 10,
    candidate_pool: int = 100,
    half_life_days: float = 30.0,
    freshness_lambda: float = 0.2,
    understanding: ReviewQueryUnderstanding | None = None,
) -> list[RankedReviewResult]:
    del half_life_days, freshness_lambda
    if candidate_pool < top_k:
        candidate_pool = top_k

    table = ensure_review_table_initialized()
    query = normalize(query_vector)
    raw_results = table.search(query).distance_type("cosine").limit(candidate_pool).to_list()

    ranked: list[RankedReviewResult] = []
    for row in raw_results:
        if understanding and not row_matches_understanding(row, understanding):
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


def lexical_fallback_reviews(
    query_text: str,
    *,
    top_k: int = 10,
) -> tuple[ReviewQueryUnderstanding, list[RankedReviewResult]]:
    understanding = understand_review_query(query_text)
    if not EMBEDDINGS_PATH.exists():
        return understanding, []

    rows = load_jsonl(EMBEDDINGS_PATH)
    ranked: list[RankedReviewResult] = []
    for row in rows:
        if not row_matches_understanding(row, understanding):
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
    return understanding, ranked[:top_k]


def results_to_dicts(results: list[RankedReviewResult]) -> list[dict[str, Any]]:
    return [asdict(item) for item in results]


def results_to_simple_dicts(results: list[RankedReviewResult]) -> list[dict[str, Any]]:
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
                "rating": item.rating,
                "score": round(item.final_score, 4),
                "semantic_similarity": round(item.semantic_similarity, 4),
                "lexical_bonus": round(item.lexical_bonus, 4),
                "source": item.source,
                "text": item.text,
            }
        )
    return simple


def search_reviews_api(
    query_text: str,
    *,
    top_k: int = 10,
    candidate_pool: int = 100,
    half_life_days: float = 30.0,
    freshness_lambda: float = 0.2,
    embedding_backend: str = "auto",
) -> dict[str, Any]:
    del half_life_days, freshness_lambda
    understanding = understand_review_query(query_text)

    backend_used = embedding_backend
    results: list[RankedReviewResult] = []

    if embedding_backend == "auto":
        backend_used = "clip" if _clip_backend_available() else "lexical_fallback"

    if backend_used == "clip":
        try:
            query_vector = embed_query_text_clip(understanding.normalized_query or query_text)
            results = retrieve_reviews(
                query_vector,
                top_k=top_k,
                candidate_pool=candidate_pool,
                understanding=understanding,
            )
        except Exception:
            results = []
            backend_used = "lexical_fallback"

    if not results:
        understanding, results = lexical_fallback_reviews(query_text, top_k=top_k)
        backend_used = "lexical_fallback"

    return {
        "query": query_text,
        "backend_used": backend_used,
        "must_include": understanding.must_include,
        "results": results_to_simple_dicts(results),
    }


def print_results(results: list[RankedReviewResult]) -> None:
    print("\nReview retrieval results")
    print("=" * 140)
    header = (
        f"{'rank':<5} {'restaurant':<24} {'rating':<8} {'sem_sim':>10} {'lexical':>9} "
        f"{'final':>10} {'borough':<12} text"
    )
    print(header)
    print("-" * 140)
    for idx, item in enumerate(results, start=1):
        preview = item.text.replace("\n", " ")[:60]
        print(
            f"{idx:<5} {item.restaurant_name:<24} {str(item.rating):<8} {item.semantic_similarity:>10.4f} "
            f"{item.lexical_bonus:>9.4f} {item.final_score:>10.4f} {str(item.borough or '-'): <12} {preview}"
        )


def demo_from_jsonl(path: str | Path) -> None:
    rows = load_jsonl(path)
    ingest_documents(rows, overwrite=True)
    payload = search_reviews_api("friendly staff", top_k=5)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if EMBEDDINGS_PATH.exists():
        demo_from_jsonl(EMBEDDINGS_PATH)
    else:
        raise FileNotFoundError(f"Review embeddings not found: {EMBEDDINGS_PATH}")
