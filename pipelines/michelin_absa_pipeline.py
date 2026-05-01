"""
Michelin Restaurant — Aspect-Based Sentiment Analysis (ABSA) Pipeline
======================================================================
  data/
  ├── csv/social_reviews.csv
  ├── csv/restaurant_lookup.csv
  └── embeddings/review_embeddings.jsonl

pip install pandas numpy scikit-learn umap-learn hdbscan plotly transformers torch tqdm
python michelin_absa_pipeline.py
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import json, re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go

from transformers import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import hdbscan


# ── 1. Config ─────────────────────────────────────────────────────────────────
DATA_DIR         = Path("data")
REVIEWS_CSV      = DATA_DIR / "csv" / "social_reviews.csv"
EMBEDDINGS_JSONL = DATA_DIR / "embeddings" / "review_embeddings.jsonl"
LOOKUP_CSV       = DATA_DIR / "csv" / "restaurant_lookup.csv"
CACHE_PATH       = Path("absa_cache.jsonl")

#keywords
ASPECTS = {
    "food_quality": ["food", "dish", "flavor", "taste", "fresh", "delicious", "cuisine",
                     "ingredient", "menu", "chef", "portion", "quality", "meal", "cooked"],
    "service":      ["service", "staff", "waiter", "waitress", "server", "attentive",
                     "friendly", "rude", "helpful", "hospitality", "host", "manager"],
    "ambiance":     ["ambiance", "atmosphere", "decor", "interior", "vibe", "setting",
                     "environment", "cozy", "romantic", "noise", "loud", "quiet", "beautiful"],
    "value":        ["price", "value", "worth", "expensive", "cheap", "affordable",
                     "overpriced", "cost", "bill", "money", "pricey"],
    "wait_time":    ["wait", "reservation", "table", "slow", "fast", "quick", "prompt",
                     "long wait", "seated", "delay", "minutes"],
}


# ── 2. Load data ──────────────────────────────────────────────────────────────
def load_data():
    reviews = pd.read_csv(REVIEWS_CSV)
    print(f"✓ Loaded {len(reviews)} reviews")

    embeddings = []
    with open(EMBEDDINGS_JSONL) as f:
        for line in f:
            r = json.loads(line)
            embeddings.append({
                "doc_id":          r["doc_id"],
                "restaurant_id":   r["restaurant_id"],
                "restaurant_name": r["restaurant_name"],
                "text":            r["text"],
                "rating":          r.get("rating"),
                "vector":          r["vector"],
            })
    emb_df = pd.DataFrame(embeddings)
    print(f"✓ Loaded {len(emb_df)} review embeddings")

    lookup = pd.read_csv(LOOKUP_CSV)
    print(f"✓ Loaded {len(lookup)} restaurants")

    return reviews, emb_df, lookup


# ── 3. ABSA────────────────────────────────────────
def extract_aspect_sentences(text: str) -> dict:
    """
    把一条 review 按句子切开，找出和每个 aspect 相关的句子。
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    aspect_sentences = {asp: [] for asp in ASPECTS}

    for sent in sentences:
        sent_lower = sent.lower()
        for asp, keywords in ASPECTS.items():
            if any(kw in sent_lower for kw in keywords):
                aspect_sentences[asp].append(sent)

    return aspect_sentences


def run_absa(emb_df: pd.DataFrame) -> pd.DataFrame:
    """
    对每条 review 的每个 aspect 跑 sentiment model，
    输出 1-5 分（negative=1-2, neutral=3, positive=4-5）。
    用本地 cache 避免重复计算。
    """
    print("\n[Step 2] Loading sentiment model (first run downloads ~250MB)...")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )
    print("✓ Model ready")

    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["doc_id"]] = rec["scores"]
        print(f"✓ Cache loaded: {len(cache)} records")

    def sentiment_to_score(label: str, score: float) -> float:
        """Convert POSITIVE/NEGATIVE + confidence → 1-5 scale."""
        if label == "POSITIVE":
            return 3.0 + 2.0 * score   # 3.0 – 5.0
        else:
            return 3.0 - 2.0 * score   # 1.0 – 3.0

    results = []
    with open(CACHE_PATH, "a") as cache_file:
        for _, row in tqdm(emb_df.iterrows(), total=len(emb_df), desc="ABSA"):
            doc_id = row["doc_id"]

            if doc_id in cache:
                scores = cache[doc_id]
            else:
                aspect_sents = extract_aspect_sentences(row["text"])
                scores = {}
                for asp, sents in aspect_sents.items():
                    if not sents:
                        scores[asp] = None
                        continue
                    combined = " ".join(sents)[:400]
                    result = sentiment_model(combined)[0]
                    scores[asp] = round(sentiment_to_score(result["label"], result["score"]), 2)

                cache[doc_id] = scores
                cache_file.write(json.dumps({"doc_id": doc_id, "scores": scores}) + "\n")

            results.append({
                "doc_id":          row["doc_id"],
                "restaurant_id":   row["restaurant_id"],
                "restaurant_name": row["restaurant_name"],
                "rating":          row["rating"],
                **{f"asp_{k}": v for k, v in scores.items()},
            })

    return pd.DataFrame(results)


# ── 4. Aggregate per restaurant ───────────────────────────────────────────────
def aggregate_by_restaurant(absa_df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    asp_cols = [f"asp_{a}" for a in ASPECTS]

    agg = (
        absa_df
        .groupby(["restaurant_id", "restaurant_name"])[asp_cols + ["rating"]]
        .agg(lambda x: round(x.dropna().mean(), 2) if x.dropna().size > 0 else np.nan)
        .reset_index()
    )

    review_counts = absa_df.groupby("restaurant_id")["doc_id"].count().reset_index()
    review_counts.columns = ["restaurant_id", "review_count"]
    agg = agg.merge(review_counts, on="restaurant_id", how="left")

    agg = agg.merge(
        lookup[["rest_id", "borough", "michelin_category"]],
        left_on="restaurant_id", right_on="rest_id", how="left"
    )
    return agg


# ── 5. Clustering ─────────────────────────────────────────────────────────────
def cluster_embedding_space(emb_df: pd.DataFrame) -> pd.DataFrame:
    """UMAP + HDBSCAN on raw review embeddings → topic clusters."""
    print("\n[Step 4] Running UMAP + HDBSCAN on review embeddings...")
    vectors = np.array(emb_df["vector"].tolist())

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(vectors)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
    labels = clusterer.fit_predict(coords)

    emb_df = emb_df.copy()
    emb_df["umap_x"]  = coords[:, 0]
    emb_df["umap_y"]  = coords[:, 1]
    emb_df["cluster"] = labels
    print(f"✓ Found {emb_df['cluster'].nunique()} clusters")
    return emb_df


def cluster_by_aspect_profile(restaurant_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """KMeans on 5-dim aspect score profile → restaurant type clusters."""
    asp_cols = [f"asp_{a}" for a in ASPECTS]
    profile  = restaurant_df[asp_cols].fillna(restaurant_df[asp_cols].mean())
    scaled   = StandardScaler().fit_transform(profile)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    restaurant_df = restaurant_df.copy()
    restaurant_df["aspect_cluster"] = km.fit_predict(scaled)
    return restaurant_df


# ── 6. Visualizations ────────────────────────────────────────────────────────
def plot_umap_reviews(emb_df: pd.DataFrame):
    fig = px.scatter(
        emb_df,
        x="umap_x", y="umap_y",
        color=emb_df["cluster"].astype(str),
        hover_data=["restaurant_name", "text"],
        title="Review Embedding Space (UMAP) — colored by cluster",
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.6))
    fig.write_html("umap_reviews.html")
    print("✓ Saved: umap_reviews.html")


def plot_restaurant_radar(restaurant_df: pd.DataFrame, top_n: int = 12):
    asp_cols = [f"asp_{a}" for a in ASPECTS]
    labels   = [a.replace("_", " ").title() for a in ASPECTS]
    top      = restaurant_df.nlargest(top_n, "review_count")

    fig = go.Figure()
    for _, row in top.iterrows():
        values = [row[c] if pd.notna(row[c]) else 3 for c in asp_cols]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=row["restaurant_name"][:25],
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
        title=f"Aspect Scores — Top {top_n} Most-Reviewed Restaurants",
        template="plotly_dark",
    )
    fig.write_html("radar_charts.html")
    print("✓ Saved: radar_charts.html")


def plot_heatmap(restaurant_df: pd.DataFrame):
    asp_cols = [f"asp_{a}" for a in ASPECTS]
    df_plot  = (
        restaurant_df
        .dropna(subset=asp_cols, how="all")
        .sort_values(["aspect_cluster", "rating"], ascending=[True, False])
        .reset_index(drop=True)
    )
    matrix = df_plot[asp_cols].fillna(3).values

    fig = px.imshow(
        matrix.T,
        x=df_plot["restaurant_name"].str[:20],
        y=[a.replace("_", " ").title() for a in ASPECTS],
        color_continuous_scale="RdYlGn",
        zmin=1, zmax=5,
        title="Aspect Score Heatmap — All Restaurants",
        aspect="auto",
    )
    fig.update_layout(xaxis_tickangle=-45, height=500, template="plotly_dark")
    fig.write_html("heatmap_aspects.html")
    print("✓ Saved: heatmap_aspects.html")


def plot_michelin_comparison(restaurant_df: pd.DataFrame):
    asp_cols = [f"asp_{a}" for a in ASPECTS]
    melted   = restaurant_df.melt(
        id_vars=["michelin_category"],
        value_vars=asp_cols,
        var_name="aspect", value_name="score",
    )
    melted["aspect"] = melted["aspect"].str.replace("asp_", "").str.replace("_", " ").str.title()

    fig = px.box(
        melted.dropna(subset=["score", "michelin_category"]),
        x="aspect", y="score",
        color="michelin_category",
        title="Aspect Scores by Michelin Category",
        template="plotly_dark",
        points="outliers",
    )
    fig.update_layout(yaxis_range=[1, 5])
    fig.write_html("michelin_comparison.html")
    print("✓ Saved: michelin_comparison.html")


# ── 7. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Michelin ABSA Pipeline  (no API key needed)")
    print("=" * 55)

    # Step 1
    print("\n[Step 1] Loading data...")
    reviews, emb_df, lookup = load_data()

    # Step 2 — ABSA (cached after first run)
    absa_df = run_absa(emb_df)
    absa_df.to_csv("absa_results.csv", index=False)
    print(f"✓ Saved absa_results.csv  ({len(absa_df)} reviews scored)")

    # Step 3 — Aggregate
    print("\n[Step 3] Aggregating per restaurant...")
    restaurant_df = aggregate_by_restaurant(absa_df, lookup)
    restaurant_df = cluster_by_aspect_profile(restaurant_df)
    restaurant_df.to_csv("restaurant_profiles.csv", index=False)
    print(f"✓ Saved restaurant_profiles.csv  ({len(restaurant_df)} restaurants)")

    # Step 4 — Cluster embedding space
    emb_clustered = cluster_embedding_space(emb_df)
    emb_clustered[["doc_id", "restaurant_name", "umap_x", "umap_y", "cluster"]].to_csv(
        "review_clusters.csv", index=False
    )

    # Step 5 — Visualize
    print("\n[Step 5] Generating visualizations...")
    plot_umap_reviews(emb_clustered)
    plot_restaurant_radar(restaurant_df, top_n=12)
    plot_heatmap(restaurant_df)
    plot_michelin_comparison(restaurant_df)

    print("\n✅ Done! Output files:")
    print("   absa_results.csv         — per-review aspect scores")
    print("   restaurant_profiles.csv  — per-restaurant aggregated scores + cluster")
    print("   review_clusters.csv      — UMAP coords + cluster per review")
    print("   umap_reviews.html        — interactive UMAP scatter")
    print("   radar_charts.html        — radar charts for top restaurants")
    print("   heatmap_aspects.html     — full restaurant × aspect heatmap")
    print("   michelin_comparison.html — aspect scores by Michelin tier")
