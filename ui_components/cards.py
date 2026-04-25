"""
ui_components/cards.py
======================
Streamlit UI component for rendering a single restaurant result card.

Provides ``_render_result_card``, a reusable card that displays restaurant
metadata, image, coverage metrics, bio, menu highlights, and review snippets.
The rating label is automatically switched between "Your Rating" (actual) and
"Predicted Rating" (MDN output) depending on the row's data.
"""
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any
from core.data_loader import _clean_text, _resolve_path, _truncate

# ── Sentiment data loader ─────────────────────────────────────────────────────
_SENTIMENT_DF: pd.DataFrame | None = None

def _load_sentiment() -> pd.DataFrame:
    """Load restaurant_profiles.csv once and cache it in memory."""
    global _SENTIMENT_DF
    if _SENTIMENT_DF is None:
        candidates = [
            Path(__file__).resolve().parent.parent / "data" / "restaurant_profiles.csv",
            Path(__file__).resolve().parent.parent / "restaurant_profiles.csv",
        ]
        for p in candidates:
            if p.exists():
                _SENTIMENT_DF = pd.read_csv(p).set_index("restaurant_id")
                break
        if _SENTIMENT_DF is None:
            _SENTIMENT_DF = pd.DataFrame()
    return _SENTIMENT_DF


ASP_COLS  = ["asp_food_quality", "asp_service", "asp_ambiance", "asp_value", "asp_wait_time"]
ASP_LABELS = ["Food", "Service", "Ambiance", "Value", "Wait Time"]


def _render_sentiment(rest_id: str) -> None:
    """
    Render sentiment score bars + radar chart for a given restaurant.
    Shows nothing silently if no sentiment data is available.
    """
    df = _load_sentiment()
    if df.empty or rest_id not in df.index:
        return

    row = df.loc[rest_id]
    scores = [row.get(c) for c in ASP_COLS]

    # Only show aspects that actually have a score
    valid = [(label, score) for label, score in zip(ASP_LABELS, scores)
             if score is not None and not (isinstance(score, float) and pd.isna(score))]
    if not valid:
        return

    labels_v, scores_v = zip(*valid)

    st.markdown("**Review Sentiment**")

    # ── Score bars ────────────────────────────────────────────────────────────
    for label, score in valid:
        col_bar, col_num = st.columns([4, 1])
        col_bar.progress(float(score) / 5.0, text=label)
        col_num.markdown(f"**{float(score):.1f}**/5")

    # ── Radar chart ───────────────────────────────────────────────────────────
    r_vals = list(scores_v) + [scores_v[0]]   # close the polygon
    theta  = list(labels_v) + [labels_v[0]]

    fig = go.Figure(go.Scatterpolar(
        r=r_vals,
        theta=theta,
        fill="toself",
        fillcolor="rgba(255, 75, 75, 0.15)",
        line=dict(color="#FF4B4B", width=2),
        name="Sentiment",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"radar_{rest_id}")


# ── Main card renderer ────────────────────────────────────────────────────────
def _render_result_card(row: Dict[str, Any], score_label: str) -> None:
    """
    Render a styled restaurant result card in the current Streamlit column.

    Displays the restaurant name, borough, Michelin category, and a rating
    metric that reads "Your Rating" (actual star value) when the user has
    already rated the restaurant, or "Predicted Rating" (MDN expected mean)
    otherwise.

    Parameters
    ----------
    row : dict
        A single catalog row dict.  Recognised keys: ``restaurant_name``,
        ``rest_id``, ``borough``, ``michelin_category``,
        ``representative_image_path``, ``homepage``, ``menu_count``,
        ``review_count``, ``image_count``, ``bio_text``, ``menu_items``,
        ``review_snippets``, ``score``, ``predicted_rating``, ``actual_rating``.
    score_label : str
        Label text for the primary score metric (e.g. ``"Match Score"``).
    """
    with st.container(border=True):
        left, right = st.columns([1, 2])

        with left:
            image_path = _clean_text(row.get("representative_image_path"))
            resolved = _resolve_path(image_path) if image_path else None
            if resolved and resolved.exists():
                st.image(str(resolved), use_container_width=True)
            else:
                st.info("No representative image available.")

        with right:
            st.markdown(f"### {_clean_text(row.get('restaurant_name'))}")
            st.caption(
                f"{_clean_text(row.get('rest_id'))} • {_clean_text(row.get('borough')) or 'N/A'} • "
                f"{_clean_text(row.get('michelin_category')) or 'N/A'}"
            )

            metrics = st.columns(2)
            metrics[0].metric(score_label, f"{float(row.get('score', 0.0)):.3f}")
            actual_rating = row.get("actual_rating")
            predicted_rating = row.get("predicted_rating")

            if pd.notna(actual_rating):
                metrics[1].metric("Your Rating", f"{float(actual_rating):.1f} ⭐")
            elif pd.notna(predicted_rating):
                metrics[1].metric("Predicted Rating", f"{float(predicted_rating):.1f} ⭐")

            homepage = _clean_text(row.get("homepage"))
            if homepage:
                st.markdown(f"[Homepage]({homepage})")

            counts = st.columns(4)
            counts[0].metric("Menus", int(row.get("menu_count", 0)))
            counts[1].metric("Reviews", int(row.get("review_count", 0)))
            counts[2].metric("Images", int(row.get("image_count", 0)))
            counts[3].metric("Bio", "Yes" if _clean_text(row.get("bio_text")) else "No")

            bio_text = _clean_text(row.get("bio_text"))
            if bio_text:
                title = _truncate(bio_text, 80)
                with st.expander(title):
                    st.write(bio_text)

            menu_items = row.get("menu_items", []) or []
            if menu_items:
                st.markdown("**Menu highlights**")
                for item in menu_items[:3]:
                    st.write(f"- {_truncate(_clean_text(item), 160)}")

            review_snippets = row.get("review_snippets", []) or []
            if review_snippets:
                st.markdown("**Review snippets**")
                for snippet in review_snippets[:2]:
                    title = _truncate(snippet, 75)
                    with st.expander(title):
                        st.write(snippet)

        # ── Sentiment section (full width, below image+info) ──────────────────
        rest_id = _clean_text(row.get("rest_id", ""))
        if rest_id:
            _render_sentiment(rest_id)

        