"""
ui_components/cards.py
======================
Standardized UI cards for displaying restaurant results in Streamlit.

This module provides the `_render_result_card` function, which handles:
1. Dynamic metadata rendering (stars, price, sentiment).
2. Mode-specific rating display (Actual Rating vs. MDN Predicted Interval).
3. Expandable detail sections for Bio, Menu Highlights, and Review snippets.
4. Interactive rating slider for user feedback.
"""
from html import escape

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
            Path(__file__).resolve().parent.parent / "data" / "csv" / "restaurant_profiles.csv",
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


def _menu_item_key(value: Any) -> str:
    cleaned_value = _clean_text(value)
    if not cleaned_value:
        return ""
    return cleaned_value.split(" — ", 1)[0].lower()


def _badge_html(text: str, variant: str = "soft") -> str:
    cleaned_text = _clean_text(text)
    if not cleaned_text:
        return ""
    return f'<span class="co-inline-badge co-inline-badge--{variant}">{escape(cleaned_text)}</span>'


ASP_COLS  = ["asp_food_quality", "asp_service", "asp_ambiance", "asp_value", "asp_wait_time"]
ASP_LABELS = ["Food", "Service", "Ambiance", "Value", "Wait Time"]


def _render_sentiment(rest_id: str, element_key: str | None = None) -> None:
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
        fillcolor="rgba(239, 106, 71, 0.14)",
        line=dict(color="#ef6a47", width=2.5),
        name="Sentiment",
    ))
    fig.update_layout(
        font=dict(color="#5f6d78"),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickfont=dict(size=10, color="#8a959e"),
                gridcolor="rgba(31, 35, 40, 0.16)",
                linecolor="rgba(31, 35, 40, 0.14)",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#8a959e"),
                gridcolor="rgba(31, 35, 40, 0.16)",
                linecolor="rgba(31, 35, 40, 0.14)",
            ),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    chart_key = f"radar_{element_key or rest_id}"
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=chart_key)


# ── Main card renderer ────────────────────────────────────────────────────────
def _render_result_card(row: Dict[str, Any], score_label: str, render_key: str | None = None) -> None:
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
    render_key : str | None
        Optional per-render unique key suffix used to avoid duplicate Streamlit
        element IDs when the same restaurant card appears in multiple tabs.
    """
    with st.container(border=True):
        left, right = st.columns([1, 2])

        with left:
            image_path = _clean_text(row.get("representative_image_path"))
            resolved = _resolve_path(image_path) if image_path else None
            if resolved and resolved.exists():
                st.image(str(resolved), width="stretch")
            else:
                st.info("No representative image available.")

        with right:
            restaurant_name = _clean_text(row.get("restaurant_name")) or "Unnamed restaurant"
            borough = _clean_text(row.get("borough"))
            michelin_category = _clean_text(row.get("michelin_category"))
            michelin_stars = row.get("michelin_stars", 0)
            michelin_award = _clean_text(row.get("michelin_award"))
            michelin_distinction = _clean_text(row.get("michelin_distinction"))
            rest_id_label = _clean_text(row.get("rest_id"))
            badge_parts = []
            if borough:
                badge_parts.append(_badge_html(borough, "soft"))
            # Show Michelin star/distinction badge from michelin_awards.csv if available
            try:
                stars_int = int(float(michelin_stars)) if michelin_stars else 0
            except (TypeError, ValueError):
                stars_int = 0
            if stars_int > 0:
                star_label = "⭐" * stars_int + f" {michelin_award or michelin_distinction or 'Michelin Star'}"
                badge_parts.append(_badge_html(star_label, "accent"))
            elif michelin_award:
                badge_parts.append(_badge_html(michelin_award, "accent"))
            elif michelin_distinction:
                badge_parts.append(_badge_html(michelin_distinction, "accent"))
            elif michelin_category:
                badge_parts.append(_badge_html(michelin_category, "accent"))

            st.markdown(
                f"""
                <div class="co-result-meta">
                    <p class="co-card-kicker">Restaurant Card</p>
                    <h3 class="co-card-title">{escape(restaurant_name)}</h3>
                    <div class="co-badge-row">{''.join(badge_parts)}</div>
                    <p class="co-card-id">Catalog ID · {escape(rest_id_label)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Rating Section (Now prominent under the title) ───────────────
            rest_id = _clean_text(row.get("rest_id", ""))
            if rest_id and "user_ratings" in st.session_state:
                r_col1, r_col2 = st.columns([2, 1])
                with r_col1:
                    current_rating = st.session_state.user_ratings.get(rest_id)
                    f_key = f"rate_half_{render_key or 'default'}_{rest_id}"
                    
                    # Select slider for half-star steps
                    new_rating = st.select_slider(
                        "Your Rating",
                        options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                        value=float(current_rating or 4.0),
                        key=f_key,
                        help="Select a rating (1-5). Supports half-stars!"
                    )
                    
                    # Visual Star Indicator: ⭐ for full stars, ✨ for half
                    full_stars = int(new_rating)
                    half_star = "✨" if (new_rating % 1 != 0) else ""
                    st.markdown(f"**{new_rating}** {'⭐' * full_stars}{half_star}")

                    # Logic to save changes
                    if current_rating is None:
                        touch_key = f"touched_{f_key}"
                        if st.session_state.get(touch_key) or new_rating != 4.0:
                            st.session_state.user_ratings[rest_id] = new_rating
                            st.session_state[touch_key] = True
                            st.rerun()
                    elif current_rating != new_rating:
                        st.session_state.user_ratings[rest_id] = new_rating
                        st.rerun()
                with r_col2:
                    if current_rating:
                        # Align button slightly lower to match star widget height
                        st.write("") 
                        if st.button("Clear Rating", key=f"clear_btn_{render_key or 'default'}_{rest_id}", use_container_width=True):
                            del st.session_state.user_ratings[rest_id]
                            st.rerun()
            st.divider()

            actual_rating = row.get("actual_rating")
            predicted_rating = row.get("predicted_rating")
            score = row.get("score")

            # Only show the similarity score if it's meaningful (not the 0.0 placeholder from browsing)
            show_score = pd.notna(score) and score > 0 and score_label != "Catalog score"
            show_rating = pd.notna(actual_rating) or pd.notna(predicted_rating)

            if show_score and show_rating:
                m_cols = st.columns(2)
                m_cols[0].metric(score_label, f"{float(score):.3f}")
                if pd.notna(actual_rating):
                    m_cols[1].metric("Your Rating", f"{float(actual_rating):.1f} ⭐")
                else:
                    m_cols[1].metric("Predicted Rating", f"{float(predicted_rating):.1f} ⭐")
            elif show_score:
                st.metric(score_label, f"{float(score):.3f}")
            elif show_rating:
                if pd.notna(actual_rating):
                    st.metric("Your Rating", f"{float(actual_rating):.1f} ⭐")
                else:
                    st.metric("Predicted Rating", f"{float(predicted_rating):.1f} ⭐")

            homepage = _clean_text(row.get("homepage"))
            if homepage:
                safe_homepage = escape(homepage, quote=True)
                st.markdown(
                    f'<a class="co-link-pill" href="{safe_homepage}" target="_blank">Visit homepage</a>',
                    unsafe_allow_html=True,
                )

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

            matched_menu_items = row.get("matched_menu_items", []) or []
            matched_menu_items = [
                _clean_text(item) for item in matched_menu_items if _clean_text(item)
            ]

            menu_items = row.get("menu_items", []) or []
            remaining_menu_items = []
            matched_menu_item_keys = {_menu_item_key(item) for item in matched_menu_items if _menu_item_key(item)}
            for item in menu_items:
                cleaned_item = _clean_text(item)
                cleaned_key = _menu_item_key(cleaned_item)
                if cleaned_item and cleaned_key not in matched_menu_item_keys:
                    remaining_menu_items.append(cleaned_item)

            if matched_menu_items or remaining_menu_items:
                st.markdown("**Menu highlights**")

                seen_highlights = set()
                for item in matched_menu_items[:3]:
                    item_key = _menu_item_key(item)
                    if not item_key or item_key in seen_highlights:
                        continue
                    st.markdown(
                        f"<div class='co-menu-match'><strong>Matched dish</strong> · {escape(_truncate(item, 160))}</div>",
                        unsafe_allow_html=True,
                    )
                    seen_highlights.add(item_key)

                for item in remaining_menu_items[:3]:
                    item_key = _menu_item_key(item)
                    if not item_key or item_key in seen_highlights:
                        continue
                    st.write(f"- {_truncate(item, 160)}")
                    seen_highlights.add(item_key)

            review_snippets = row.get("review_snippets", []) or []
            if review_snippets:
                st.markdown("**Review snippets**")
                for snippet in review_snippets[:2]:
                    title = _truncate(snippet, 75)
                    with st.expander(title):
                        st.write(snippet)

        # ── Sentiment section (full width, below image+info) ──────────
        rest_id = _clean_text(row.get("rest_id", ""))
        if rest_id:
            sentiment_key = f"{render_key}_{rest_id}" if render_key else rest_id
            _render_sentiment(rest_id, element_key=sentiment_key)

        
