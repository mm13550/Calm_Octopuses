import streamlit as st
from typing import Dict, Any
from core.data_loader import _clean_text, _resolve_path, _truncate

def _render_result_card(row: Dict[str, Any], score_label: str) -> None:
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
            if "actual_rating" in row:
                metrics[1].metric("Your Rating", f"{float(row.get('actual_rating', 0.0)):.1f} ⭐")
            elif "predicted_rating" in row:
                metrics[1].metric("Predicted Rating", f"{float(row.get('predicted_rating', 0.0)):.1f} ⭐")

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
