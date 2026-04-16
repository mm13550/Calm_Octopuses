import os
import streamlit as st
import pandas as pd
from PIL import Image

from algorithms.image_comparison import get_similar_images

# Streamlit App Configuration
st.set_page_config(page_title="Image Similarity Explorer", layout="wide")

st.title("Multimodal App")

tab1, tab2, tab3 = st.tabs(["Image Similarity Explorer", "Dual Encoder Training Logs", "Generalization Analysis"])

# --- Data Loading ---
@st.cache_data
def load_data():
    parquet_path = os.path.join("data", "embeddings", "image_embeddings.parquet")
    if not os.path.exists(parquet_path):
        return None
    
    df = pd.read_parquet(parquet_path)
    
    if not df.empty and 'image_path' in df.columns:
        df['image_path'] = df['image_path'].apply(
            lambda p: os.path.join('data', p) if p.replace('\\', '/').startswith('images/') else p
        )
        
    return df

df = load_data()

with tab1:
    st.header("Image Similarity Explorer")
    st.markdown("Use this tool to select an image from your dataset and instantly see the most visually similar images based on CLIP embeddings. Helpful for debugging your similarity metrics!")

    if df is None or df.empty:
        st.warning("No embeddings found! Run `python generate_embeddings.py` first to create the `embeddings/image_embeddings.parquet` file.")
    else:
        # --- UI Layout ---
        st.sidebar.header("Controls (Tab 1)")

        # Use a selectbox for picking the target image path
        image_paths = df['image_path'].tolist()
        selected_path = st.sidebar.selectbox("Select Target Image", image_paths)
        top_k = st.sidebar.slider("Number of Similar Images", min_value=1, max_value=50, value=10)

        if selected_path:
            # Safely load and display the target image
            if os.path.exists(selected_path):
                target_img = Image.open(selected_path)
                st.sidebar.image(target_img, caption="Target Image", use_container_width=True)
            else:
                st.sidebar.error(f"Image not found at path: {selected_path}")
            
            # Extract embedding
            target_row = df[df['image_path'] == selected_path].iloc[0]
            target_embedding = target_row['embedding']
            
            # Retrieve top K similar images
            with st.spinner("Calculating similarities..."):
                similar_df = get_similar_images(target_embedding, df, top_k=top_k)
            
            # We remove the target image itself from the results
            similar_df = similar_df[similar_df['image_path'] != selected_path]
            
            from ui_components.image_grid import render_image_grid
            render_image_grid(similar_df, top_k)

with tab2:
    st.header("Dual Encoder Training Logs")
    st.markdown("Visualizing the execution trace of the Cross-Modal Autoencoder PyTorch Lightning Loop.")
    
    import glob
    import subprocess
    import time
    import sys
    
    logs_dir = os.path.join("data", "yelp_sandbox", "models", "cross_modal_logs")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        start_btn = st.button("🚀 Start Training")
    
    chart_placeholder = st.empty()
    status_text = st.empty()
    
    def get_latest_metrics_df():
        if not os.path.exists(logs_dir): return None
        versions = glob.glob(os.path.join(logs_dir, "version_*"))
        if not versions: return None
        versions.sort(key=os.path.getmtime, reverse=True)
        latest_v = versions[0]
        metrics_path = os.path.join(latest_v, "metrics.csv")
        if os.path.exists(metrics_path):
            try:
                df = pd.read_csv(metrics_path)
                return df
            except:
                return None
        return None

    def plot_metrics(df):
        if df is not None and not df.empty:
            cols = df.columns.tolist()
            for c in ['step', 'epoch']:
                if c in cols: cols.remove(c)
            df = df.ffill()
            loss_cols = [c for c in cols if 'loss' in c]
            if loss_cols:
                chart_placeholder.line_chart(df, x='epoch', y=loss_cols)

    # If the user clicks start, run a loop
    if start_btn:
        status_text.info("Initiating Training Dual Encoder...")
        os.makedirs(logs_dir, exist_ok=True)
        # Start subprocess
        proc = subprocess.Popen([sys.executable, os.path.join("pipelines", "yelp", "cross_modal_embeddings.py")])
        
        # Wait a moment for PyTorch Lightning to create the log file
        time.sleep(3)
        status_text.warning("Training actively running. Polling metrics...")
        
        while proc.poll() is None:
            df = get_latest_metrics_df()
            plot_metrics(df)
            time.sleep(2)
            
        if proc.returncode == 0:
            status_text.success("Training Completed Successfully!")
            df = get_latest_metrics_df()
            plot_metrics(df)
        else:
            status_text.error("Training failed or was interrupted. Check console logs.")
    else:
        # Just plot the latest available logs if not actively training
        df = get_latest_metrics_df()
        if df is not None:
            plot_metrics(df)
        else:
            st.info("No training logs found yet. Click 'Start Training' to begin.")

# ============================================================
# Tab 3 — Cross-Modal Generalization Analysis
# ============================================================
with tab3:
    import altair as alt
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipelines', 'yelp'))
    from evaluate_generalization import compute_generalization_metrics, compute_tsne

    st.header("🔬 Cross-Modal Generalization Analysis")
    st.markdown(
        """
        The dual-tower autoencoder was trained on **general Yelp casual-dining** data.
        This tab evaluates how well its learned 256-D alignment space transfers to the
        **held-out Philadelphia high-end cohort** (price tier 3–4, stars ≥ 4.0) —
        our proxy for out-of-distribution fine-dining inference.
        """
    )

    # --- Cached wrappers (heavy: loads ~2.6 GB PT file on first call) ---
    @st.cache_data(show_spinner=False)
    def _cached_gen_metrics():
        return compute_generalization_metrics(max_train_samples=1000, max_val_samples=1000)

    @st.cache_data(show_spinner=False)
    def _cached_tsne(train_key, val_key):
        """train_key / val_key are shape tuples used as cache discriminators."""
        m = _cached_gen_metrics()
        return compute_tsne(m['train']['img_latents'], m['val']['img_latents'])

    # --- Run button ---
    btn_col, info_col = st.columns([1, 5])
    with btn_col:
        run_gen = st.button("🔬 Run Analysis", key="run_gen_btn")
    with info_col:
        st.info(
            "First run loads the model checkpoint and embedding tensors (~30 s). "
            "Results are cached for the rest of the session."
        )

    if run_gen:
        st.session_state['gen_run'] = True
    
    if not st.session_state.get('gen_run'):
        st.markdown("---")
        # Show epoch curves from existing logs even without running inference
        st.subheader("Training Epoch Curves")
        st.caption("Loaded from the most recent CSVLogger run — no inference required.")
        hist_df = get_latest_metrics_df()
        if hist_df is not None and not hist_df.empty:
            hist_df = hist_df.ffill()
            alignment_cols = [c for c in hist_df.columns if 'alignment_loss' in c]
            loss_cols      = [c for c in hist_df.columns if c in ('train_loss', 'val_loss')]
            if alignment_cols and 'epoch' in hist_df.columns:
                st.line_chart(hist_df, x='epoch', y=alignment_cols)
            if loss_cols and 'epoch' in hist_df.columns:
                st.line_chart(hist_df, x='epoch', y=loss_cols)
        else:
            st.info("No training logs found. Run training in the 'Dual Encoder Training Logs' tab first.")
        st.stop()

    # ---- Main dashboard (after analysis has been triggered) ----
    try:
        with st.spinner("Loading checkpoint and running encoder inference on both cohorts..."):
            metrics = _cached_gen_metrics()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Guard against stale cache from before discriminability metrics were added
    if 'train_disc' not in metrics:
        _cached_gen_metrics.clear()
        with st.spinner("Cache refreshed (new metrics added). Re-running inference..."):
            metrics = _cached_gen_metrics()

    tm = metrics['train']
    vm = metrics['val']

    st.success(f"✅ Checkpoint loaded: `{metrics['checkpoint']}`  "
               f"({tm['n_samples']} train samples · {vm['n_samples']} val samples)")
    st.markdown("---")

    # ── KPI Row ──────────────────────────────────────────────────────────
    st.subheader("Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Train Alignment MSE",
        f"{tm['alignment_mse']:.4f}",
        help="MSE between image and text latents for Yelp general "
             "(casual-dining) pairs. Lower = better cross-modal alignment.",
    )
    k2.metric(
        "Val Alignment MSE",
        f"{vm['alignment_mse']:.4f}",
        delta=f"{metrics['alignment_gap']:+.4f} vs train",
        delta_color="inverse",
        help="Same metric on the Philadelphia high-end cohort. "
             "A positive delta means the OOD alignment is worse.",
    )
    k3.metric(
        "Mean Cosine Sim (Train)",
        f"{tm['cos_sims'].mean():.4f}",
        help="Average cosine similarity between img_latent and txt_latent "
             "for matched pairs in the training distribution.",
    )
    k4.metric(
        "Mean Cosine Sim (Val)",
        f"{vm['cos_sims'].mean():.4f}",
        delta=f"{metrics['cos_sim_gap']:+.4f} vs train",
        delta_color="normal",
        help="Cosine similarity on the OOD val cohort. "
             "A negative delta means embedding alignment degrades on fine-dining data.",
    )

    # ── Discriminability KPI Row ─────────────────────────────────────────────
    td = metrics['train_disc']
    vd = metrics['val_disc']
    st.markdown("**Restaurant Discriminability** — do the embeddings separate *different* restaurants?")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "Val Restaurants",
        f"{vd['n_restaurants']}",
        help="Unique restaurants in the val cohort sample (after grouping by business_id).",
    )
    d2.metric(
        "Inter-Restaurant Distance (Val)",
        f"{vd['inter_restaurant_dist']:.4f}",
        delta=f"{vd['inter_restaurant_dist'] - td['inter_restaurant_dist']:+.4f} vs train",
        delta_color="normal",
        help="Mean pairwise cosine distance between restaurant centroids. "
             "Higher = restaurants are more separable in the 256-D space. Range: 0–2.",
    )
    d3.metric(
        "Intra-Restaurant Sim (Val)",
        f"{vd['intra_restaurant_sim']:.4f}" if vd['intra_restaurant_sim'] is not None else "N/A",
        help="Mean cosine similarity between embeddings from the *same* restaurant. "
             "Higher = consistent, stable per-restaurant representation.",
    )
    d4.metric(
        "Discriminability Score (Val)",
        f"{vd['discriminability_score']:.3f}",
        delta=f"{vd['discriminability_score'] - td['discriminability_score']:+.3f} vs train",
        delta_color="normal",
        help="inter_dist / (1 − intra_sim). Higher is better. "
             " > 1.0 is good; < 0.3 suggests latent space collapse.",
    )
    st.markdown("---")

    # ── Charts Row 1: Epoch curves + Reconstruction MSE ──────────────────
    curve_col, bar_col = st.columns(2)

    with curve_col:
        st.subheader("Alignment Loss Over Training")
        st.caption("Train vs Val cohort alignment loss — indicates whether the "
                   "model was already seeing the OOD distribution during training.")
        hist_df = get_latest_metrics_df()
        if hist_df is not None and not hist_df.empty:
            hist_df = hist_df.ffill()
            alignment_cols = [c for c in hist_df.columns if 'alignment_loss' in c]
            if alignment_cols and 'epoch' in hist_df.columns:
                st.line_chart(hist_df.set_index('epoch')[alignment_cols])
            else:
                st.info("No alignment_loss columns found in metrics CSV.")
        else:
            st.info("Training logs not yet available.")

    with bar_col:
        st.subheader("Reconstruction MSE by Cohort & Tower")
        st.caption("Higher val MSE means the decoder struggles to reconstruct "
                   "OOD fine-dining embeddings — a direct signal of domain shift.")
        recon_data = pd.DataFrame([
            {"Tower": "Image", "Cohort": "Train (Yelp)",          "Recon MSE": tm['img_recon_mse']},
            {"Tower": "Image", "Cohort": "Val (Philadelphia HE)", "Recon MSE": vm['img_recon_mse']},
            {"Tower": "Text",  "Cohort": "Train (Yelp)",          "Recon MSE": tm['txt_recon_mse']},
            {"Tower": "Text",  "Cohort": "Val (Philadelphia HE)", "Recon MSE": vm['txt_recon_mse']},
        ])
        bar_chart = (
            alt.Chart(recon_data)
            .mark_bar()
            .encode(
                x=alt.X("Tower:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Recon MSE:Q", title="Mean Reconstruction MSE"),
                color=alt.Color(
                    "Cohort:N",
                    scale=alt.Scale(range=["#4C78A8", "#F58518"]),
                    legend=alt.Legend(orient="bottom"),
                ),
                xOffset="Cohort:N",
                tooltip=["Tower", "Cohort", alt.Tooltip("Recon MSE:Q", format=".4f")],
            )
            .properties(height=300)
        )
        st.altair_chart(bar_chart, use_container_width=True)

    st.markdown("---")

    # ── Charts Row 2: Cosine Sim distribution + t-SNE ────────────────────
    hist_col, tsne_col = st.columns(2)

    with hist_col:
        st.subheader("Cross-Modal Cosine Similarity Distribution")
        st.caption("Distribution of per-pair cosine similarity between image and text "
                   "latents. A leftward shift in Val indicates weaker alignment on "
                   "fine-dining data.")
        cos_data = pd.DataFrame(
            {"Cosine Similarity": list(tm['cos_sims']) + list(vm['cos_sims']),
             "Cohort": (["Train (Yelp)"] * len(tm['cos_sims']))
                       + (["Val (Philadelphia HE)"] * len(vm['cos_sims']))}
        )
        hist_chart = (
            alt.Chart(cos_data)
            .mark_bar(opacity=0.65, binSpacing=0)
            .encode(
                x=alt.X("Cosine Similarity:Q", bin=alt.Bin(maxbins=40),
                         title="Cosine Similarity"),
                y=alt.Y("count()", title="Count", stack=None),
                color=alt.Color(
                    "Cohort:N",
                    scale=alt.Scale(range=["#4C78A8", "#F58518"]),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=["Cohort", "count()"],
            )
            .properties(height=300)
        )
        st.altair_chart(hist_chart, use_container_width=True)

    with tsne_col:
        st.subheader("t-SNE: Image Latent Space")
        st.caption("2-D projection of 256-D image latents (400 points per cohort). "
                   "Tight overlap = good generalization; clear separation = domain shift.")
        with st.spinner("Running t-SNE projection (one-time, ~20s)..."):
            train_shape = tm['img_latents'].shape
            val_shape   = vm['img_latents'].shape
            coords, labels, n_train = _cached_tsne(train_shape, val_shape)

        tsne_df = pd.DataFrame({
            "x":      coords[:, 0],
            "y":      coords[:, 1],
            "Cohort": labels,
        })
        scatter = (
            alt.Chart(tsne_df)
            .mark_circle(size=40, opacity=0.7)
            .encode(
                x=alt.X("x:Q", axis=None, title=""),
                y=alt.Y("y:Q", axis=None, title=""),
                color=alt.Color(
                    "Cohort:N",
                    scale=alt.Scale(range=["#4C78A8", "#F58518"]),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=["Cohort"],
            )
            .properties(height=300)
        )
        st.altair_chart(scatter, use_container_width=True)

    # ── Charts Row 3: Pairwise inter-restaurant distance distribution ─────────
    st.markdown("---")
    st.subheader("Inter-Restaurant Pairwise Distance Distribution")
    st.caption(
        "Each point is one (restaurant A, restaurant B) centroid pair. "
        "A distribution shifted right = restaurants are more separable. "
        "A spike near 0 = many restaurants collapsed to the same embedding."
    )
    dist_chart_data = pd.DataFrame(
        {
            "Cosine Distance": (
                list(td['pairwise_distances']) + list(vd['pairwise_distances'])
            ),
            "Cohort": (
                ["Train (Yelp)"] * len(td['pairwise_distances'])
                + ["Val (Philadelphia HE)"] * len(vd['pairwise_distances'])
            ),
        }
    )
    dist_hist = (
        alt.Chart(dist_chart_data)
        .mark_bar(opacity=0.65, binSpacing=0)
        .encode(
            x=alt.X(
                "Cosine Distance:Q",
                bin=alt.Bin(maxbins=50),
                title="Pairwise Cosine Distance between Restaurant Centroids",
            ),
            y=alt.Y("count()", title="Number of Restaurant Pairs", stack=None),
            color=alt.Color(
                "Cohort:N",
                scale=alt.Scale(range=["#4C78A8", "#F58518"]),
                legend=alt.Legend(orient="bottom-right"),
            ),
            tooltip=["Cohort", "count()", alt.Tooltip("Cosine Distance:Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(dist_hist, use_container_width=True)

    # ── Interpretation panel ──────────────────────────────────────────────
    st.markdown("---")
    gap     = metrics['alignment_gap']
    cos_gap = metrics['cos_sim_gap']
    disc_v  = metrics['val_disc']['discriminability_score']

    if gap < 0.01:
        align_severity = "🟢 **Minimal** — alignment generalizes well to fine-dining."
    elif gap < 0.05:
        align_severity = "🟡 **Moderate** — some generalization loss, within acceptable range."
    else:
        align_severity = "🔴 **Significant** — notable alignment domain shift."

    if disc_v > 1.0:
        disc_severity = "🟢 **Good** — restaurants are well-separated in the latent space."
        disc_action   = "Proceed to per-restaurant embedding aggregation."
    elif disc_v > 0.3:
        disc_severity = "🟡 **Moderate** — some separation, but embeddings may overlap."
        disc_action   = "Aggregation may work; monitor regression head MAE carefully."
    else:
        disc_severity = "🔴 **Poor** — latent space collapse detected. Restaurants look identical."
        disc_action   = "Fine-tuning or architecture changes needed before aggregation."

    if gap < 0.01 and disc_v > 1.0:
        overall_action = "**🟢 Ready for Step 3**: build per-restaurant aggregation pipeline."
    elif gap < 0.05 and disc_v > 0.3:
        overall_action = "**🟡 Proceed with caution**: build aggregation, but validate regression head MAE before committing."
    else:
        overall_action = "**🔴 Consider fine-tuning** (see `DOMAIN_ADAPTATION_PLAN.md`) before building aggregation."

    st.subheader("Interpretation")
    st.markdown(f"""
    | Metric | Train | Val | Assessment |
    |---|---|---|---|
    | **Alignment gap** (val − train MSE) | — | `{gap:+.4f}` | {align_severity} |
    | **Cosine sim gap** (val − train) | — | `{cos_gap:+.4f}` | |
    | **Inter-restaurant distance** | `{metrics['train_disc']['inter_restaurant_dist']:.4f}` | `{metrics['val_disc']['inter_restaurant_dist']:.4f}` | |
    | **Discriminability score** | `{metrics['train_disc']['discriminability_score']:.3f}` | `{metrics['val_disc']['discriminability_score']:.3f}` | {disc_severity} |

    **Overall recommendation:** {overall_action}
    """)

