"""
tests.py

Diagnostic Streamlit frontend for the Calm Octopuses ML pipeline.
Provides interfaces for the Image Similarity Explorer, Dual Encoder Training Logs,
Cross-Modal Generalization Analysis, Risk Regression Analysis,
and MDN Opinionatedness testing on NYC Michelin restaurants.
"""
import os
import streamlit as st
import pandas as pd
from PIL import Image

from algorithms.image_comparison import get_similar_images

# Streamlit App Configuration
st.set_page_config(page_title="Calm Octopuses — Diagnostics", layout="wide")

st.title("🧪 Diagnostics")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Image Similarity Explorer", "Dual Encoder Training Logs", "Generalization Analysis", "Risk Regression Analysis", "MDN Opinionatedness Test"])

# --- Data Loading ---
@st.cache_data
def load_data():
    """
    Loads pre-computed image embeddings from a Parquet file.
    
    Returns:
        pd.DataFrame or None: DataFrame containing the image embeddings and paths,
                              or None if the file does not exist.
    """
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
                st.sidebar.image(target_img, caption="Target Image", width="stretch")
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
        """
        Retrieves the most recent metrics.csv file from the PyTorch Lightning
        training logs directory.
        
        Returns:
            pd.DataFrame or None: DataFrame containing training metrics over time.
        """
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
        """
        Plots the training loss curves on the Streamlit dashboard.
        
        Args:
            df (pd.DataFrame): DataFrame containing 'epoch' and loss columns.
        """
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
    def render_tab3_content():
        """
        Renders the Cross-Modal Generalization Analysis tab.
        This includes key metrics, alignment loss curves, reconstruction MSE charts,
        cosine similarity distributions, and t-SNE projections.
        """
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
            """
            Computes and caches generalization metrics across train and val cohorts.
            
            Returns:
                dict: A dictionary of computed metrics including alignment MSE,
                      cosine similarities, and discriminability scores.
            """
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
            return

        # ---- Main dashboard (after analysis has been triggered) ----
        try:
            with st.spinner("Loading checkpoint and running encoder inference on both cohorts..."):
                metrics = _cached_gen_metrics()
        except FileNotFoundError as e:
            st.error(str(e))
            return

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
            st.altair_chart(bar_chart, width="stretch")

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
            st.altair_chart(hist_chart, width="stretch")

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
            st.altair_chart(scatter, width="stretch")

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
        st.altair_chart(dist_hist, width="stretch")

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

    render_tab3_content()

# ============================================================
# Tab 4 — Risk Regression Analysis
# ============================================================
with tab4:
    st.header("📈 Risk Regression Analysis")
    st.markdown(
        """
        Visualizing the **1025-D concatenated** Multi-Layer Perceptron (MDNScorer) trained 
        with Mixture Density Loss. It models the **95% Confidence Interval** 
        for an expected user rating explicitly, using native 512-D CLIP features.
        """
    )
    
    st.info("Training computes locally on baseline Yelp users natively extracted from SQLite, and validates zero-shot against held-out Michelin test users.")
    
    # Persist results in session state so they don't vanish when clicking the toggle
    if 'reg_train_res' not in st.session_state:
        st.session_state['reg_train_res'] = None
    if 'reg_test_res' not in st.session_state:
        st.session_state['reg_test_res'] = None

    if st.button("🚀 Train & Evaluate Risk Network"):
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'algorithms'))
        try:
            from algorithms.mdn_regression import evaluate_regression
            
            _ROOT = os.path.dirname(__file__)
            TRAIN_JSON = os.path.join(_ROOT, 'data', 'yelp_sandbox', 'regression_train_set.json')
            TEST_JSON  = os.path.join(_ROOT, 'data', 'yelp_sandbox', 'regression_val_set.json')
            TRAIN_EMB  = os.path.join(_ROOT, 'data', 'yelp_sandbox', 'toy_embeddings', 'toy_restaurant_embeddings_train.pt')
            VAL_EMB    = os.path.join(_ROOT, 'data', 'yelp_sandbox', 'toy_embeddings', 'toy_restaurant_embeddings_val.pt')
            
            with st.spinner("Extracting datasets and dynamically training interval network (~30s)..."):
                train_res, test_res = evaluate_regression(TRAIN_JSON, TEST_JSON, TRAIN_EMB, VAL_EMB, max_epochs=10)
            
            if train_res is not None and not train_res.empty:
                st.session_state['reg_train_res'] = train_res
                st.session_state['reg_test_res'] = test_res
                st.success("Training and inference achieved successfully.")
            else:
                st.error("Training dataset not generated! Check background export script.")

        except Exception as e:
            st.error(f"Error computing regression: {e}")

    # Render dashboard if results exist in session state
    if st.session_state['reg_test_res'] is not None:
        train_res = st.session_state['reg_train_res']
        test_res  = st.session_state['reg_test_res']
        
        t_mae = (train_res['Actual_Rating'] - train_res['Predicted_Median']).abs().mean()
        t_cov = train_res['In_Bounds'].mean() * 100
        v_mae = (test_res['Actual_Rating'] - test_res['Predicted_Median']).abs().mean()
        v_cov = test_res['In_Bounds'].mean() * 100
        
        st.subheader("Generalization KPIs")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Generic Train MAE", f"{t_mae:.3f}★")
        c2.metric("Michelin Test MAE", f"{v_mae:.3f}★", delta=f"{v_mae - t_mae:+.3f} vs train", delta_color="inverse")
        c3.metric("Michelin Coverage", f"{v_cov:.1f}%", help="Percentage of test ratings that fell within the 95% HDR.")
        
        # Integrity Metrics
        avg_w = test_res['HDR_95_Width'].mean()
        avg_c = test_res['HDR_95_Segments_Count'].mean()
        
        # Using a separate section for integrity to keep it clean
        st.write("---")
        st.subheader("Model Integrity & Diagnostic Metrics")
        i1, i2, i3 = st.columns(3)
        i1.metric("Avg. 95% HDR Width", f"{avg_w:.2f}★", help="Shows how 'sharp' the model is. Smaller is better.")
        i2.metric("Avg. HDR Segments", f"{avg_c:.2f}", help="Closer to 1.0 means the model is providing continuous, non-gapped regions.")
        i3.metric("HDR Quality Score", f"{v_cov / (avg_w + 1e-6):.2f}", help="Information Gain proxy: Coverage divided by Width.")
        
        import altair as alt
        # Plot test results scatter
        st.subheader("Michelin Holdout Predictions vs Network Confidence bounds")
        
        # We limit to 200 points to not overcrowd the chart
        vis_df = test_res.head(200).copy()
        vis_df['User_ID_Proxy'] = range(len(vis_df))
        
        # Add jitter for the "Actual Rating" view
        import numpy as np
        vis_df['jitter'] = np.random.uniform(-0.3, 0.3, size=len(vis_df))
        
        # UI Toggle for X-axis view
        view_mode = st.radio(
            "Select Visualization Perspective:",
            ["Per-Sample Index", "Grouped by Actual Rating"],
            horizontal=True,
            help="Switch between seeing individual intervals and seeing how the model clusters around specific rating buckets."
        )
        
        if view_mode == "Per-Sample Index":
            x_enc = alt.X('User_ID_Proxy:O', title="Test User Profile #", axis=alt.Axis(labels=False, ticks=False))
            x_offset = alt.value(0)
        else:
            x_enc = alt.X('Actual_Rating:O', title="Ground Truth Rating [1-5]")
            x_offset = alt.XOffset('jitter:Q')

        # Create segment-specific DataFrames for Altair to draw multiple bars per sample
        rows_50 = []
        rows_95 = []
        for _, row in vis_df.iterrows():
            for slo, shi in row['HDR_50_Segments']:
                new_row = row.to_dict()
                new_row['Seg_Low'] = slo
                new_row['Seg_High'] = shi
                rows_50.append(new_row)
            for slo, shi in row['HDR_95_Segments']:
                new_row = row.to_dict()
                new_row['Seg_Low'] = slo
                new_row['Seg_High'] = shi
                rows_95.append(new_row)
        
        df_50 = pd.DataFrame(rows_50)
        df_95 = pd.DataFrame(rows_95)

        # Create a point chart for the Median predictions vs Truth
        points = alt.Chart(vis_df).mark_circle(size=60).encode(
            x=x_enc,
            xOffset=x_offset,
            y=alt.Y('Predicted_Median:Q', title="Rating [1-5]"),
            color=alt.Color('In_Bounds:N', scale=alt.Scale(domain=[1, 0], range=["#2ca02c", "#d62728"])),
            tooltip=['Actual_Rating', 'Predicted_Median', 'HDR_95_Width', 'HDR_95_Segments_Count', 'HDR_50_Segments', 'HDR_95_Segments']
        )
        
        # 95% HDR: Outer Halo (Multiple bars possible)
        halo_95 = alt.Chart(df_95).mark_errorbar(opacity=0.3, thickness=1).encode(
            x=x_enc,
            xOffset=x_offset,
            y=alt.Y('Seg_Low:Q', title=''),
            y2=alt.Y2('Seg_High:Q'),
            color=alt.value('rgba(150, 150, 150, 0.4)')
        )
        
        # 50% HDR: Inner Core (Multiple bars possible)
        core_50 = alt.Chart(df_50).mark_errorbar(opacity=0.6, thickness=4).encode(
            x=x_enc,
            xOffset=x_offset,
            y=alt.Y('Seg_Low:Q', title=''),
            y2=alt.Y2('Seg_High:Q'),
            color=alt.value('rgba(100, 100, 100, 0.6)')
        )
        
        # Actual rating markers
        actuals = alt.Chart(vis_df).mark_tick(thickness=2, size=15, color='white').encode(
            x=x_enc,
            xOffset=x_offset,
            y=alt.Y('Actual_Rating:Q')
        )
        
        st.altair_chart((halo_95 + core_50 + points + actuals).properties(height=400), width="stretch")


# ============================================================
# Tab 5 — MDN Opinionatedness Test
# ============================================================
with tab5:
    import sys as _sys
    import numpy as np
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parent
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))

    from core.data_loader import build_restaurant_catalog, load_restaurant_embeddings, _clean_text
    from algorithms.mdn_regression import load_mdn_model
    import torch

    st.header("🎯 MDN Opinionatedness Test")
    st.markdown(
        """
        Rate a handful of **NYC Michelin** restaurants below, then click **Run Predictions**
        to see how sharply the MDN distributes its probability mass for the restaurants you
        haven't rated yet.  
        A well-calibrated, opinionated model should produce **narrow, peaked PDFs**.
        Flat / uniform distributions suggest the model is hedging — a sign of domain shift
        or insufficient user signal.
        """
    )

    # --- Session state for this tab ---
    if "mdn_test_ratings" not in st.session_state:
        st.session_state["mdn_test_ratings"] = {}

    @st.cache_data(show_spinner=False)
    def _load_catalog_cached():
        return build_restaurant_catalog()

    catalog_df = _load_catalog_cached()

    if catalog_df.empty:
        st.warning("No restaurant catalog available. Check data files.")
    else:
        restaurant_names = catalog_df["restaurant_name"].tolist()

        # ── Rating panel ───────────────────────────────────────────────────
        st.subheader("Step 1 — Rate some restaurants")
        st.caption("Select a restaurant and give it a star rating. Rate at least 1 to enable predictions.")

        col_sel, col_stars = st.columns([3, 2])
        with col_sel:
            pick_name = st.selectbox(
                "Restaurant",
                restaurant_names,
                key="mdn_test_pick",
                label_visibility="collapsed",
            )
        with col_stars:
            star_val = st.feedback("stars", key=f"mdn_test_stars_{pick_name}")
            if star_val is not None:
                pick_id = catalog_df[catalog_df["restaurant_name"] == pick_name].iloc[0]["rest_id"]
                st.session_state["mdn_test_ratings"][pick_id] = float(star_val + 1)

        # Show current ratings table
        if st.session_state["mdn_test_ratings"]:
            rated_rows = []
            for rid, rating in st.session_state["mdn_test_ratings"].items():
                name_match = catalog_df[catalog_df["rest_id"] == rid]["restaurant_name"]
                name = name_match.iloc[0] if not name_match.empty else rid
                rated_rows.append({"Restaurant": name, "Your Rating": f"{int(rating)} ⭐"})
            st.dataframe(pd.DataFrame(rated_rows), use_container_width=True, hide_index=True)

            if st.button("🗑 Clear all ratings", key="mdn_test_clear"):
                st.session_state["mdn_test_ratings"] = {}
                st.rerun()

        st.divider()

        # ── Prediction panel ───────────────────────────────────────────────
        st.subheader("Step 2 — Run MDN predictions")

        n_display = st.slider(
            "Restaurants to predict",
            min_value=5, max_value=min(40, len(catalog_df)),
            value=15,
            help="Number of unrated restaurants to show predicted PDFs for.",
        )

        run_btn = st.button(
            "▶ Run Predictions",
            disabled=len(st.session_state["mdn_test_ratings"]) == 0,
            key="mdn_test_run",
        )

        if run_btn or st.session_state.get("mdn_test_ran"):
            st.session_state["mdn_test_ran"] = True
            user_ratings = st.session_state["mdn_test_ratings"]

            model = load_mdn_model()
            if model is None:
                st.error("MDN checkpoint not found at expected path.")
            else:
                embeddings_map = load_restaurant_embeddings()

                # Build user vector
                hist_vecs, hist_weights = [], []
                for rid, rating in user_ratings.items():
                    vec = embeddings_map.get(rid)
                    if vec is not None:
                        w = float(rating) / 5.0
                        hist_vecs.append(torch.from_numpy(vec).float() * w)
                        hist_weights.append(w)

                if not hist_vecs:
                    st.error("None of your rated restaurants have embeddings. Try rating different ones.")
                else:
                    user_vec = torch.stack(hist_vecs).sum(dim=0) / sum(hist_weights)
                    mean_hist = sum(user_ratings.values()) / len(user_ratings)
                    scalar_feat = torch.tensor([mean_hist], dtype=torch.float32)

                    # Score unrated restaurants
                    results = []
                    for _, row in catalog_df.iterrows():
                        rid = _clean_text(row.get("rest_id"))
                        if rid in user_ratings:
                            continue
                        vec_np = embeddings_map.get(rid)
                        if vec_np is None:
                            continue

                        target_vec = torch.from_numpy(vec_np).float()
                        feat = torch.cat([user_vec, target_vec, scalar_feat]).unsqueeze(0)

                        with torch.no_grad():
                            mus, log_sigmas, pi_logits = model(feat)
                            pis = torch.softmax(pi_logits, dim=1)
                            expected_mu = (mus * pis).sum(dim=1).item()

                            grid_y = torch.linspace(1.0, 5.0, 101)
                            g_exp = grid_y.view(1, -1, 1)
                            m_exp = mus.unsqueeze(1)
                            s_exp = torch.exp(log_sigmas).unsqueeze(1)
                            p_exp = pis.unsqueeze(1)
                            pdfs = (1.0 / (2.0 * s_exp)) * torch.exp(-torch.abs(g_exp - m_exp) / s_exp)
                            pdf_arr = (p_exp * pdfs).sum(dim=2)[0].cpu().numpy()

                        # Sharpness: std of the PDF
                        x_grid = np.linspace(1.0, 5.0, 101)
                        mean_pdf = float(np.dot(x_grid, pdf_arr) / (pdf_arr.sum() + 1e-9))
                        var_pdf  = float(np.dot((x_grid - mean_pdf) ** 2, pdf_arr) / (pdf_arr.sum() + 1e-9))
                        sharpness = float(np.sqrt(var_pdf))  # std → lower is sharper

                        results.append({
                            "rest_id": rid,
                            "restaurant_name": _clean_text(row.get("restaurant_name")),
                            "predicted_rating": expected_mu,
                            "sharpness": sharpness,
                            "pdf": pdf_arr,
                        })

                    if not results:
                        st.warning("No predictions available — all restaurants may be rated or missing embeddings.")
                    else:
                        results.sort(key=lambda r: r["predicted_rating"], reverse=True)
                        results = results[:n_display]

                        # ── Aggregate sharpness KPIs ───────────────────
                        avg_sharp = float(np.mean([r["sharpness"] for r in results]))
                        avg_pred  = float(np.mean([r["predicted_rating"] for r in results]))

                        k1, k2, k3 = st.columns(3)
                        k1.metric("Restaurants Predicted", len(results))
                        k2.metric("Avg Predicted Rating", f"{avg_pred:.2f} ⭐")
                        k3.metric(
                            "Avg PDF Std Dev",
                            f"{avg_sharp:.3f}",
                            help="Lower = model is more opinionated. >0.8 suggests uniform hedging.",
                        )

                        st.divider()
                        st.subheader("Predicted Rating Distributions")
                        st.caption(
                            "Each chart shows the full probability density over ratings 1–5. "
                            "Narrow peaks = opinionated; flat lines = uncertain."
                        )

                        # Two-column grid of area charts
                        import altair as alt
                        pairs = list(zip(results[::2], results[1::2]))
                        if len(results) % 2 == 1:
                            # Odd one out
                            leftover = results[-1]
                        else:
                            leftover = None

                        for left_r, right_r in pairs:
                            lcol, rcol = st.columns(2)
                            for col, r in ((lcol, left_r), (rcol, right_r)):
                                x_grid = np.linspace(1.0, 5.0, 101)
                                chart_df = pd.DataFrame({
                                    "Rating": x_grid,
                                    "Density": r["pdf"],
                                })
                                chart = (
                                    alt.Chart(chart_df)
                                    .mark_area(
                                        line={"color": "#FF4B4B"},
                                        color=alt.Gradient(
                                            gradient="linear",
                                            stops=[
                                                alt.GradientStop(color="rgba(255,75,75,0.05)", offset=0),
                                                alt.GradientStop(color="rgba(255,75,75,0.5)",  offset=1),
                                            ],
                                            x1=1, x2=1, y1=1, y2=0,
                                        ),
                                    )
                                    .encode(
                                        x=alt.X("Rating:Q", scale=alt.Scale(domain=[1, 5]), title="Rating"),
                                        y=alt.Y("Density:Q", title="PDF"),
                                        tooltip=[alt.Tooltip("Rating:Q", format=".2f"), alt.Tooltip("Density:Q", format=".4f")],
                                    )
                                    .properties(
                                        title=alt.Title(
                                            f"{r['restaurant_name']}",
                                            subtitle=f"Predicted: {r['predicted_rating']:.2f}⭐  |  Std: {r['sharpness']:.3f}",
                                        ),
                                        height=180,
                                    )
                                )
                                col.altair_chart(chart, use_container_width=True)

                        if leftover:
                            r = leftover
                            x_grid = np.linspace(1.0, 5.0, 101)
                            chart_df = pd.DataFrame({"Rating": x_grid, "Density": r["pdf"]})
                            chart = (
                                alt.Chart(chart_df)
                                .mark_area(
                                    line={"color": "#FF4B4B"},
                                    color=alt.Gradient(
                                        gradient="linear",
                                        stops=[
                                            alt.GradientStop(color="rgba(255,75,75,0.05)", offset=0),
                                            alt.GradientStop(color="rgba(255,75,75,0.5)",  offset=1),
                                        ],
                                        x1=1, x2=1, y1=1, y2=0,
                                    ),
                                )
                                .encode(
                                    x=alt.X("Rating:Q", scale=alt.Scale(domain=[1, 5])),
                                    y=alt.Y("Density:Q", title="PDF"),
                                    tooltip=[alt.Tooltip("Rating:Q", format=".2f"), alt.Tooltip("Density:Q", format=".4f")],
                                )
                                .properties(
                                    title=alt.Title(
                                        f"{r['restaurant_name']}",
                                        subtitle=f"Predicted: {r['predicted_rating']:.2f}⭐  |  Std: {r['sharpness']:.3f}",
                                    ),
                                    height=180,
                                )
                            )
                            st.altair_chart(chart, use_container_width=True)
