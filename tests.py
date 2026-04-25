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
st.set_page_config(page_title="Calm Octopuses â€” Diagnostics", layout="wide")

st.title("ðŸ§ª Diagnostics")

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
        start_btn = st.button("ðŸš€ Start Training")
    
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
# Tab 3 â€” Embedding Analysis (CLIP restaurant profiles)
# ============================================================
with tab3:
    import numpy as np
    import sys as _sys
    from pathlib import Path as _Path

    _ROOT3 = _Path(__file__).resolve().parent
    if str(_ROOT3) not in _sys.path:
        _sys.path.insert(0, str(_ROOT3))

    import altair as alt
    from core.data_loader import (
        load_restaurant_embeddings,
        load_finegrained_embeddings,
        build_restaurant_catalog,
        _clean_text,
        MENU_EMBEDDINGS_JSONL,
        REVIEW_EMBEDDINGS_JSONL,
    )

    st.header("ðŸ”¬ Embedding Space Analysis")
    st.markdown(
        """
        Analyzes the **512-D CLIP restaurant profile embeddings** stored in
        `data/embeddings/restaurant_profiles.jsonl` â€” the same vectors used
        by the frontend for semantic search and MDN scoring.

        Use this tab to check for **embedding quality**, **inter-restaurant
        discriminability**, and **latent space geometry** before trusting
        MDN predictions on new restaurants.
        """
    )

    run_emb = st.button("ðŸ”¬ Run Embedding Analysis", key="run_emb_btn")
    if run_emb:
        st.session_state["emb_run"] = True

    if not st.session_state.get("emb_run"):
        st.info("Click **Run Embedding Analysis** to load and inspect the restaurant embeddings.")
    else:
        with st.spinner("Loading restaurant embeddings and catalogâ€¦"):
            emb_map = load_restaurant_embeddings()
            catalog_df = build_restaurant_catalog()
            menu_map = load_finegrained_embeddings(str(MENU_EMBEDDINGS_JSONL))
            review_map = load_finegrained_embeddings(str(REVIEW_EMBEDDINGS_JSONL))

        if not emb_map:
            st.error("No embeddings found. Run the embedding pipeline first.")
        else:
            # â”€â”€ Build matrix of restaurant vectors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            rest_ids = list(emb_map.keys())
            matrix = np.stack([emb_map[rid] for rid in rest_ids])   # (N, 512)
            N = len(rest_ids)

            # â”€â”€ KPI row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            st.subheader("Key Metrics")
            norms = np.linalg.norm(matrix, axis=1)
            avg_norm = float(norms.mean())
            min_norm = float(norms.min())
            max_norm = float(norms.max())

            # Compute pairwise cosine distances (fast via matrix multiply)
            normed = matrix / (norms[:, None] + 1e-9)   # (N, 512) unit vecs
            cosine_sim_matrix = normed @ normed.T        # (N, N)
            np.fill_diagonal(cosine_sim_matrix, np.nan)
            inter_sim = float(np.nanmean(cosine_sim_matrix))
            inter_dist = 1.0 - inter_sim

            # Intra-restaurant similarity (menu vectors vs profile)
            intra_sims = []
            for rid in rest_ids:
                profile_vec = emb_map.get(rid)
                menu_vecs = menu_map.get(rid, [])
                if profile_vec is not None and menu_vecs:
                    pv = profile_vec / (np.linalg.norm(profile_vec) + 1e-9)
                    for mv in menu_vecs:
                        mv_n = mv / (np.linalg.norm(mv) + 1e-9)
                        intra_sims.append(float(np.dot(pv, mv_n)))
            avg_intra = float(np.mean(intra_sims)) if intra_sims else None

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Restaurants w/ Embeddings", N,
                      help="Number of restaurants that have a pre-computed profile vector.")
            k2.metric("Mean Inter-Restaurant Cosine Dist", f"{inter_dist:.4f}",
                      help="Higher = restaurants are more separable in the 512-D CLIP space.")
            k3.metric("Avg Vector Norm", f"{avg_norm:.3f}",
                      help="Should be ~1.0 if vectors are already L2-normalised.")
            k4.metric(
                "Avg Profileâ†”Menu Cosine Sim",
                f"{avg_intra:.4f}" if avg_intra is not None else "N/A",
                help="How consistently menu item vectors align with the restaurant's profile vector. Higher = tighter semantic coherence.",
            )
            st.markdown("---")

            # â”€â”€ Row 1: Norm histogram + pairwise cosine-sim histogram â”€â”€â”€â”€â”€â”€â”€
            hist_col, cos_col = st.columns(2)

            with hist_col:
                st.subheader("Vector Norm Distribution")
                st.caption("Ideally tightly clustered around 1.0 for L2-normalised CLIP embeddings.")
                norm_df = pd.DataFrame({"L2 Norm": norms})
                norm_chart = (
                    alt.Chart(norm_df)
                    .mark_bar(color="#4C78A8")
                    .encode(
                        x=alt.X("L2 Norm:Q", bin=alt.Bin(maxbins=30), title="L2 Norm"),
                        y=alt.Y("count()", title="Restaurants"),
                        tooltip=["count()"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(norm_chart, use_container_width=True)

            with cos_col:
                st.subheader("Pairwise Cosine Similarity Distribution")
                st.caption(
                    "Distribution of all (restaurant_i, restaurant_j) cosine similarities. "
                    "A tighter, lower distribution = more discriminable embeddings."
                )
                upper_tri = cosine_sim_matrix[np.triu_indices(N, k=1)]
                upper_tri = upper_tri[~np.isnan(upper_tri)]
                cos_df = pd.DataFrame({"Cosine Similarity": upper_tri})
                cos_chart = (
                    alt.Chart(cos_df)
                    .mark_bar(color="#F58518", opacity=0.8)
                    .encode(
                        x=alt.X("Cosine Similarity:Q", bin=alt.Bin(maxbins=40)),
                        y=alt.Y("count()", title="Restaurant Pairs"),
                        tooltip=["count()"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(cos_chart, use_container_width=True)

            st.markdown("---")

            # â”€â”€ Row 2: PCA 2-D scatter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            st.subheader("PCA Projection of Restaurant Embeddings")
            st.caption(
                "2-D PCA projection of the 512-D CLIP restaurant profiles. "
                "Tight clusters by borough or Michelin category indicate good semantic grouping."
            )

            from sklearn.decomposition import PCA

            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(matrix)
            var_explained = pca.explained_variance_ratio_

            pca_df = pd.DataFrame({
                "PC1": coords[:, 0],
                "PC2": coords[:, 1],
                "rest_id": rest_ids,
            })

            # Join catalog metadata for colour coding
            if not catalog_df.empty:
                meta = catalog_df[["rest_id", "restaurant_name", "borough", "michelin_category"]].copy()
                pca_df = pca_df.merge(meta, on="rest_id", how="left")
                pca_df["borough"] = pca_df["borough"].fillna("Unknown")
                color_field = "borough:N"
                tooltip_fields = ["restaurant_name", "borough", "michelin_category"]
            else:
                pca_df["label"] = pca_df["rest_id"]
                color_field = "label:N"
                tooltip_fields = ["rest_id"]

            scatter = (
                alt.Chart(pca_df)
                .mark_circle(size=70, opacity=0.8)
                .encode(
                    x=alt.X("PC1:Q", title=f"PC1 ({var_explained[0]*100:.1f}% var)"),
                    y=alt.Y("PC2:Q", title=f"PC2 ({var_explained[1]*100:.1f}% var)"),
                    color=alt.Color(color_field, legend=alt.Legend(orient="right")),
                    tooltip=tooltip_fields,
                )
                .properties(height=420)
                .interactive()
            )
            st.altair_chart(scatter, use_container_width=True)

            st.markdown("---")

            # â”€â”€ Row 3: Top-K most / least similar pairs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            st.subheader("Most & Least Similar Restaurant Pairs")
            st.caption("Sanity check: semantically similar restaurants should cluster at the top.")

            pairs_data = []
            for i in range(N):
                for j in range(i + 1, N):
                    pairs_data.append((rest_ids[i], rest_ids[j], float(cosine_sim_matrix[i, j])))

            pairs_data.sort(key=lambda x: x[2], reverse=True)

            def _name(rid):
                if catalog_df.empty:
                    return rid
                match = catalog_df[catalog_df["rest_id"] == rid]["restaurant_name"]
                return match.iloc[0] if not match.empty else rid

            top10 = [{"Restaurant A": _name(a), "Restaurant B": _name(b), "Cosine Sim": f"{s:.4f}"}
                     for a, b, s in pairs_data[:10]]
            bot10 = [{"Restaurant A": _name(a), "Restaurant B": _name(b), "Cosine Sim": f"{s:.4f}"}
                     for a, b, s in pairs_data[-10:]]

            sim_l, sim_r = st.columns(2)
            with sim_l:
                st.markdown("**Most similar** (highest cosine sim)")
                st.dataframe(pd.DataFrame(top10), use_container_width=True, hide_index=True)
            with sim_r:
                st.markdown("**Least similar** (lowest cosine sim)")
                st.dataframe(pd.DataFrame(bot10), use_container_width=True, hide_index=True)

# ============================================================
# Tab 4 â€” Risk Regression Analysis
# ============================================================
with tab4:
    st.header("ðŸ“ˆ Risk Regression Analysis")
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

    if st.button("ðŸš€ Train & Evaluate Risk Network"):
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
        c1.metric("Generic Train MAE", f"{t_mae:.3f}â˜…")
        c2.metric("Michelin Test MAE", f"{v_mae:.3f}â˜…", delta=f"{v_mae - t_mae:+.3f} vs train", delta_color="inverse")
        c3.metric("Michelin Coverage", f"{v_cov:.1f}%", help="Percentage of test ratings that fell within the 95% HDR.")
        
        # Integrity Metrics
        avg_w = test_res['HDR_95_Width'].mean()
        avg_c = test_res['HDR_95_Segments_Count'].mean()
        
        # Using a separate section for integrity to keep it clean
        st.write("---")
        st.subheader("Model Integrity & Diagnostic Metrics")
        i1, i2, i3 = st.columns(3)
        i1.metric("Avg. 95% HDR Width", f"{avg_w:.2f}â˜…", help="Shows how 'sharp' the model is. Smaller is better.")
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
# Tab 5 â€” MDN Opinionatedness Test
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

    st.header("ðŸŽ¯ MDN Opinionatedness Test")
    st.markdown(
        """
        Rate a handful of **NYC Michelin** restaurants below, then click **Run Predictions**
        to see how sharply the MDN distributes its probability mass for the restaurants you
        haven't rated yet.  
        A well-calibrated, opinionated model should produce **narrow, peaked PDFs**.
        Flat / uniform distributions suggest the model is hedging â€” a sign of domain shift
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

        # â”€â”€ Rating panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.subheader("Step 1 â€” Rate some restaurants")
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
                rated_rows.append({"Restaurant": name, "Your Rating": f"{int(rating)} â­"})
            st.dataframe(pd.DataFrame(rated_rows), use_container_width=True, hide_index=True)

            if st.button("ðŸ—‘ Clear all ratings", key="mdn_test_clear"):
                st.session_state["mdn_test_ratings"] = {}
                st.rerun()

        st.divider()

        # â”€â”€ Prediction panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.subheader("Step 2 â€” Run MDN predictions")

        n_display = st.slider(
            "Restaurants to predict",
            min_value=5, max_value=min(40, len(catalog_df)),
            value=15,
            help="Number of unrated restaurants to show predicted PDFs for.",
        )

        run_btn = st.button(
            "â–¶ Run Predictions",
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
                        sharpness = float(np.sqrt(var_pdf))  # std â†’ lower is sharper

                        results.append({
                            "rest_id": rid,
                            "restaurant_name": _clean_text(row.get("restaurant_name")),
                            "predicted_rating": expected_mu,
                            "sharpness": sharpness,
                            "pdf": pdf_arr,
                        })

                    if not results:
                        st.warning("No predictions available â€” all restaurants may be rated or missing embeddings.")
                    else:
                        results.sort(key=lambda r: r["predicted_rating"], reverse=True)
                        results = results[:n_display]

                        # â”€â”€ Aggregate sharpness KPIs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                        avg_sharp = float(np.mean([r["sharpness"] for r in results]))
                        avg_pred  = float(np.mean([r["predicted_rating"] for r in results]))

                        k1, k2, k3 = st.columns(3)
                        k1.metric("Restaurants Predicted", len(results))
                        k2.metric("Avg Predicted Rating", f"{avg_pred:.2f} â­")
                        k3.metric(
                            "Avg PDF Std Dev",
                            f"{avg_sharp:.3f}",
                            help="Lower = model is more opinionated. >0.8 suggests uniform hedging.",
                        )

                        st.divider()
                        import altair as alt

                        # HDR helper
                        def _compute_hdr_segments(pdf, x_grid, mass_threshold):
                            total = pdf.sum()
                            if total == 0:
                                return []
                            norm_pdf = pdf / total
                            sorted_idx = np.argsort(norm_pdf)[::-1]
                            cumulative = 0.0
                            in_hdr = np.zeros(len(pdf), dtype=bool)
                            for i in sorted_idx:
                                in_hdr[i] = True
                                cumulative += norm_pdf[i]
                                if cumulative >= mass_threshold:
                                    break
                            segments = []
                            start = None
                            for i, v in enumerate(in_hdr):
                                if v and start is None:
                                    start = x_grid[i]
                                elif not v and start is not None:
                                    segments.append((start, x_grid[i - 1]))
                                    start = None
                            if start is not None:
                                segments.append((start, x_grid[-1]))
                            return segments

                        # Build scatter data
                        x_grid_hdr = np.linspace(1.0, 5.0, 101)
                        scatter_rows, rows_50, rows_95 = [], [], []
                        for _sidx, _r in enumerate(results):
                            _segs50 = _compute_hdr_segments(_r['pdf'], x_grid_hdr, 0.50)
                            _segs95 = _compute_hdr_segments(_r['pdf'], x_grid_hdr, 0.95)
                            _base = {'idx': _sidx, 'name': _r['restaurant_name'],
                                     'predicted_rating': _r['predicted_rating'], 'sharpness': _r['sharpness']}
                            scatter_rows.append(_base)
                            for lo, hi in _segs50:
                                rows_50.append({**_base, 'Seg_Low': lo, 'Seg_High': hi})
                            for lo, hi in _segs95:
                                rows_95.append({**_base, 'Seg_Low': lo, 'Seg_High': hi})

                        scatter_df = pd.DataFrame(scatter_rows)
                        _ec = ['idx', 'name', 'predicted_rating', 'sharpness', 'Seg_Low', 'Seg_High']
                        df_50 = pd.DataFrame(rows_50) if rows_50 else pd.DataFrame(columns=_ec)
                        df_95 = pd.DataFrame(rows_95) if rows_95 else pd.DataFrame(columns=_ec)

                        st.subheader('Predicted Ratings with Confidence Intervals')
                        st.caption('Restaurants sorted by predicted rating (left=highest). Red bar = 50% HDR, grey halo = 95% HDR. Narrow bars = opinionated model.')
                        _x = alt.X('idx:O', title='Restaurant (sorted by predicted rating)', axis=alt.Axis(labels=False, ticks=False))
                        _halo = (alt.Chart(df_95).mark_errorbar(opacity=0.35, thickness=1)
                            .encode(x=_x, y=alt.Y('Seg_Low:Q', title='Rating [1-5]', scale=alt.Scale(domain=[1, 5])),
                                    y2=alt.Y2('Seg_High:Q'), color=alt.value('rgba(150,150,150,0.45)'),
                                    tooltip=['name:N', alt.Tooltip('predicted_rating:Q', format='.2f'), alt.Tooltip('sharpness:Q', format='.3f')]))
                        _core = (alt.Chart(df_50).mark_errorbar(opacity=0.75, thickness=4)
                            .encode(x=_x, y=alt.Y('Seg_Low:Q', title=''), y2=alt.Y2('Seg_High:Q'),
                                    color=alt.value('rgba(255,75,75,0.75)')))
                        _pts = (alt.Chart(scatter_df).mark_circle(size=70)
                            .encode(x=_x, y=alt.Y('predicted_rating:Q', scale=alt.Scale(domain=[1, 5])),
                                    color=alt.Color('sharpness:Q', scale=alt.Scale(scheme='redyellowgreen', reverse=True),
                                                    legend=alt.Legend(title='PDF Std (lower=sharper)')),
                                    tooltip=['name:N', alt.Tooltip('predicted_rating:Q', format='.2f'), alt.Tooltip('sharpness:Q', format='.3f')]))
                        st.altair_chart((_halo + _core + _pts).properties(height=380), use_container_width=True)

                        st.divider()
                        st.subheader('Per-Restaurant Rating Distributions')
                        st.caption('Each chart shows the full probability density over ratings 1-5. Narrow peaks = opinionated; flat lines = uncertain.')

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
                                            subtitle=f"Predicted: {r['predicted_rating']:.2f}â­  |  Std: {r['sharpness']:.3f}",
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
                                        subtitle=f"Predicted: {r['predicted_rating']:.2f}â­  |  Std: {r['sharpness']:.3f}",
                                    ),
                                    height=180,
                                )
                            )
                            st.altair_chart(chart, use_container_width=True)
