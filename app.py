import os
import streamlit as st
import pandas as pd
from PIL import Image

from algorithms.image_comparison import get_similar_images

# Streamlit App Configuration
st.set_page_config(page_title="Image Similarity Explorer", layout="wide")

st.title("Multimodal App")

tab1, tab2 = st.tabs(["Image Similarity Explorer", "Dual Encoder Training Logs"])

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
        proc = subprocess.Popen([sys.executable, os.path.join("pipelines", "cross_modal_embeddings.py")])
        
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
