import os
import streamlit as st
import pandas as pd
from PIL import Image

from algorithms.image_comparison import get_similar_images

# Streamlit App Configuration
st.set_page_config(page_title="Image Similarity Explorer", layout="wide")

st.title("Image Similarity Explorer")
st.markdown("Use this tool to select an image from your dataset and instantly see the most visually similar images based on CLIP embeddings. Helpful for debugging your similarity metrics!")

# --- Data Loading ---
@st.cache_data
def load_data():
    parquet_path = os.path.join("embeddings", "image_embeddings.parquet")
    if not os.path.exists(parquet_path):
        return None
    
    df = pd.read_parquet(parquet_path)
    # The 'embedding' column contains normalized numpy arrays representing the CLIP features
    return df

df = load_data()

if df is None or df.empty:
    st.error("No embeddings found! Run `python generate_embeddings.py` first to create the `embeddings/image_embeddings.parquet` file.")
    st.stop()

# --- Similarity Computation ---
# Imported from algorithms.image_comparison

# --- UI Layout ---
# Setup the Sidebar
st.sidebar.header("Controls")

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
