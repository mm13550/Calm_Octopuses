import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

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
def get_similar_images(target_embedding, df, top_k=10):
    # Retrieve all embeddings as a 2D numpy array
    all_embeddings = np.stack(df['embedding'].values)
    
    # Calculate Cosine Similarities via Dot Product (since all vectors are L2-normalized)
    similarities = np.dot(all_embeddings, target_embedding)
    
    # Add similarity scores to the dataframe
    df_scores = df.copy()
    df_scores['similarity'] = similarities
    
    # Sort backwards for highest similarity, skip the first match if it is the image itself (score ~1.0)
    # However we'll just sort and then display the top_k. The target image will logically be first.
    return df_scores.sort_values(by="similarity", ascending=False).head(top_k + 1)

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
    
    st.subheader(f"Top {top_k} Similar Images")
    
    # Display in a grid
    cols_per_row = 5
    for i in range(0, len(similar_df), cols_per_row):
        cols = st.columns(cols_per_row)
        batch = similar_df.iloc[i:i+cols_per_row]
        for col, (_, row) in zip(cols, batch.iterrows()):
            img_path = row['image_path']
            score = row['similarity']
            
            with col:
                st.markdown(f"**Score:** `{score:.4f}`")
                
                # Render Image
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        # We use a constrained column display
                        st.image(img, use_container_width=True)
                    except Exception as e:
                        st.error("Failed to render image.")
                else:
                    st.error("Image file missing.")
                
                # Show path details
                filename = os.path.basename(img_path)
                st.caption(filename)
