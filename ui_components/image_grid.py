import os
import streamlit as st
from PIL import Image

def render_image_grid(similar_df, top_k):
    """
    Renders a responsive Streamlit grid of visually similar images.
    """
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
