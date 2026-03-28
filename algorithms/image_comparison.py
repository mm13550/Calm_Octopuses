import numpy as np

def get_similar_images(target_embedding, df, top_k=10):
    """
    Computes cosine similarity between a target embedding and all embeddings in the dataset.
    Since embeddings are L2-normalized, a dot product is sufficient.
    """
    # Retrieve all embeddings as a 2D numpy array
    all_embeddings = np.stack(df['embedding'].values)
    
    # Calculate Cosine Similarities via Dot Product
    similarities = np.dot(all_embeddings, target_embedding)
    
    # Add similarity scores to the dataframe
    df_scores = df.copy()
    df_scores['similarity'] = similarities
    
    # Sort backwards for highest similarity
    return df_scores.sort_values(by="similarity", ascending=False).head(top_k + 1)
