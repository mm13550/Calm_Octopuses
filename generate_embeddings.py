import os
import glob
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def main():
    """
    Main function to generate image embeddings using the CLIP model.
    It reads all images from the 'images' directory, processes them through the 
    OpenAI CLIP-ViT-Base-Patch32 model, normalizes the resulting feature vectors, 
    and saves them to a Parquet file for downstream use (e.g., similarity search).
    """
    image_dir = "images"
    output_dir = "embeddings"
    output_file = os.path.join(output_dir, "image_embeddings.parquet")
    
    # Ensure the outputs directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify that the target image directory exists before proceeding
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        return

    # Aggregate all supported image files (jpg, jpeg, png) from the directory
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
    print(f"Found {len(image_paths)} images in '{image_dir}'.")
    if not image_paths:
        return

    # Initialize the CLIP model and processor. Use GPU if available for faster inference.
    model_id = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{model_id}' on {device}...")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval() # Set model to evaluation mode

    # List to store the resulting embeddings and metadata for each image
    results = []
    
    print("Generating embeddings...")
    # Disable gradient calculation to conserve memory and speed up inference
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Processing Images"):
            try:
                # Open image and ensure it has 3 channels (RGB) to prevent dimension errors
                image = Image.open(path).convert("RGB")
                
                # Preprocess the image and move the input tensors to the appropriate device
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                # Extract image features. The output format varies across transformers versions, 
                # so we robustly extract the tensor (either directly or via the pooler_output attribute).
                outputs = model.get_image_features(**inputs)
                image_features = outputs if isinstance(outputs, torch.Tensor) else (outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0])
                
                # L2-Normalize the embeddings to allow for cosine similarity matching via dot product later
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Move features back to CPU RAM, convert to NumPy array, and flatten to a 1D vector
                embedding = image_features.cpu().numpy().flatten()
                
                results.append({
                    "image_path": path,
                    "embedding": embedding
                })
            except Exception as e:
                print(f"\nError processing {path}: {e}")

    # Convert the collected embeddings list into a pandas DataFrame and save as Parquet
    if results:
        df = pd.DataFrame(results)
        df.to_parquet(output_file, index=False)
        print(f"\nSaved embeddings for {len(results)} images to '{output_file}'.")
    else:
        print("\nNo embeddings generated.")

if __name__ == "__main__":
    main()
