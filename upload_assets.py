import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Configuration
REPO_ID = "CONFUCIUS-MDP/Calm-Octopuses-Assets"
REPO_TYPE = "dataset"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

FILES_TO_UPLOAD = {
    # Destination path in HF repo -> Local absolute path
    "images.zip": DATA_DIR / "images.zip",
    "extracted_menus/final_parsed_menus.json": DATA_DIR / "extracted_menus" / "final_parsed_menus.json",
    "extracted_menus/parsed_menus.json": DATA_DIR / "extracted_menus" / "parsed_menus.json",
    "embeddings/restaurant_profiles.jsonl": DATA_DIR / "embeddings" / "restaurant_profiles.jsonl",
    "embeddings/menu_embeddings.jsonl": DATA_DIR / "embeddings" / "menu_embeddings.jsonl",
    "embeddings/review_embeddings.jsonl": DATA_DIR / "embeddings" / "review_embeddings.jsonl",
    "embeddings/restaurant_metadata.json": DATA_DIR / "embeddings" / "restaurant_metadata.json",
    "yelp_sandbox/mdn_models/clip_v2/clip_v2_full.ckpt": DATA_DIR / "yelp_sandbox" / "mdn_models" / "clip_v2" / "clip_v2_full.ckpt",
}

def zip_images():
    """Compress the images directory to images.zip"""
    images_dir = DATA_DIR / "images"
    zip_path = DATA_DIR / "images.zip"
    
    if not images_dir.exists():
        print(f"Warning: {images_dir} does not exist. Skipping image zipping.")
        return False
        
    print(f"Compressing {images_dir} into {zip_path}...")
    shutil.make_archive(str(DATA_DIR / "images"), 'zip', str(images_dir))
    print("Compression complete!")
    return True

def upload_to_hf():
    """Upload specified files to Hugging Face dataset repository."""
    api = HfApi()
    
    try:
        api.dataset_info(repo_id=REPO_ID)
        print(f"Found repository: {REPO_ID}")
    except HfHubHTTPError as e:
        print(f"Error accessing repository {REPO_ID}: {e}")
        print("\nPlease make sure you are logged in using `huggingface-cli login`")
        print(f"and that the repository {REPO_ID} exists.")
        return

    print("\nStarting uploads...")
    for repo_path, local_path in FILES_TO_UPLOAD.items():
        if not local_path.exists():
            print(f"  [SKIP] Local file not found: {local_path}")
            continue
            
        print(f"  [UPLOAD] {local_path.name} -> {repo_path}")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )
            print(f"  [SUCCESS] Uploaded {local_path.name}")
        except Exception as e:
            print(f"  [ERROR] Failed to upload {local_path.name}: {e}")
            
    print("\nAll uploads completed!")

if __name__ == "__main__":
    print("=== Calm Octopuses Asset Uploader ===")
    zip_images()
    upload_to_hf()
