import os
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# Configuration
REPO_ID = "CONFUCIUS-MDP/Calm-Octopuses-Assets"
REPO_TYPE = "dataset"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# List of files to download from the HF repo
FILES_TO_DOWNLOAD = [
    "images.zip",
    "extracted_menus/final_parsed_menus.json",
    "extracted_menus/parsed_menus.json",
    "embeddings/restaurant_profiles.jsonl",
    "embeddings/menu_embeddings.jsonl",
    "embeddings/review_embeddings.jsonl",
    "embeddings/restaurant_metadata.json",
    "yelp_sandbox/mdn_models/clip_v2/clip_v2_full.ckpt",
]

def download_from_hf():
    """Download required data files from Hugging Face."""
    print(f"Downloading assets from Hugging Face dataset: {REPO_ID}")
    
    for file_path in FILES_TO_DOWNLOAD:
        local_target = DATA_DIR / file_path
        
        # Create parent directories if they don't exist
        local_target.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"  [FETCHING] {file_path} ...")
        try:
            # hf_hub_download caches files locally and returns the path to the cached file
            # By default it stores in ~/.cache/huggingface/hub.
            # To place it in our data directory, we can use local_dir.
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=file_path,
                local_dir=str(DATA_DIR),
                local_dir_use_symlinks=False
            )
            print(f"  [SUCCESS] Saved to {downloaded_path}")
        except HfHubHTTPError as e:
            print(f"  [ERROR] Failed to download {file_path}. Ensure the repository is public and the file exists. Error: {e}")
        except Exception as e:
            print(f"  [ERROR] An unexpected error occurred: {e}")

def extract_images():
    """Extract the downloaded images.zip file into data/images."""
    zip_path = DATA_DIR / "images.zip"
    images_dir = DATA_DIR / "images"
    
    if not zip_path.exists():
        print(f"\nSkipping extraction: {zip_path} not found.")
        return
        
    print(f"\nExtracting {zip_path} into {images_dir}...")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        print("Extraction complete!")
        # Optionally, remove the zip file after extraction
        # zip_path.unlink()
    except Exception as e:
        print(f"Failed to extract images: {e}")

if __name__ == "__main__":
    print("=== Calm Octopuses Asset Downloader ===")
    download_from_hf()
    extract_images()
    print("\nAll setup tasks completed! You can now run the frontend.")
