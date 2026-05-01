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

DOWNLOAD_GROUPS = {
    "image archive": ["images.zip"],
    "parsed menus": ["extracted_menus/final_parsed_menus.json"],
    "restaurant profile embeddings": ["embeddings/restaurant_profiles.jsonl"],
    "menu embeddings": ["embeddings/menu_embeddings.jsonl"],
    "review embeddings": ["embeddings/review_embeddings.jsonl"],
    "food image embeddings": ["embeddings/image_embeddings_food.jsonl"],
    "interior image embeddings": ["embeddings/image_embeddings_interior.jsonl"],
    "restaurant metadata": ["embeddings/restaurant_metadata.json"],
    "mdn checkpoint": ["yelp_sandbox/mdn_models/clip_v2/clip_v2_full.ckpt"],
}


def _download_first_available(label: str, remote_paths: list[str]) -> bool:
    """Download the first available file from *remote_paths* into ``data/``."""
    print(f"  [FETCHING] {label} ...")

    for remote_path in remote_paths:
        local_target = DATA_DIR / remote_path
        local_target.parent.mkdir(parents=True, exist_ok=True)

        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=remote_path,
                local_dir=str(DATA_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"  [SUCCESS] Saved {remote_path} to {downloaded_path}")
            return True
        except HfHubHTTPError as e:
            print(f"    [MISS] {remote_path} not available yet ({e})")
        except Exception as e:
            print(f"    [ERROR] Failed while downloading {remote_path}: {e}")

    print(f"  [ERROR] Could not find any file for {label}. Tried: {', '.join(remote_paths)}")
    return False

def download_from_hf():
    """Download required data files from Hugging Face."""
    print(f"Downloading assets from Hugging Face dataset: {REPO_ID}")

    for label, remote_paths in DOWNLOAD_GROUPS.items():
        _download_first_available(label, remote_paths)

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
