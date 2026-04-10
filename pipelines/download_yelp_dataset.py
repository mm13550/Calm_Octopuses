"""
Yelp Dataset Downloader Pipeline

This script serves as an automated downloading tool to onboard new developers 
quickly into Phase 1 of the architecture. Instead of manually signing agreements
and dealing with temporary authenticated links, this script integrates directly
with the Kaggle API to mirror the official Yelp Open Dataset.

Usage Prerequisites:
1. Create a free Kaggle Account at https://www.kaggle.com
2. Go to Profile -> Settings -> "Create New API Token"
3. Open the downloaded `kaggle.json` file.
4. Copy the username and key into your project's `.env` file like this:
   KAGGLE_USERNAME="your_username_here"
   KAGGLE_KEY="your_key_here"
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables (KAGGLE_USERNAME and KAGGLE_KEY) before importing kaggle
load_dotenv()

def main():
    """
    Main orchestration loop for the Yelp dataset downloader.
    
    Workflow:
    1. Verify the 'kaggle' library and token exist securely without exposing credentials.
    2. Establish the `data/yelp_sandbox/` target directory safely.
    3. Trigger the programmatic download & extraction of the multi-gigabyte dataset.
    """
    print("--- Yelp Dataset Automated Downloader ---")
    
    # Establish Sandbox path
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'yelp_sandbox')
    os.makedirs(data_dir, exist_ok=True)
    
    # Try importing Kaggle, catching authentication missing errors immediately.
    try:
        if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
            print("\nERROR: Missing Kaggle Credentials in .env file!")
            print("To use this automated script seamlessly, please:")
            print("1. Log into Kaggle.com and click 'Account' -> 'Create New API Token'")
            print("2. Open the downloaded kaggle.json and copy the details to your .env:")
            print("   KAGGLE_USERNAME=\"your_username\"")
            print("   KAGGLE_KEY=\"your_key\"")
            sys.exit(1)
            
        import kaggle
    except OSError as e:
        print(f"\nERROR: Kaggle Authentication Failed: {e}")
        sys.exit(1)
    except ImportError:
        print("\nERROR: Kaggle package not installed.")
        print("Please run: pip install kaggle")
        sys.exit(1)

    print(f"\nAuthenticated successfully.")
    print("WARNING: The Yelp Dataset is massive (~5-7 GB). This process will take...")
    print("1. Several minutes to securely download.")
    print("2. Several minutes to unzip into JSON formats.")
    print(f"Target Directory: {data_dir}")
    print("\nStarting download... (Please endure silence during the zip transfer)")
    
    dataset_names = [
        "yelp-dataset/yelp-dataset",       # The core JSON reviews/users datasets
        "joshiatri/yelp-dataset-photos"    # The official Kaggle mirrored Photos tarball
    ]
    
    try:
        # The kaggle package will authenticate implicitly using the .env variables
        kaggle.api.authenticate()
        
        for dataset in dataset_names:
            print(f"\n[+] Processing: {dataset}")
            # Download and automatically unzip straight into the sandbox
            kaggle.api.dataset_download_files(
                dataset, 
                path=data_dir, 
                unzip=True,
                quiet=False
            )
            
        print("\nSUCCESS! The entire Yelp Open Dataset (Text and Photos) has been unpacked into your sandbox.")
    except Exception as exp:
        print(f"\nDownload Process completely Failed! Reason: {exp}")
        sys.exit(1)

if __name__ == "__main__":
    main()
