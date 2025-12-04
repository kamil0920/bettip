import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, upload_folder


def main():
    parser = argparse.ArgumentParser(description="Upload data to Hugging Face Hub")
    parser.add_argument("--season", type=str, default="2025", help="Season to upload")
    args = parser.parse_args()

    REPO_ID = "czlowiekZplanety/bettip-data"
    TOKEN = os.getenv("HF_TOKEN")

    if not TOKEN:
        print("❌ Error: HF_TOKEN env variable is missing!")
        exit(1)

    print(f"🚀 Starting upload to {REPO_ID} for season {args.season}...")

    api = HfApi(token=TOKEN)

    raw_path = Path("data/01-raw")
    if raw_path.exists():
        print("Uploading Raw Data...")
        api.upload_folder(
            folder_path=str(raw_path),
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo="data/01-raw",
            commit_message=f"Update raw data for season {args.season}"
        )

    prep_path = Path("data/02-preprocessed")
    if prep_path.exists():
        print("Uploading Preprocessed Data...")
        api.upload_folder(
            folder_path=str(prep_path),
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo="data/02-preprocessed",
            commit_message=f"Update preprocessed data for season {args.season}"
        )

    feat_path = Path("data/03-features")
    if feat_path.exists():
        print("Uploading Features...")
        api.upload_folder(
            folder_path=str(feat_path),
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo="data/03-features",
            commit_message=f"Update features for season {args.season}"
        )

    print("✅ Upload finished!")


if __name__ == "__main__":
    main()