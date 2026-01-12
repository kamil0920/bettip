import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload data to Hugging Face Hub")
    parser.add_argument("--season", type=str, default="2025", help="Season to upload")
    args = parser.parse_args()

    REPO_ID = "czlowiekZplanety/bettip-data"
    TOKEN = os.getenv("HF_TOKEN")

    if not TOKEN:
        print("❌ Error: HF_TOKEN env variable is missing!")
        exit(1)

    print(f"🚀 Starting consolidated upload to {REPO_ID}...")

    api = HfApi(token=TOKEN)

    folder_to_upload = "data"

    if not Path(folder_to_upload).exists():
        print(f"❌ Error: 'data' folder not found!")
        exit(1)

    try:
        print(f"📦 Uploading 'data/' folder (01-raw, 02-preprocessed, 03-features)...")

        future = api.upload_folder(
            folder_path=folder_to_upload,
            path_in_repo="data",
            repo_id=REPO_ID,
            repo_type="dataset",
            revision="main",
            allow_patterns=[
                "01-raw/**",
                "02-preprocessed/**",
                "03-features/**",
                "odds-cache/**"
            ],
            ignore_patterns=[".git", ".venv", "__pycache__", ".DS_Store"],
            commit_message=f"Update data pipeline: Season {args.season}"
        )

        print(f"✅ Success! All data updated in a SINGLE commit.")
        print(f"🔗 Commit URL: {future.commit_url}")

    except Exception as e:
        print(f"❌ ERROR: Failed to upload data: {e}")
        exit(1)


if __name__ == "__main__":
    main()
