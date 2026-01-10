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

    print(f"🚀 Starting smart upload to {REPO_ID}...")

    api = HfApi(token=TOKEN)

    dirs_to_upload = [
        "data/01-raw",
        "data/02-preprocessed",
        "data/03-features"
    ]

    for folder in dirs_to_upload:
        path = Path(folder)
        if path.exists():
            # List files in folder for debugging
            files = list(path.glob("**/*.csv")) + list(path.glob("**/*.parquet"))
            print(f"📂 Uploading {folder}... ({len(files)} files)")
            for f in files[:5]:
                print(f"   - {f.relative_to(path)}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")

            try:
                api.upload_folder(
                    folder_path=folder,
                    path_in_repo=folder,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    allow_patterns=["**/*"],
                    ignore_patterns=[".git", ".venv", "__pycache__"],
                    commit_message=f"Upload {folder} for season {args.season}"
                )
                print(f"✅ {folder} uploaded successfully")
            except Exception as e:
                print(f"❌ ERROR: Failed to upload {folder}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"ℹ️ Skipping {folder} (not found)")

    print("🎉 All uploads finished!")

if __name__ == "__main__":
    main()