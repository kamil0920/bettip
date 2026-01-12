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

    try:
        user_info = api.whoami()
        print(f"👤 Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"⚠️ Could not verify identity: {e}")

    dirs_to_upload = [
        "data/01-raw",
        "data/02-preprocessed",
        "data/03-features"
    ]

    for folder in dirs_to_upload:
        path = Path(folder)
        if path.exists():
            files = list(path.glob("**/*"))
            # Filter out directories
            files = [f for f in files if f.is_file()]

            print(f"📂 Uploading contents of {folder}... ({len(files)} files)")

            for f in files[:3]:
                print(f"   - Found local file: {f.name}")

            try:
                future = api.upload_folder(
                    folder_path=folder,
                    path_in_repo=folder,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    revision="main",
                    allow_patterns=["**/*"],
                    ignore_patterns=[".git", ".venv", "__pycache__", ".DS_Store"],
                    commit_message=f"Upload {folder} for season {args.season}"
                )

                print(f"✅ Success! Commit URL: {future.commit_url}")
                print(f"   (Check this URL to see where the files went)")

            except Exception as e:
                print(f"❌ ERROR: Failed to upload {folder}: {e}")
                exit(1)
        else:
            print(f"ℹ️ Skipping {folder} (not found)")

    print("🎉 All uploads finished!")


if __name__ == "__main__":
    main()
