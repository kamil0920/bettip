import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Upload data to Hugging Face Hub")
    parser.add_argument("--season", type=str, default="2025", help="Season to upload")
    args = parser.parse_args()

    REPO_ID = os.getenv("HF_REPO_ID", "czlowiekZplanety/bettip-data")
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
        # Upload data folder
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
                "05-recommendations/**",
                "06-prematch/**",
                "odds-cache/**",
                "prematch_odds/**",
                "cache/**",
                "sportmonks_backup/**",  # SportMonks backup with merged odds for sniper optimization
            ],
            ignore_patterns=[".git", ".venv", "__pycache__", ".DS_Store"],
            commit_message=f"Update data pipeline: Season {args.season}"
        )

        print(f"✅ Data uploaded successfully.")
        print(f"🔗 Commit URL: {future.commit_url}")

        # Upload sensitive configs (not in public repo)
        print(f"📦 Uploading sensitive configs...")
        config_files = [
            ("config/strategies.yaml", "config/strategies.yaml"),
            ("config/sniper_deployment.json", "config/sniper_deployment.json"),
        ]

        for local_path, repo_path in config_files:
            if Path(local_path).exists():
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Update {repo_path}"
                )
                print(f"  ✅ {local_path}")

        # Upload feature_params folder
        feature_params_dir = Path("config/feature_params")
        if feature_params_dir.exists():
            api.upload_folder(
                folder_path=str(feature_params_dir),
                path_in_repo="config/feature_params",
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=["*.yaml"],
                ignore_patterns=["CLAUDE.md"],
                commit_message="Update feature params"
            )
            print(f"  ✅ config/feature_params/")

        # Upload models folder
        models_dir = Path("models")
        if models_dir.exists():
            api.upload_folder(
                folder_path=str(models_dir),
                path_in_repo="models",
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=["*.joblib"],
                commit_message="Update trained models"
            )
            print(f"  ✅ models/*.joblib")

        print(f"✅ All uploads complete.")

    except Exception as e:
        print(f"❌ ERROR: Failed to upload data: {e}")
        exit(1)


if __name__ == "__main__":
    main()
