import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root on path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hf_utils import upload_file, upload_folder

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Upload data to Hugging Face Hub")
    parser.add_argument("--season", type=str, default="2025", help="Season to upload")
    args = parser.parse_args()

    TOKEN = os.getenv("HF_TOKEN")

    if not TOKEN:
        print("Error: HF_TOKEN env variable is missing!")
        exit(1)

    print("Starting consolidated upload...")

    folder_to_upload = "data"

    if not Path(folder_to_upload).exists():
        print("Error: 'data' folder not found!")
        exit(1)

    try:
        # Upload data folder
        print("Uploading 'data/' folder (01-raw, 02-preprocessed, 03-features)...")

        future = upload_folder(
            folder_path=folder_to_upload,
            path_in_repo="data",
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
            commit_message=f"Update data pipeline: Season {args.season}",
        )

        print("Data uploaded successfully.")
        print(f"Commit URL: {future.commit_url}")

        # Upload sensitive configs (not in public repo)
        print("Uploading sensitive configs...")
        config_files = [
            ("config/strategies.yaml", "config/strategies.yaml"),
            ("config/sniper_deployment.json", "config/sniper_deployment.json"),
        ]

        for local_path, repo_path in config_files:
            if Path(local_path).exists():
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    commit_message=f"Update {repo_path}",
                )
                print(f"  {local_path}")

        # Upload feature_params folder
        feature_params_dir = Path("config/feature_params")
        if feature_params_dir.exists():
            upload_folder(
                folder_path=str(feature_params_dir),
                path_in_repo="config/feature_params",
                allow_patterns=["*.yaml"],
                ignore_patterns=["CLAUDE.md"],
                commit_message="Update feature params",
            )
            print("  config/feature_params/")

        # Upload models folder
        models_dir = Path("models")
        if models_dir.exists():
            upload_folder(
                folder_path=str(models_dir),
                path_in_repo="models",
                allow_patterns=["*.joblib"],
                commit_message="Update trained models",
            )
            print("  models/*.joblib")

        print("All uploads complete.")

    except Exception as e:
        print(f"ERROR: Failed to upload data: {e}")
        exit(1)


if __name__ == "__main__":
    main()
