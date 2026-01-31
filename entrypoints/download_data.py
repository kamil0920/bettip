# entrypoints/download_data.py
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()


def download_data():
    repo_id = os.getenv("HF_REPO_ID", "czlowiekZplanety/bettip-data")
    token = os.getenv("HF_TOKEN")

    if not token:
        print("⚠️ WARNING: HF_TOKEN not found. Download might fail for private repos.")

    print("📥 Downloading data from Hugging Face...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=".",
            allow_patterns=[
                "data/01-raw/**",
                "data/02-preprocessed/**",
                "data/03-features/**",
                "data/05-recommendations/**",
                "data/06-prematch/**",
                "data/odds-cache/**",
                "models/**",
                # Sensitive configs (not in public repo)
                "config/strategies.yaml",
                "config/sniper_deployment.json",
                "config/feature_params/**",
                "experiments/outputs/*_pipeline.json",
            ],
            ignore_patterns=[".gitattributes"],
            token=token
        )
        print("✅ Data synced locally (raw, preprocessed, features, models, and configs)")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        exit(1)


if __name__ == "__main__":
    download_data()
