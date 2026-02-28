# entrypoints/download_data.py
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root on path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hf_utils import download_snapshot

load_dotenv()


def download_data():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not found. Download might fail for private repos.")

    print("Downloading data from Hugging Face...")
    try:
        download_snapshot(
            patterns=[
                "data/01-raw/**",
                "data/02-preprocessed/**",
                "data/03-features/**",
                "data/05-recommendations/**",
                "data/06-prematch/**",
                "data/odds-cache/**",
                "data/prematch_odds/**",
                "data/cache/**",
                "models/**",
                # Sensitive configs (not in public repo)
                "config/strategies.yaml",
                "config/sniper_deployment.json",
                "config/feature_params/**",
                "experiments/outputs/*_pipeline.json",
            ],
            ignore_patterns=[".gitattributes"],
        )
        print("Data synced locally (raw, preprocessed, features, models, and configs)")
    except Exception as e:
        print(f"Download failed: {e}")
        exit(1)


if __name__ == "__main__":
    download_data()
