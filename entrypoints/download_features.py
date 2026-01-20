# entrypoints/download_features.py
import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

def download_features(include_raw: bool = False):
    """
    Download features from HF Hub.

    Args:
        include_raw: If True, also download raw match_stats for niche markets
    """
    repo_id = "czlowiekZplanety/bettip-data"
    token = os.getenv("HF_TOKEN")

    if not token:
        print("WARNING: HF_TOKEN not found. Download might fail for private repos.")

    # Base patterns - features and odds
    patterns = ["data/03-features/**", "data/odds-cache/**"]

    # Add raw data patterns for niche markets (corners, shots, fouls)
    if include_raw:
        patterns.append("data/01-raw/**/match_stats.parquet")
        print("Downloading features, odds, AND raw match_stats from Hugging Face...")
    else:
        print("Downloading ONLY features & odds from Hugging Face...")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=".",
            allow_patterns=patterns,
            token=token
        )
        print("Features synced locally (Ready for Training)")
    except Exception as e:
        print(f"Download failed: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-raw', action='store_true',
                       help='Also download raw match_stats for niche market optimization')
    args = parser.parse_args()
    download_features(include_raw=args.include_raw)
