# entrypoints/download_features.py
import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

def download_features(include_raw: bool = False, include_preprocessed: bool = False):
    """
    Download features from HF Hub.

    Args:
        include_raw: If True, also download raw match_stats for niche markets
        include_preprocessed: If True, also download preprocessed data for feature regeneration
    """
    repo_id = "czlowiekZplanety/bettip-data"
    token = os.getenv("HF_TOKEN")

    if not token:
        print("WARNING: HF_TOKEN not found. Download might fail for private repos.")

    # Base patterns - features, odds, and SportMonks odds (for BTTS)
    patterns = [
        "data/03-features/**",
        "data/odds-cache/**",
        "data/sportmonks_odds/processed/**",  # SportMonks BTTS/corners/cards odds
    ]

    # Add preprocessed data for feature regeneration
    if include_preprocessed:
        patterns.append("data/02-preprocessed/**")
        patterns.append("data/01-raw/**")  # Raw data also needed for match_stats
        print("Downloading features, odds, preprocessed/raw data, AND SportMonks odds...")
    elif include_raw:
        # Add raw data patterns for niche markets (corners, shots, fouls)
        patterns.append("data/01-raw/**/match_stats.parquet")
        print("Downloading features, odds, raw match_stats, AND SportMonks odds...")
    else:
        print("Downloading features, odds, AND SportMonks odds from Hugging Face...")

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
    parser.add_argument('--include-preprocessed', action='store_true',
                       help='Also download preprocessed data for feature regeneration')
    args = parser.parse_args()
    download_features(include_raw=args.include_raw, include_preprocessed=args.include_preprocessed)
