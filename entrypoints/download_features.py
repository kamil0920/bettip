# entrypoints/download_features.py
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root on path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hf_utils import download_snapshot

load_dotenv()


def download_features(include_raw: bool = False, include_preprocessed: bool = False):
    """
    Download features from HF Hub.

    Args:
        include_raw: If True, also download raw match_stats for niche markets
        include_preprocessed: If True, also download preprocessed data for feature regeneration
    """
    token = os.getenv("HF_TOKEN")

    if not token:
        print("WARNING: HF_TOKEN not found. Download might fail for private repos.")

    # Base patterns - features, odds, and SportMonks odds (for BTTS)
    patterns = [
        "data/03-features/**",
        "data/odds-cache/**",
        "data/sportmonks_odds/processed/**",  # SportMonks BTTS/corners/cards odds
        "data/sportmonks_backup/**",  # SportMonks backup with merged odds (for sniper optimization)
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
        download_snapshot(patterns)
        print("Features synced locally (Ready for Training)")
    except Exception as e:
        print(f"Download failed after all retries: {e}")
        exit(1)

    # Verify preprocessed data actually downloaded (snapshot_download can silently skip)
    if include_preprocessed:
        preprocessed_dir = Path("data/02-preprocessed")
        parquet_files = (
            list(preprocessed_dir.glob("**/matches.parquet")) if preprocessed_dir.exists() else []
        )
        if not parquet_files:
            print(
                "WARNING: Preprocessed data not found after download. Retrying with force_download=True..."
            )
            try:
                download_snapshot(
                    ["data/02-preprocessed/**", "data/01-raw/**"],
                    force_download=True,
                )
                parquet_files = list(preprocessed_dir.glob("**/matches.parquet"))
                print(f"Force download complete: {len(parquet_files)} match files found")
            except Exception as e:
                print(f"Force download failed: {e}")
                exit(1)
        else:
            print(f"Verified: {len(parquet_files)} preprocessed match files present")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also download raw match_stats for niche market optimization",
    )
    parser.add_argument(
        "--include-preprocessed",
        action="store_true",
        help="Also download preprocessed data for feature regeneration",
    )
    args = parser.parse_args()
    download_features(include_raw=args.include_raw, include_preprocessed=args.include_preprocessed)
