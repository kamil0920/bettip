# entrypoints/download_features.py
import os
import argparse
import time
import random
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()


def download_with_retry(
    repo_id: str,
    patterns: list,
    token: str,
    max_retries: int = 5,
    base_delay: float = 10.0
):
    """
    Download from HF Hub with exponential backoff retry.

    Args:
        repo_id: HuggingFace repo ID
        patterns: File patterns to download
        token: HF API token
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
    """
    for attempt in range(max_retries):
        try:
            # Add jitter to avoid thundering herd when parallel jobs retry
            if attempt > 0:
                jitter = random.uniform(0, 5)
                delay = base_delay * (2 ** (attempt - 1)) + jitter
                print(f"Retry {attempt}/{max_retries} after {delay:.1f}s delay...")
                time.sleep(delay)

            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=".",
                allow_patterns=patterns,
                token=token
            )
            return True

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise

    return False


def download_features(include_raw: bool = False, include_preprocessed: bool = False):
    """
    Download features from HF Hub.

    Args:
        include_raw: If True, also download raw match_stats for niche markets
        include_preprocessed: If True, also download preprocessed data for feature regeneration
    """
    repo_id = os.getenv("HF_REPO_ID", "czlowiekZplanety/bettip-data")
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
        download_with_retry(repo_id, patterns, token, max_retries=5, base_delay=10.0)
        print("Features synced locally (Ready for Training)")
    except Exception as e:
        print(f"Download failed after all retries: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-raw', action='store_true',
                       help='Also download raw match_stats for niche market optimization')
    parser.add_argument('--include-preprocessed', action='store_true',
                       help='Also download preprocessed data for feature regeneration')
    args = parser.parse_args()
    download_features(include_raw=args.include_raw, include_preprocessed=args.include_preprocessed)
