#!/usr/bin/env python3
"""Download a config file from HF Hub with retry.

Replaces inline hf_hub_download blocks in GitHub Actions YAML workflows.

Usage:
    python scripts/hf_download_config.py config/sniper_deployment.json
    python scripts/hf_download_config.py config/feature_params/cards.yaml
    python scripts/hf_download_config.py config/feature_params/cards.yaml --dest config/feature_params/cards.yaml
    python scripts/hf_download_config.py models/cards_over_15_catboost.joblib --force
"""

import argparse
import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hf_utils import download_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a file from HF Hub with retry")
    parser.add_argument(
        "filename",
        help="Path within the HF repo (e.g. config/sniper_deployment.json)",
    )
    parser.add_argument(
        "--dest",
        help="Local destination path (default: same as filename relative to cwd)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    args = parser.parse_args()

    try:
        kwargs = {}
        if args.force:
            kwargs["force_download"] = True

        path = download_file(args.filename, local_dir=".", **kwargs)
        print(f"Downloaded: {path}")

        # If --dest specified and different from download location, copy
        if args.dest:
            dest = Path(args.dest)
            src = Path(path)
            if dest.resolve() != src.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                print(f"Copied to: {dest}")

        return 0
    except Exception as e:
        print(f"Failed to download {args.filename}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
