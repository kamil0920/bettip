#!/usr/bin/env python3
"""
Clean up orphan models from HF Hub.

Finds .joblib models on HF Hub that are not referenced by any market's
saved_models in the deployment config, and optionally deletes them.

Only targets sniper-pattern models (e.g. home_win_lightgbm.joblib),
leaving niche pipeline models (e.g. fouls_over_24_5_model.joblib) untouched.

Usage:
    python scripts/cleanup_orphan_models.py              # dry-run (default)
    python scripts/cleanup_orphan_models.py --delete      # actually delete
    python scripts/cleanup_orphan_models.py --league-group americas --delete
"""
import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv

load_dotenv()

KNOWN_MARKETS = [
    'home_win', 'away_win', 'over25', 'under25',
    'fouls', 'shots', 'corners', 'btts', 'cards',
]

KNOWN_MODEL_TYPES = [
    'lightgbm', 'catboost', 'xgboost', 'fastai',
    'two_stage_lgb', 'two_stage_xgb',
]

# Only models matching this pattern are candidates for cleanup.
# This protects niche pipeline models like fouls_over_24_5_model.joblib.
SNIPER_MODEL_PATTERN = re.compile(
    r'^(' + '|'.join(KNOWN_MARKETS) + r')_'
    r'(' + '|'.join(KNOWN_MODEL_TYPES) + r')\.joblib$'
)


def get_referenced_models(config: dict) -> set[str]:
    """Extract all saved_models filenames from deployment config."""
    referenced = set()
    for market_cfg in config.get('markets', {}).values():
        for model_path in market_cfg.get('saved_models', []):
            # saved_models entries may be bare filenames or paths like models/foo.joblib
            filename = os.path.basename(model_path)
            referenced.add(filename)
    return referenced


def find_orphans(
    hub_models: list[str],
    referenced: set[str],
) -> list[str]:
    """Find hub models that match sniper pattern but are not referenced."""
    orphans = []
    for model_path in hub_models:
        filename = os.path.basename(model_path)
        if SNIPER_MODEL_PATTERN.match(filename) and filename not in referenced:
            orphans.append(model_path)
    return sorted(orphans)


def main():
    parser = argparse.ArgumentParser(
        description='Clean up orphan models from HF Hub'
    )
    parser.add_argument(
        '--delete', action='store_true',
        help='Actually delete orphan models (default: dry-run)'
    )
    parser.add_argument(
        '--league-group', type=str, default='',
        help='League group namespace (e.g. americas)'
    )
    args = parser.parse_args()

    dry_run = not args.delete
    league_group = args.league_group

    token = os.environ.get('HF_TOKEN')
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

    repo_id = os.getenv('HF_REPO_ID', 'czlowiekZplanety/bettip-data')

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)

    # 1. Download current deployment config
    if league_group:
        config_filename = f'config/sniper_deployment_{league_group}.json'
    else:
        config_filename = 'config/sniper_deployment.json'

    print(f"Downloading deployment config: {config_filename}")
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            repo_type='dataset',
            token=token,
        )
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to download deployment config: {e}")
        sys.exit(1)

    # 2. Extract referenced models
    referenced = get_referenced_models(config)
    print(f"\nReferenced models ({len(referenced)}):")
    for m in sorted(referenced):
        print(f"  {m}")

    # 3. List all .joblib files on HF Hub under models/
    models_prefix = f'models/{league_group}/' if league_group else 'models/'
    all_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type='dataset',
    )
    hub_models = [f for f in all_files if f.startswith(models_prefix) and f.endswith('.joblib')]
    print(f"\nHub models under {models_prefix} ({len(hub_models)}):")
    for m in sorted(hub_models):
        print(f"  {m}")

    # 4. Find orphans (sniper-pattern models not in referenced set)
    orphans = find_orphans(hub_models, referenced)

    if not orphans:
        print("\nNo orphan models found.")
        return

    mode = "DRY RUN" if dry_run else "DELETING"
    print(f"\nOrphan models ({len(orphans)}) [{mode}]:")
    for orphan in orphans:
        print(f"  {orphan}")

    if dry_run:
        print(f"\nRe-run with --delete to remove {len(orphans)} orphan model(s).")
        return

    # 5. Delete orphans
    deleted = 0
    for orphan in orphans:
        try:
            api.delete_file(
                path_in_repo=orphan,
                repo_id=repo_id,
                repo_type='dataset',
                token=token,
                commit_message=f'Cleanup orphan model: {os.path.basename(orphan)}',
            )
            print(f"  Deleted: {orphan}")
            deleted += 1
        except Exception as e:
            print(f"  Failed to delete {orphan}: {e}")

    print(f"\nDeleted {deleted}/{len(orphans)} orphan model(s).")


if __name__ == '__main__':
    main()
