#!/usr/bin/env python3
"""
Apply optimized feature parameters from sniper optimization results to YAML configs.

Reads optimization JSON outputs and updates config/feature_params/{market}.yaml
with the best_params values.

Usage:
    python scripts/apply_feature_params.py --run 50
    python scripts/apply_feature_params.py --run 50 --markets home_win under25
    python scripts/apply_feature_params.py --run 50 --dry-run
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
FEATURE_PARAMS_DIR = PROJECT_ROOT / "config" / "feature_params"


def find_optimization_json(run_number: int, market: str) -> Path | None:
    """Find the feature param optimization JSON for a given run and market."""
    base = ARTIFACTS_DIR / f"sniper-all-results-{run_number}"
    param_dir = base / f"feature-params-{market}-{run_number}" / "outputs" / "feature_params"

    if not param_dir.exists():
        return None

    # Find the market-specific JSON (not the "all" one)
    for f in sorted(param_dir.glob(f"feature_params_{market}_*.json")):
        return f
    return None


def load_yaml(path: Path) -> dict:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    """Save dict to YAML, preserving readable format."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def apply_params(run_number: int, markets: list[str] | None = None, dry_run: bool = False) -> None:
    """Apply optimized params from a sniper run to YAML configs."""
    all_markets = [p.stem for p in FEATURE_PARAMS_DIR.glob("*.yaml") if p.stem != "default"]
    target_markets = markets or all_markets

    updated = 0
    for market in target_markets:
        opt_json = find_optimization_json(run_number, market)
        if opt_json is None:
            logger.info(f"{market}: no R{run_number} optimization results found, skipping")
            continue

        with open(opt_json, "r") as f:
            opt_data = json.load(f)

        best_params = opt_data.get("best_params", {})
        if not best_params:
            logger.warning(f"{market}: empty best_params, skipping")
            continue

        yaml_path = FEATURE_PARAMS_DIR / f"{market}.yaml"
        if not yaml_path.exists():
            logger.warning(f"{market}: YAML config not found at {yaml_path}")
            continue

        config = load_yaml(yaml_path)

        # Apply optimized params
        changes = []
        for key, value in best_params.items():
            old_value = config.get(key)
            if old_value != value:
                changes.append(f"  {key}: {old_value} -> {value}")
                config[key] = value

        # Update metadata
        config["optimized"] = True
        config["optimization_date"] = datetime.now().isoformat()
        config["precision"] = round(opt_data.get("precision", 0), 4)
        config["roi"] = round(opt_data.get("roi", 0), 2)
        config["n_trials"] = opt_data.get("n_trials", 0)

        if changes:
            logger.info(f"{market}: {len(changes)} param changes:")
            for c in changes:
                logger.info(c)
        else:
            logger.info(f"{market}: params already match R{run_number} optimals")

        if not dry_run:
            save_yaml(yaml_path, config)
            logger.info(f"{market}: saved to {yaml_path}")
            updated += 1
        else:
            logger.info(f"{market}: [DRY RUN] would save to {yaml_path}")

    logger.info(f"\nDone. Updated {updated} configs from R{run_number} results.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply optimized feature params to YAML configs")
    parser.add_argument("--run", type=int, required=True, help="Optimization run number (e.g., 50)")
    parser.add_argument("--markets", nargs="+", help="Specific markets to update (default: all available)")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    apply_params(args.run, args.markets, args.dry_run)


if __name__ == "__main__":
    main()
