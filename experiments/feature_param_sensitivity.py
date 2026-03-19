#!/usr/bin/env python3
"""
Feature Parameter Sensitivity Analysis

Evaluates the impact of each feature parameter by measuring log_loss delta
when the parameter is set to its min vs max value. Parameters with zero or
near-zero sensitivity are noise dimensions that waste Optuna budget.

Usage:
    python experiments/feature_param_sensitivity.py --bet-type fouls --n-folds 3
    python experiments/feature_param_sensitivity.py --bet-type btts --deployment-config config/sniper_deployment.json
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np

from experiments.run_feature_param_optimization import (
    FeatureParamOptimizer,
    BET_TYPES,
    FEATURE_PARAM_BASE_MAP,
    NumpyEncoder,
)
from src.features.config_manager import (
    BetTypeFeatureConfig,
    get_search_space_for_bet_type,
    PARAMETER_SEARCH_SPACES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("experiments/outputs/feature_param_sensitivity")


def run_sensitivity_analysis(
    bet_type: str,
    n_folds: int = 3,
    deployment_config_path: str = None,
) -> Dict[str, Any]:
    """Run sensitivity analysis for a single bet type.

    For each parameter in the search space, evaluates log_loss at min and
    max values while keeping all other parameters at their defaults.

    Args:
        bet_type: Bet type to analyze.
        n_folds: Walk-forward folds for evaluation.
        deployment_config_path: Optional path to deployment config for
            comparing informed vs full search space.

    Returns:
        Dict with per-parameter sensitivity results and metadata.
    """
    search_space = get_search_space_for_bet_type(bet_type)
    logger.info(f"Analyzing {len(search_space)} parameters for {bet_type}")

    # Load informed params if deployment config provided
    informed_params = set()
    if deployment_config_path:
        from src.features.config_manager import (
            get_informed_search_space,
            load_selected_features_from_deployment,
        )
        features = load_selected_features_from_deployment(deployment_config_path, bet_type)
        if features:
            informed_space = get_informed_search_space(bet_type, features)
            informed_params = set(informed_space.keys())
            logger.info(f"Informed search space: {len(informed_params)} params")

    # Create optimizer (reuse its data loading and evaluation logic)
    optimizer = FeatureParamOptimizer(
        bet_type=bet_type,
        n_trials=1,  # Not used, we call evaluate_config directly
        n_folds=n_folds,
        min_bets=10,
        use_regeneration=False,
    )

    # Load features once
    features_df = optimizer.load_base_features()
    logger.info(f"Loaded {len(features_df)} matches")

    # Evaluate baseline (all defaults)
    default_config = BetTypeFeatureConfig(bet_type=bet_type)
    logger.info("Evaluating baseline (default params)...")
    baseline = optimizer.evaluate_config(default_config, features_df)
    baseline_ll = -baseline["neg_log_loss"]
    logger.info(f"Baseline log_loss: {baseline_ll:.4f}")

    # Evaluate each parameter at min and max
    results = []
    for param_name, (min_val, max_val, param_type) in search_space.items():
        default_val = getattr(default_config, param_name, None)

        # Min evaluation
        min_config = BetTypeFeatureConfig(bet_type=bet_type, **{param_name: min_val})
        min_metrics = optimizer.evaluate_config(min_config, features_df)
        min_ll = -min_metrics["neg_log_loss"]

        # Max evaluation
        max_config = BetTypeFeatureConfig(bet_type=bet_type, **{param_name: max_val})
        max_metrics = optimizer.evaluate_config(max_config, features_df)
        max_ll = -max_metrics["neg_log_loss"]

        sensitivity = abs(max_ll - min_ll)
        in_informed = param_name in informed_params if informed_params else None

        results.append({
            "param": param_name,
            "default": default_val,
            "min": min_val,
            "max": max_val,
            "ll_at_min": min_ll,
            "ll_at_max": max_ll,
            "sensitivity": sensitivity,
            "in_informed": in_informed,
        })

        logger.info(
            f"  {param_name:<28} Δlog_loss={sensitivity:.4f}  "
            f"(min={min_ll:.4f}, max={max_ll:.4f})"
            f"{'  [INFORMED]' if in_informed else ''}"
        )

    # Sort by sensitivity descending
    results.sort(key=lambda x: x["sensitivity"], reverse=True)

    return {
        "bet_type": bet_type,
        "n_folds": n_folds,
        "baseline_log_loss": baseline_ll,
        "n_params": len(results),
        "n_informed": len(informed_params) if informed_params else None,
        "params": results,
        "timestamp": datetime.now().isoformat(),
    }


def print_results(analysis: Dict[str, Any]) -> None:
    """Print formatted sensitivity analysis results."""
    bet_type = analysis["bet_type"]
    baseline = analysis["baseline_log_loss"]
    params = analysis["params"]
    has_informed = params and params[0]["in_informed"] is not None

    print(f"\n{'='*90}")
    print(f"  FEATURE PARAMETER SENSITIVITY ANALYSIS: {bet_type.upper()}")
    print(f"  Baseline log_loss: {baseline:.4f}")
    if analysis["n_informed"] is not None:
        print(f"  Informed params: {analysis['n_informed']}/{analysis['n_params']}")
    print(f"{'='*90}\n")

    header = f"{'Rank':>4}  {'Parameter':<28} {'Default':>8} {'Min':>8} {'Max':>8} {'Δlog_loss':>10}"
    if has_informed:
        header += f" {'Informed':>8}"
    print(header)
    print("-" * len(header))

    for i, p in enumerate(params, 1):
        default_str = f"{p['default']}" if p["default"] is not None else "N/A"
        line = (
            f"{i:>4}  {p['param']:<28} {default_str:>8} "
            f"{p['min']:>8} {p['max']:>8} {p['sensitivity']:>10.4f}"
        )
        if has_informed:
            marker = "YES" if p["in_informed"] else ""
            line += f" {marker:>8}"
        print(line)

    # Summary: top-5 vs bottom-5
    print(f"\n{'='*60}")
    top5 = [p["param"] for p in params[:5]]
    bottom5 = [p["param"] for p in params[-5:]]
    print(f"  Top-5 (high impact):  {', '.join(top5)}")
    print(f"  Bottom-5 (low impact): {', '.join(bottom5)}")

    if has_informed:
        informed_set = {p["param"] for p in params if p["in_informed"]}
        high_sens_not_informed = [p["param"] for p in params[:5] if not p["in_informed"]]
        low_sens_in_informed = [p["param"] for p in params[-5:] if p["in_informed"]]
        if high_sens_not_informed:
            print(f"\n  WARNING: High-sensitivity params NOT in informed set: {high_sens_not_informed}")
        if low_sens_in_informed:
            print(f"  NOTE: Low-sensitivity params IN informed set: {low_sens_in_informed}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Feature Parameter Sensitivity Analysis")
    parser.add_argument("--bet-type", type=str, required=True,
                        help="Bet type to analyze")
    parser.add_argument("--n-folds", type=int, default=3,
                        help="Walk-forward folds for evaluation")
    parser.add_argument("--deployment-config", type=str, default=None,
                        help="Path to sniper_deployment.json (shows informed vs full comparison)")
    args = parser.parse_args()

    # Map line variants to base market
    base_bet_type = re.sub(r'_(over|under)_\d+$', '', args.bet_type)
    base_bet_type = FEATURE_PARAM_BASE_MAP.get(base_bet_type, base_bet_type)
    if base_bet_type not in BET_TYPES:
        logger.error(f"Unknown bet type: {args.bet_type}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analysis = run_sensitivity_analysis(
        bet_type=base_bet_type,
        n_folds=args.n_folds,
        deployment_config_path=args.deployment_config,
    )

    print_results(analysis)

    # Save JSON
    output_path = OUTPUT_DIR / f"sensitivity_{base_bet_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
