#!/usr/bin/env python3
"""
Generate deployment config from sniper optimization results.

This script reads the latest sniper optimization outputs and creates
a unified deployment config that prediction scripts use.

Supports "deploy only when better" mode which compares new results against
the current deployed config and only updates markets that improved.

Usage:
    python scripts/generate_deployment_config.py
    python scripts/generate_deployment_config.py --source experiments/outputs
    python scripts/generate_deployment_config.py --only-if-better --metric roi
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def download_current_config() -> dict | None:
    """Download current deployment config from HF Hub."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.hf_utils import download_file

        config_path = download_file('config/sniper_deployment.json')
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not download current config: {e}")
        return None


def is_better(new_market: dict, old_market: dict, metric: str = 'roi') -> tuple[bool, str]:
    """
    Compare new market config against old one.

    Returns:
        Tuple of (is_better, reason)
    """
    if not old_market:
        return True, "new market"

    new_val = new_market.get(metric, 0) or 0
    old_val = old_market.get(metric, 0) or 0

    # For most metrics, higher is better
    if metric in ('roi', 'sharpe', 'sortino', 'p_profit'):
        if new_val > old_val:
            return True, f"{metric}: {old_val:.2f} → {new_val:.2f} (+{new_val - old_val:.2f})"
        else:
            return False, f"{metric}: {old_val:.2f} → {new_val:.2f} ({new_val - old_val:.2f})"

    return new_val > old_val, f"{metric}: {old_val:.2f} → {new_val:.2f}"


def generate_config(source_dir: Path, min_roi: float = 0, min_p_profit: float = 0.7) -> dict:
    """Generate deployment config from optimization results."""
    config = {
        "generated_at": datetime.now().isoformat(),
        "source": str(source_dir),
        "min_roi_threshold": min_roi,
        "min_p_profit_threshold": min_p_profit,
        "markets": {}
    }

    # Find all optimization result files
    patterns = ['*_full_optimization.json', 'sniper_*.json']
    files = []
    for pattern in patterns:
        files.extend(source_dir.glob(pattern))

    if not files:
        print(f"No optimization files found in {source_dir}")
        return config

    for f in sorted(files):
        try:
            with open(f) as fp:
                data = json.load(fp)
        except json.JSONDecodeError:
            print(f"  Skipping invalid JSON: {f.name}")
            continue

        # Handle both dict (single market) and list (combined results) formats
        if isinstance(data, list):
            entries = data
            from_combined = True
        elif isinstance(data, dict):
            entries = [data]
            from_combined = False
        else:
            print(f"  Skipping unexpected JSON type: {f.name}")
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            bet_type = entry.get('bet_type')
            if not bet_type:
                continue

            # Skip combined-file entries if we already have this market
            # from a per-market file (per-market files are more authoritative)
            if bet_type in config["markets"] and from_combined:
                continue

            # SniperResult field names (from run_sniper_optimization.py)
            roi = entry.get('roi', 0) or 0
            model = entry.get('best_model', 'XGBoost')
            threshold = entry.get('best_threshold', 0.5)

            # Holdout metrics live in a sub-dict
            holdout = entry.get('holdout_metrics') or {}
            sharpe = holdout.get('sharpe', 0) or 0
            sortino = holdout.get('sortino', 0) or 0

            # Always enable — models should stay active; sniper runs update thresholds
            enabled = True

            market_config = {
                "enabled": enabled,
                "model": model,
                "threshold": round(threshold, 4),
                "roi": round(roi, 2),
                "sharpe": round(sharpe, 4),
                "sortino": round(sortino, 4),
                "n_bets": entry.get('n_bets', 0),
                "ece": holdout.get('ece') if isinstance(holdout, dict) else None,
                "selected_features": entry.get('optimal_features', []),
                "best_params": entry.get('best_params', {}),
                "saved_models": entry.get('saved_models', []),
                # Walk-forward validation results
                "walkforward": entry.get('walkforward', {}),
                # Meta-learner stacking weights
                "stacking_weights": entry.get('stacking_weights'),
                "stacking_alpha": entry.get('stacking_alpha'),
                # Calibration
                "calibration_method": entry.get('calibration_method'),
                # Sniper tuning params
                "threshold_alpha": entry.get('threshold_alpha'),
                "sample_decay_rate": entry.get('sample_decay_rate'),
                # Uncertainty (MAPIE conformal)
                "uncertainty_penalty": entry.get('uncertainty_penalty'),
                # Holdout (unbiased) metrics
                "holdout_metrics": holdout if holdout else None,
                "holdout_uncertainty_roi": entry.get('holdout_uncertainty_roi'),
            }

            config["markets"][bet_type] = market_config

    return config


ENSEMBLE_STRATEGIES = {
    'stacking', 'average', 'agreement', 'temporal_blend',
    'disagree_lgb_filtered', 'disagree_xgb_filtered', 'disagree_cat_filtered',
}

SINGLE_MODEL_STRATEGIES = {
    'lightgbm', 'catboost', 'xgboost', 'fastai',
    'two_stage_lgb', 'two_stage_xgb',
}


def validate_config(
    config: dict,
    models_dir: Path | None = None,
    min_n_bets: int = 0,
    max_ece: float = 1.0,
) -> list[str]:
    """
    Validate deployment config for common issues.

    Returns list of warning messages. Critical issues (n_bets below minimum,
    ECE above maximum) auto-disable the market in-place.
    """
    validation_warnings = []

    for market, cfg in config.get('markets', {}).items():
        enabled = cfg.get('enabled', False)
        saved_models = cfg.get('saved_models', [])
        model = cfg.get('model', '').lower()

        # 1. Enabled but no saved_models
        if enabled and not saved_models:
            validation_warnings.append(
                f"[{market}] Enabled but saved_models is empty — "
                f"prediction pipeline will skip this market"
            )

        # 2. Strategy-model count mismatch
        if saved_models:
            n_models = len(saved_models)
            if model in ENSEMBLE_STRATEGIES and n_models < 2:
                validation_warnings.append(
                    f"[{market}] Ensemble strategy '{model}' has only "
                    f"{n_models} model(s) — needs at least 2"
                )
            if model in SINGLE_MODEL_STRATEGIES and n_models > 1:
                validation_warnings.append(
                    f"[{market}] Single-model strategy '{model}' has "
                    f"{n_models} models — extra models loaded unnecessarily"
                )

        # 3. Artifact existence check (local models dir)
        if models_dir and saved_models:
            for model_path in saved_models:
                filename = os.path.basename(model_path)
                if not (models_dir / filename).exists():
                    validation_warnings.append(
                        f"[{market}] Model artifact missing locally: {filename}"
                    )

        # 4. Minimum holdout bet count (auto-disable)
        if cfg.get('enabled', False) and min_n_bets > 0:
            n_bets = cfg.get('n_bets', 0)
            if n_bets < min_n_bets:
                validation_warnings.append(
                    f"[{market}] BLOCKED: {n_bets} holdout bets < minimum {min_n_bets}"
                )
                cfg['enabled'] = False

        # 5. Verify model files exist on disk (non-blocking WARNING)
        if models_dir and cfg.get('enabled', False):
            for model_path in saved_models:
                filename = os.path.basename(model_path)
                if filename and not (models_dir / filename).exists():
                    validation_warnings.append(
                        f"[{market}] WARNING: model file '{filename}' not found in {models_dir}"
                    )

        # 6. Maximum ECE — calibration quality gate (auto-disable)
        if cfg.get('enabled', False) and max_ece < 1.0:
            ece = cfg.get('ece')
            if ece is None:
                holdout_metrics = cfg.get('holdout_metrics')
                if isinstance(holdout_metrics, dict):
                    ece = holdout_metrics.get('ece')
            if ece is not None and ece > max_ece:
                validation_warnings.append(
                    f"[{market}] BLOCKED: ECE {ece:.4f} > max {max_ece}"
                )
                cfg['enabled'] = False

    return validation_warnings


def main():
    parser = argparse.ArgumentParser(description='Generate deployment config')
    parser.add_argument('--source', type=str, default='experiments/outputs',
                        help='Source directory for optimization results')
    parser.add_argument('--output', type=str, default='config/sniper_deployment.json',
                        help='Output config file path')
    parser.add_argument('--min-roi', type=float, default=0,
                        help='Minimum ROI to enable market')
    parser.add_argument('--min-p-profit', type=float, default=0.7,
                        help='Minimum P(profit) to enable market')
    parser.add_argument('--only-if-better', action='store_true',
                        help='Only update markets that improved vs current deployment')
    parser.add_argument('--metric', type=str, default='roi',
                        choices=['roi', 'sharpe', 'sortino', 'p_profit'],
                        help='Metric to use for comparison (default: roi)')
    parser.add_argument('--min-n-bets', type=int, default=20,
                        help='Minimum holdout bets to enable market (default: 20)')
    parser.add_argument('--max-ece', type=float, default=0.15,
                        help='Maximum ECE to enable market (default: 0.15)')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    source_dir = project_root / args.source
    output_path = project_root / args.output

    print("="*60)
    print("DEPLOYMENT CONFIG GENERATOR")
    print("="*60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_path}")
    print(f"Min ROI: {args.min_roi}%")
    print(f"Min P(profit): {args.min_p_profit}")
    print(f"Only if better: {args.only_if_better}")
    if args.only_if_better:
        print(f"Comparison metric: {args.metric}")
    print()

    # Generate new config from optimization results
    new_config = generate_config(source_dir, args.min_roi, args.min_p_profit)

    # Always download current config to preserve markets not in this run
    print("Downloading current deployment config from HF Hub...")
    current_config = download_current_config()

    if current_config:
        current_markets = current_config.get('markets', {})
        new_markets = new_config.get('markets', {})
        final_markets = {}

        if args.only_if_better:
            print("\n" + "="*60)
            print("COMPARISON: New vs Current")
            print("="*60)

            updates = []
            skipped = []

            for market, new_cfg in new_markets.items():
                old_cfg = current_markets.get(market, {})
                better, reason = is_better(new_cfg, old_cfg, args.metric)

                if better:
                    final_markets[market] = new_cfg
                    updates.append((market, reason, new_cfg.get('enabled', False)))
                else:
                    # Keep old config
                    if old_cfg:
                        final_markets[market] = old_cfg
                    skipped.append((market, reason, old_cfg.get('enabled', False) if old_cfg else False))

            # Print updates
            if updates:
                print("\n  UPDATING (better results):")
                for market, reason, enabled in updates:
                    status = "ENABLED" if enabled else "DISABLED"
                    print(f"  {market:<12} {status:<10} {reason}")

            if skipped:
                print("\n  KEEPING CURRENT (new results not better):")
                for market, reason, enabled in skipped:
                    status = "ENABLED" if enabled else "DISABLED"
                    print(f"  {market:<12} {status:<10} {reason}")

            new_config['comparison'] = {
                'metric': args.metric,
                'updated_markets': [u[0] for u in updates],
                'kept_markets': [s[0] for s in skipped],
                'previous_config_date': current_config.get('generated_at', 'unknown')
            }
        else:
            # Without --only-if-better, always use new results for optimized markets
            final_markets.update(new_markets)
            print(f"\nUpdating {len(new_markets)} market(s) from this run: {', '.join(new_markets)}")

        # Always preserve markets from current config that weren't in new results
        preserved = []
        for market, old_cfg in current_markets.items():
            if market not in final_markets:
                final_markets[market] = old_cfg
                preserved.append(market)
        if preserved:
            print(f"Preserved {len(preserved)} existing market(s): {', '.join(preserved)}")

        new_config['markets'] = final_markets
        print(f"\n  Summary: {len(final_markets)} total markets in config")
    else:
        print("No current config found - deploying all new results")

    # Validate config (critical warnings auto-disable markets in-place)
    models_dir = project_root / 'models'
    validation_warnings = validate_config(
        new_config,
        models_dir=models_dir if models_dir.exists() else None,
        min_n_bets=args.min_n_bets,
        max_ece=args.max_ece,
    )
    if validation_warnings:
        blocked = [w for w in validation_warnings if "BLOCKED" in w]
        other = [w for w in validation_warnings if "BLOCKED" not in w]
        print("\n" + "="*60)
        print("VALIDATION WARNINGS")
        print("="*60)
        for w in validation_warnings:
            print(f"  WARN: {w}")
        if blocked:
            print(f"\n{len(blocked)} market(s) auto-disabled (BLOCKED)")
        if other:
            print(f"{len(other)} non-blocking warning(s)")
    else:
        print("\nValidation: OK (no warnings)")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path, 'w') as f:
        json.dump(new_config, f, indent=2)

    # Print summary
    print("\n" + "-"*60)
    print("Final deployment config:")
    print("-"*60)
    for market, cfg in new_config.get('markets', {}).items():
        status = "ENABLED" if cfg.get('enabled', False) else "DISABLED"
        model = cfg.get('model', 'unknown')
        threshold = cfg.get('threshold', 0.5)
        roi = cfg.get('roi', 0)
        print(f"  {market:<12} {status:<10} {model:<12} "
              f"thresh={threshold:.2f} ROI={roi:.1f}%")

    enabled_count = sum(1 for m in new_config.get('markets', {}).values() if m.get('enabled', False))
    print("-"*60)
    print(f"Enabled markets: {enabled_count}/{len(new_config.get('markets', {}))}")
    print(f"\nSaved to: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
