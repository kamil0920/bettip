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
        from huggingface_hub import hf_hub_download

        token = os.environ.get('HF_TOKEN')
        config_path = hf_hub_download(
            repo_id='czlowiekZplanety/bettip-data',
            filename='config/sniper_deployment.json',
            repo_type='dataset',
            token=token
        )
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


def parse_strategy(strategy: str) -> tuple:
    """Parse strategy string like 'LogisticReg >= 0.45' into model and threshold."""
    if not strategy:
        return 'XGBoost', 0.5
    parts = strategy.split(' >= ')
    model = parts[0] if parts else 'XGBoost'
    threshold = float(parts[1]) if len(parts) > 1 else 0.5
    return model, threshold


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

        # Skip if data is not a dict (e.g., some files contain lists)
        if not isinstance(data, dict):
            print(f"  Skipping non-dict JSON: {f.name}")
            continue

        bet_type = data.get('bet_type')
        if not bet_type:
            continue

        roi = data.get('best_roi', 0)
        p_profit = data.get('best_p_profit', 0)
        strategy = data.get('best_strategy', '')
        model, threshold = parse_strategy(strategy)

        # Always enable — models should stay active; sniper runs update thresholds
        enabled = True

        # Get Sharpe-optimized alternative
        sharpe_strategy = data.get('best_sharpe_strategy', '')
        sharpe_model, sharpe_threshold = parse_strategy(sharpe_strategy)

        market_config = {
            "enabled": enabled,
            "model": model,
            "threshold": threshold,
            "roi": round(roi, 2),
            "p_profit": round(p_profit, 3),
            "sharpe": round(data.get('best_sharpe', 0), 4),
            "sortino": round(data.get('best_sortino', 0), 4),
            "n_bets": data.get('best_bets', 0),
            "selected_features": data.get('selected_features', []),
            "best_params": data.get('best_params', {}),
            "saved_models": data.get('saved_models', []),
            # Walk-forward validation results
            "walkforward": data.get('walkforward', {}),
            # Meta-learner stacking weights
            "stacking_weights": data.get('stacking_weights'),
            "stacking_alpha": data.get('stacking_alpha'),
            # Risk-adjusted alternative
            "sharpe_optimized": {
                "model": sharpe_model,
                "threshold": sharpe_threshold,
                "sharpe": round(data.get('best_sharpe_value', 0), 4),
                "roi": round(data.get('best_sharpe_roi', 0), 2),
            }
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


def validate_config(config: dict, models_dir: Path | None = None) -> list[str]:
    """
    Validate deployment config for common issues.

    Returns list of warning messages. Warnings are non-blocking — they are
    printed to the log but do not prevent deployment.
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

    # If --only-if-better, compare against current deployed config
    if args.only_if_better:
        print("Downloading current deployment config from HF Hub...")
        current_config = download_current_config()

        if current_config:
            print("\n" + "="*60)
            print("COMPARISON: New vs Current")
            print("="*60)

            current_markets = current_config.get('markets', {})
            new_markets = new_config.get('markets', {})
            final_markets = {}

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

            # Keep markets from current config that weren't in new results
            for market, old_cfg in current_markets.items():
                if market not in new_markets:
                    final_markets[market] = old_cfg
                    print(f"  {market:<12} KEPT (not in new results)")

            # Print updates
            if updates:
                print("\n✅ UPDATING (better results):")
                for market, reason, enabled in updates:
                    status = "ENABLED" if enabled else "DISABLED"
                    print(f"  {market:<12} {status:<10} {reason}")

            if skipped:
                print("\n⏸️  KEEPING CURRENT (new results not better):")
                for market, reason, enabled in skipped:
                    status = "ENABLED" if enabled else "DISABLED"
                    print(f"  {market:<12} {status:<10} {reason}")

            # Update config with merged markets
            new_config['markets'] = final_markets
            new_config['comparison'] = {
                'metric': args.metric,
                'updated_markets': [u[0] for u in updates],
                'kept_markets': [s[0] for s in skipped],
                'previous_config_date': current_config.get('generated_at', 'unknown')
            }

            print(f"\n📊 Summary: {len(updates)} updated, {len(skipped)} kept current")
        else:
            print("No current config found - deploying all new results")

    # Validate config
    models_dir = project_root / 'models'
    validation_warnings = validate_config(
        new_config,
        models_dir=models_dir if models_dir.exists() else None,
    )
    if validation_warnings:
        print("\n" + "="*60)
        print("VALIDATION WARNINGS")
        print("="*60)
        for w in validation_warnings:
            print(f"  WARN: {w}")
        print(f"\n{len(validation_warnings)} warning(s) — deployment proceeds anyway")
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
        status = "ENABLED" if cfg['enabled'] else "DISABLED"
        print(f"  {market:<12} {status:<10} {cfg['model']:<12} "
              f"thresh={cfg['threshold']:.2f} ROI={cfg['roi']:.1f}%")

    enabled_count = sum(1 for m in new_config.get('markets', {}).values() if m['enabled'])
    print("-"*60)
    print(f"Enabled markets: {enabled_count}/{len(new_config.get('markets', {}))}")
    print(f"\nSaved to: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
