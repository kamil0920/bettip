#!/usr/bin/env python3
"""
Generate deployment config from sniper optimization results.

This script reads the latest sniper optimization outputs and creates
a unified deployment config that prediction scripts use.

Usage:
    python scripts/generate_deployment_config.py
    python scripts/generate_deployment_config.py --source experiments/outputs
"""
import argparse
import json
from datetime import datetime
from pathlib import Path


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

        # Determine if market should be enabled
        enabled = roi > min_roi and p_profit >= min_p_profit

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
    print()

    config = generate_config(source_dir, args.min_roi, args.min_p_profit)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Print summary
    print("Generated config:")
    print("-"*60)
    for market, cfg in config.get('markets', {}).items():
        status = "ENABLED" if cfg['enabled'] else "DISABLED"
        print(f"  {market:<12} {status:<10} {cfg['model']:<12} "
              f"thresh={cfg['threshold']:.2f} ROI={cfg['roi']:.1f}%")

    enabled_count = sum(1 for m in config.get('markets', {}).values() if m['enabled'])
    print("-"*60)
    print(f"Enabled markets: {enabled_count}/{len(config.get('markets', {}))}")
    print(f"\nSaved to: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
