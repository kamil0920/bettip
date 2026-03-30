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
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ml.deployment_gates import check_deployment_gates


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


def _model_age_days(market: dict) -> int | None:
    """Return age in days of a market's trained model, or None if unknown."""
    trained_date = market.get('trained_date')
    if not trained_date:
        return None
    try:
        td = datetime.fromisoformat(trained_date[:10])
        return (datetime.now() - td).days
    except (ValueError, TypeError):
        return None


def _get_holdout_metric(market: dict, metric: str) -> float:
    """Extract a holdout metric from a market config, handling all config formats.

    Three formats exist in production:
      1. Nested ``holdout_metrics`` dict (preferred, always present in new configs)
      2. Flat ``holdout_{metric}`` top-level fields (legacy sniper-direct format)
      3. Top-level ``precision``/``roi`` fields (from generate_config S31+, already holdout-sourced)

    Returns 0.0 when the metric is missing or None.
    """
    # 1. Try nested holdout_metrics (most reliable source)
    holdout = market.get('holdout_metrics')
    if isinstance(holdout, dict):
        val = holdout.get(metric)
        if val is not None:
            return float(val)

    # 2. Try flat holdout_{metric} (legacy format)
    flat_val = market.get(f'holdout_{metric}')
    if flat_val is not None:
        return float(flat_val)

    # 3. Fall back to top-level field (from generate_config, already holdout-sourced)
    top_val = market.get(metric)
    if top_val is not None:
        return float(top_val)

    return 0.0


def is_better(
    new_market: dict,
    old_market: dict,
    metric: str = 'roi',
    max_model_age_days: int = 0,
    staleness_tolerance: float = 0.05,
    min_holdout_bets: int = 20,
) -> tuple[bool, str]:
    """
    Compare new market config against old one.

    Hard gates fire BEFORE any metric comparison — a new market that fails
    gates is still rejected (no free pass for ``not old_market``).

    When max_model_age_days > 0 and the old model is stale (older than
    max_model_age_days), accept the new model if the regression is within
    staleness_tolerance (e.g. 0.05 = 5% relative drop allowed).

    Returns:
        Tuple of (is_better, reason)
    """
    # ── Hard gates (fire before any metric comparison) ────────────────
    holdout = new_market.get('holdout_metrics') or {}

    # Gate 1: Zero / insufficient holdout bets
    n_bets = holdout.get('n_bets')
    if n_bets is None or n_bets < min_holdout_bets:
        actual = n_bets if n_bets is not None else 0
        return False, f"REJECTED: holdout n_bets={actual} < minimum {min_holdout_bets}"

    # Gate 2: TS-rejected results
    if holdout.get('ts_rejected', False):
        return False, "REJECTED: holdout tracking signal exceeded max_ts"

    # Gate 3: ECE > 0.10
    new_ece = holdout.get('ece')
    if new_ece is not None and new_ece > 0.10:
        return False, f"REJECTED: holdout ECE {new_ece:.3f} > 0.10"

    # Gate 4: CLEAN-to-CAUTION regression (|TS| threshold: 4.0)
    old_holdout = (old_market or {}).get('holdout_metrics') or {}
    old_ts = old_holdout.get('tracking_signal')
    new_ts = holdout.get('tracking_signal')
    old_ts_clean = (
        old_ts is not None
        and not (isinstance(old_ts, float) and math.isnan(old_ts))
        and abs(old_ts) < 4.0
    )
    new_ts_caution = (
        new_ts is not None
        and not (isinstance(new_ts, float) and math.isnan(new_ts))
        and abs(new_ts) >= 4.0
    )
    if old_ts_clean and new_ts_caution:
        return False, (
            f"REJECTED: CLEAN→CAUTION regression "
            f"(old |TS|={abs(old_ts):.2f} → new |TS|={abs(new_ts):.2f})"
        )

    # ── Standard comparison ───────────────────────────────────────────
    if not old_market:
        return True, "new market"

    # Always compare holdout (out-of-sample) metrics, never WF optimization metrics.
    # New configs from generate_config() store holdout values in top-level fields,
    # but old deployed configs may only have them in holdout_metrics or flat holdout_* fields.
    new_val = _get_holdout_metric(new_market, metric)
    old_val = _get_holdout_metric(old_market, metric)

    # For most metrics, higher is better
    if metric in ('roi', 'sharpe', 'sortino', 'p_profit', 'precision'):
        if new_val > old_val:
            return True, f"holdout {metric}: {old_val:.4f} → {new_val:.4f} (+{new_val - old_val:.4f})"

        # Check staleness tolerance: accept small regression for stale models
        if max_model_age_days > 0:
            age = _model_age_days(old_market)
            if age is not None and age > max_model_age_days:
                # Allow regression within tolerance for stale models
                if old_val == 0 or (old_val - new_val) / max(abs(old_val), 1e-6) <= staleness_tolerance:
                    return True, (
                        f"holdout {metric}: {old_val:.4f} → {new_val:.4f} "
                        f"({new_val - old_val:.4f}) [STALE: {age}d old, "
                        f"within {staleness_tolerance:.0%} tolerance]"
                    )

        return False, f"holdout {metric}: {old_val:.4f} → {new_val:.4f} ({new_val - old_val:.4f})"

    return new_val > old_val, f"holdout {metric}: {old_val:.4f} → {new_val:.4f}"


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
            model = entry.get('best_model', 'XGBoost')
            threshold = entry.get('best_threshold', 0.5)

            # Holdout metrics live in a sub-dict — these are the UNBIASED metrics
            holdout = entry.get('holdout_metrics') or {}
            # S31 fix: NEVER fall through to WF metrics when holdout has 0 or
            # missing n_bets.  WF precision is inflated and wins comparison
            # against real holdout numbers — root cause of auto-regression bug.
            holdout_n_bets = holdout.get('n_bets') if isinstance(holdout, dict) else None
            if holdout_n_bets and holdout_n_bets > 0:
                roi = holdout.get('roi') or 0
                precision = holdout.get('precision') or 0
                sharpe = holdout.get('sharpe', 0) or 0
                sortino = holdout.get('sortino', 0) or 0
            else:
                # Zero holdout bets — do NOT fall through to WF (optimization set) metrics
                roi = 0
                precision = 0
                sharpe = 0
                sortino = 0

            # Always enable — models should stay active; sniper runs update thresholds
            enabled = True

            market_config = {
                "enabled": enabled,
                "model": model,
                "threshold": round(threshold, 4),
                "precision": round(precision, 4),
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
                # One-sided conformal prediction (S54+)
                "conformal_tau": entry.get('conformal_tau'),
                "conformal_alpha": entry.get('conformal_alpha'),
                # Holdout (unbiased) metrics
                "holdout_metrics": holdout if holdout else None,
                "holdout_uncertainty_roi": entry.get('holdout_uncertainty_roi'),
                # Model freshness tracking
                "trained_date": (
                    entry.get('training_data_end_date')
                    or entry.get('timestamp')
                    or datetime.now().isoformat()
                )[:10],
                # Per-league holdout validation (OOD guard)
                "approved_leagues": entry.get('approved_leagues'),
                "holdout_league_stats": entry.get('holdout_league_stats'),
            }

            config["markets"][bet_type] = market_config

    return config


ENSEMBLE_STRATEGIES = {
    'stacking', 'average', 'agreement', 'temporal_blend',
    'disagree_lgb_filtered', 'disagree_xgb_filtered', 'disagree_cat_filtered',
    'disagree_conservative_filtered', 'disagree_balanced_filtered',
    'disagree_aggressive_filtered',
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
    protected_markets: list[str] | None = None,
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
        model = (cfg.get('model') or '').lower()

        # 1. Strategy-model count mismatch (config-gen specific)
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

        # 2. Artifact existence check (local models dir)
        if models_dir and saved_models:
            for model_path in saved_models:
                filename = os.path.basename(model_path)
                if not (models_dir / filename).exists():
                    validation_warnings.append(
                        f"[{market}] Model artifact missing locally: {filename}"
                    )

        # 3. Shared deployment gates (auto-disable, unless protected)
        if cfg.get('enabled', False):
            violations = check_deployment_gates(
                market,
                cfg,
                max_ece=max_ece,
                min_bets_fallback=min_n_bets if min_n_bets > 0 else 50,
            )
            if violations:
                if protected_markets and market in protected_markets:
                    for v in violations:
                        validation_warnings.append(
                            f"[{market}] WARNING (PROTECTED): {v} — NOT auto-disabled"
                        )
                else:
                    for v in violations:
                        validation_warnings.append(
                            f"[{market}] BLOCKED: {v}"
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
    parser.add_argument('--metric', type=str, default='precision',
                        choices=['precision', 'roi', 'sharpe', 'sortino', 'p_profit'],
                        help='Metric to use for comparison (default: precision)')
    parser.add_argument('--min-n-bets', type=int, default=60,
                        help='Minimum holdout bets to enable market (default: 60)')
    parser.add_argument('--max-ece', type=float, default=0.10,
                        help='Maximum ECE to enable market (default: 0.10)')
    parser.add_argument('--force-overwrite', action='store_true',
                        help='Deploy without merging current config (DANGEROUS: loses protected markets)')
    parser.add_argument('--max-model-age-days', type=int, default=0,
                        help='Accept fresher models with small regression when old model exceeds this age (0=disabled)')
    parser.add_argument('--staleness-tolerance', type=float, default=0.05,
                        help='Max relative metric regression allowed for stale models (default: 0.05 = 5%%)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run all comparisons and print results, but skip file write and HF Hub upload')
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
    if args.max_model_age_days > 0:
        print(f"Staleness gate: accept {args.staleness_tolerance:.0%} regression for models >{args.max_model_age_days}d old")
    if args.dry_run:
        print("DRY RUN: will print results but skip file write")
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
                better, reason = is_better(
                    new_cfg, old_cfg, args.metric,
                    max_model_age_days=args.max_model_age_days,
                    staleness_tolerance=args.staleness_tolerance,
                    min_holdout_bets=args.min_n_bets,
                )

                if better:
                    final_markets[market] = new_cfg
                    updates.append((market, reason, new_cfg.get('enabled', False)))
                else:
                    # Keep old config
                    if old_cfg:
                        # Backfill approved_leagues from new run even when
                        # keeping old model — the league validation data is
                        # independent of model quality.
                        for _league_key in ("approved_leagues", "holdout_league_stats"):
                            if old_cfg.get(_league_key) is None and new_cfg.get(_league_key) is not None:
                                old_cfg[_league_key] = new_cfg[_league_key]
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
        if not args.force_overwrite:
            print("ERROR: Could not download current config from HF Hub.")
            print("Cannot safely merge — would lose protected markets.")
            print("Use --force-overwrite to deploy without merge (DANGEROUS).")
            return 1
        print("WARNING: --force-overwrite used — deploying all new results without merge")

    # Validate config (critical warnings auto-disable markets in-place)
    protected = new_config.get('protected_markets', [])
    models_dir = project_root / 'models'
    validation_warnings = validate_config(
        new_config,
        models_dir=models_dir if models_dir.exists() else None,
        min_n_bets=args.min_n_bets,
        max_ece=args.max_ece,
        protected_markets=protected,
    )

    # Restore protected markets that were disabled by validation or missing from merge
    final_markets = new_config.get('markets', {})
    if current_config:
        current_markets = current_config.get('markets', {})
        for market in protected:
            cfg = final_markets.get(market)
            if cfg and not cfg.get('enabled'):
                print(f"  PROTECTED: Re-enabling {market} (was disabled by validation)")
                cfg['enabled'] = True
            elif not cfg and market in current_markets:
                final_markets[market] = current_markets[market]
                final_markets[market]['enabled'] = True
                print(f"  PROTECTED: Restoring {market} from current config")
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

    # P1: Conflict guard — re-download and re-merge if config was modified
    # by a concurrent run since we first downloaded it.
    if current_config and args.only_if_better:
        current_config_ts = current_config.get('generated_at', '')
        fresh_config = download_current_config()
        if fresh_config:
            fresh_ts = fresh_config.get('generated_at', '')
            if fresh_ts != current_config_ts:
                print(
                    f"\n  CONFLICT GUARD: Config modified by concurrent run "
                    f"({current_config_ts} → {fresh_ts}). Re-merging."
                )
                final_markets = new_config.get('markets', {})
                for market, cfg in fresh_config.get('markets', {}).items():
                    if market not in new_markets:
                        final_markets[market] = cfg
                new_config['markets'] = final_markets
                print(f"  Re-merged: {len(final_markets)} total markets")

    # Print summary
    print("\n" + "-"*60)
    print("Final deployment config:")
    print("-"*60)
    for market, cfg in new_config.get('markets', {}).items():
        status = "ENABLED" if cfg.get('enabled', False) else "DISABLED"
        model = cfg.get('model') or 'unknown'
        threshold = cfg.get('threshold') or 0.5
        roi = cfg.get('roi') or 0
        print(f"  {market:<12} {status:<10} {model:<12} "
              f"thresh={threshold:.2f} ROI={roi:.1f}%")

    enabled_count = sum(1 for m in new_config.get('markets', {}).values() if m.get('enabled', False))
    print("-"*60)
    print(f"Enabled markets: {enabled_count}/{len(new_config.get('markets', {}))}")

    # --dry-run: skip file write and HF Hub upload
    if args.dry_run:
        print("\n  DRY RUN — no files written, no HF Hub upload.")
        return 0

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path, 'w') as f:
        json.dump(new_config, f, indent=2)

    print(f"\nSaved to: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
