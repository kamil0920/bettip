#!/usr/bin/env python3
"""
One-off script to update deployment config from R137/R138/R139 sniper results.
Replaces leaked R90 models with clean post-fix models.

The sniper JSON files are truncated (adversarial_validation section), so this
script uses a JSON repair approach: truncate at the break point and close brackets.
"""
import json
import re
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent


def repair_json(path: Path) -> dict:
    """Load a potentially truncated JSON by repairing missing closing brackets."""
    with open(path) as f:
        raw = f.read()

    # Replace NaN/Infinity with null
    raw = re.sub(r'\bNaN\b', 'null', raw)
    raw = re.sub(r'\bInfinity\b', 'null', raw)
    raw = re.sub(r'\b-Infinity\b', 'null', raw)

    # Try parsing as-is first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Truncation repair: find last complete key-value pair, close all brackets
    # Strategy: remove the adversarial_validation section and close the JSON
    # Find the "adversarial_validation" key and truncate before it
    adv_match = re.search(r',\s*"adversarial_validation"\s*:', raw)
    if adv_match:
        truncated = raw[:adv_match.start()]
        # Close any open braces
        open_braces = truncated.count('{') - truncated.count('}')
        open_brackets = truncated.count('[') - truncated.count(']')
        truncated += ']' * max(0, open_brackets) + '}' * max(0, open_braces)
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    # More aggressive: find "saved_models" and work from there
    # Try to find the last cleanly-closed section
    for key in ["saved_models", "holdout_metrics", "stacking_weights", "stacking_alpha"]:
        pattern = rf'"({key})"\s*:'
        matches = list(re.finditer(pattern, raw))
        if matches:
            last = matches[-1]
            # Find the end of this value
            pos = last.end()
            depth = 0
            in_string = False
            escape = False
            for i in range(pos, len(raw)):
                c = raw[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                if in_string:
                    continue
                if c in '{[':
                    depth += 1
                elif c in '}]':
                    depth -= 1
                    if depth < 0:
                        # Found the closing of the parent
                        truncated = raw[:i+1]
                        open_b = truncated.count('{') - truncated.count('}')
                        open_br = truncated.count('[') - truncated.count(']')
                        truncated += ']' * max(0, open_br) + '}' * max(0, open_b)
                        try:
                            return json.loads(truncated)
                        except json.JSONDecodeError:
                            break

    raise ValueError(f"Cannot repair JSON: {path}")


# Paths to sniper JSON results
UPDATES = {
    "corners": project_root / "data/artifacts/run-21827497725/sniper-all-results-137/sniper_corners_20260209_160827.json",
    "btts": project_root / "data/artifacts/run-21827497725/sniper-all-results-137/sniper_btts_20260209_165245.json",
    "cards": project_root / "data/artifacts/run-21827497725/sniper-all-results-137/sniper_cards_20260209_164350.json",
    "fouls": project_root / "data/artifacts/run-21827497725/sniper-all-results-137/sniper_fouls_20260209_175153.json",
    "shots": project_root / "data/artifacts/run-21827497725/sniper-all-results-137/sniper_shots_20260209_152552.json",
    "home_win": project_root / "data/artifacts/run-21827500715/sniper-all-results-139/sniper_home_win_20260209_150340.json",
    "over25": project_root / "data/artifacts/run-21827500715/sniper-all-results-139/sniper_over25_20260209_160034.json",
}

SOURCE_RUNS = {
    "corners": "R137 (post-fix, adversarial filter, non-neg stacking, 150 trials)",
    "btts": "R137 (post-fix, adversarial filter, non-neg stacking, 150 trials)",
    "cards": "R137 (post-fix, adversarial filter, non-neg stacking, 150 trials)",
    "fouls": "R137 (post-fix, adversarial filter, non-neg stacking, 150 trials)",
    "shots": "R137 (post-fix, adversarial filter, non-neg stacking, 150 trials)",
    "home_win": "R139 (post-fix, adversarial filter, non-neg stacking, decay=0.005, 75 trials)",
    "over25": "R139 (post-fix, adversarial filter, non-neg stacking, decay=0.005, 75 trials)",
}


def build_market_config(market: str, sniper: dict) -> dict:
    """Build deployment config entry from sniper result JSON."""
    wf = sniper.get("walkforward", {})
    wf_summary = wf.get("summary", {})
    best_model_wf = wf.get("best_model_wf", sniper.get("best_model", ""))

    best_wf = wf_summary.get(best_model_wf, {})
    wf_roi = best_wf.get("avg_roi", sniper.get("roi", 0))
    wf_std = best_wf.get("std_roi", 0)
    wf_bets = best_wf.get("total_bets", 0)
    wf_folds = best_wf.get("n_folds", 0)

    ho = sniper.get("holdout_metrics", {})
    saved = sniper.get("saved_models", [])

    if len(saved) == 1:
        model_name = sniper.get("best_model", best_model_wf)
    elif len(saved) >= 2:
        model_name = best_model_wf if best_model_wf in [
            "average", "stacking", "temporal_blend",
            "disagree_conservative_filtered", "disagree_balanced_filtered",
            "disagree_aggressive_filtered"
        ] else sniper.get("best_model", "average")
    else:
        model_name = sniper.get("best_model", "average")

    if model_name in ["average", "stacking", "temporal_blend",
                       "disagree_conservative_filtered", "disagree_balanced_filtered",
                       "disagree_aggressive_filtered"]:
        base_models = [s.replace(f"{market}_", "").replace(".joblib", "") for s in saved]
        best_params = {"ensemble_type": model_name, "base_models": base_models}
    else:
        best_params = sniper.get("best_params", {})

    return {
        "enabled": True,
        "model": model_name,
        "threshold": sniper.get("best_threshold", 0.6),
        "roi": round(wf_roi, 2),
        "p_profit": 1.0,
        "sharpe": ho.get("sharpe", 0),
        "sortino": ho.get("sortino", 0.0),
        "n_bets": wf_bets,
        "source_run": SOURCE_RUNS.get(market, ""),
        "selected_features": sniper.get("optimal_features", []),
        "best_params": best_params,
        "saved_models": saved,
        "walkforward": {
            "best_model": best_model_wf,
            "avg_roi": round(wf_roi, 2),
            "std_roi": round(wf_std, 2) if wf_std else None,
            "n_folds": wf_folds,
            "total_bets": wf_bets,
        },
        "holdout_metrics": ho if ho else {},
        "stacking_weights": sniper.get("stacking_weights", {}),
        "stacking_alpha": sniper.get("stacking_alpha", 1.0),
        "calibration_method": "sigmoid",
        "sample_decay_rate": sniper.get("sample_decay_rate", 0.002),
        "threshold_alpha": sniper.get("threshold_alpha", None),
        "min_odds": max(sniper.get("best_min_odds", 1.5), 1.5),
        "max_odds": sniper.get("best_max_odds", 2.5),
        # Uncertainty (MAPIE conformal)
        "uncertainty_penalty": sniper.get("uncertainty_penalty"),
        "holdout_uncertainty_roi": sniper.get("holdout_uncertainty_roi"),
    }


def main():
    config_path = project_root / "config" / "sniper_deployment.json"
    with open(config_path) as f:
        config = json.load(f)

    print("Updating deployment config...")
    print(f"Current markets: {list(config['markets'].keys())}")

    updated_markets = []
    for market, json_path in UPDATES.items():
        if not json_path.exists():
            print(f"  SKIP {market}: {json_path} not found")
            continue

        try:
            sniper = repair_json(json_path)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  ERROR {market}: {e}")
            continue

        new_entry = build_market_config(market, sniper)
        old_roi = config["markets"].get(market, {}).get("roi", 0)
        new_roi = new_entry["roi"]

        config["markets"][market] = new_entry
        updated_markets.append(market)
        src = new_entry['source_run'][:30]
        print(f"  UPDATE {market}: ROI {old_roi:.1f}% -> {new_roi:.1f}% ({src}...)")

    # Disable under25
    if "under25" in config["markets"]:
        config["markets"]["under25"]["enabled"] = False
        config["markets"]["under25"]["source_run"] = "DISABLED: R138 WF +0.7% — no real edge after temporal leakage fix"
        print(f"  DISABLE under25: WF +0.7% — insufficient edge")

    # Disable away_win
    if "away_win" in config["markets"]:
        config["markets"]["away_win"]["enabled"] = False
        config["markets"]["away_win"]["source_run"] = "DISABLED: empty holdout, unreliable"
        print(f"  DISABLE away_win: empty holdout")

    config["generated_at"] = datetime.utcnow().isoformat()
    config["source"] = "Post-fix deployment: R137 (niche) + R139 (H2H with decay=0.005)"
    config["comparison"] = {
        "metric": "roi",
        "updated_markets": updated_markets,
        "disabled_markets": ["under25", "away_win"],
        "previous_config_date": "2026-02-07T17:24:47.778936",
        "reason": "Replace R90 models with temporal leakage by clean post-fix models"
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nConfig written to: {config_path}")
    print(f"Updated {len(updated_markets)} markets, disabled 2")

    print("\n=== DEPLOYMENT SUMMARY ===")
    for market, entry in config["markets"].items():
        status = "ENABLED" if entry.get("enabled", False) else "DISABLED"
        model = entry.get("model", "?")
        roi = entry.get("roi", 0)
        n_models = len(entry.get("saved_models", []))
        print(f"  {market:12s} {status:8s} model={model:35s} WF_ROI={roi:+7.1f}% models={n_models}")


if __name__ == "__main__":
    main()
