"""Shared deployment gate logic — single source of truth.

All deployment gate checks live here. Three consumers import from this module:
- experiments/generate_daily_recommendations.py (production recommendations)
- src/ml/deployment_validator.py (pre-deployment validation CLI)
- scripts/generate_deployment_config.py (CI config generation)

Gate criteria:
- holdout_metrics must exist as a dict
- saved_models must be non-empty
- ECE is REQUIRED and must be < MAX_ECE
- n_bets >= MinTRL (if available) or >= MIN_HOLDOUT_BETS_FALLBACK
- Precision >= MIN_PRECISION (when available)
- Ensemble strategies must have >= 2 saved_models
"""

from typing import Optional

# ── Gate thresholds ──────────────────────────────────────────────────
MAX_ECE: float = 0.10
MIN_HOLDOUT_BETS_FALLBACK: int = 50  # used when MinTRL unavailable
MIN_PRECISION: float = 0.55  # minimum holdout precision for deployment

# Strategies that require >= 2 saved_models (multi-model ensemble)
ENSEMBLE_STRATEGIES: frozenset[str] = frozenset({
    "stacking", "average", "agreement", "temporal_blend",
    "disagree_lgb_filtered", "disagree_xgb_filtered", "disagree_cat_filtered",
    "disagree_conservative_filtered", "disagree_balanced_filtered",
    "disagree_aggressive_filtered",
})


def check_deployment_gates(
    market_name: str,
    market_config: dict,
    *,
    max_ece: float = MAX_ECE,
    min_bets_fallback: int = MIN_HOLDOUT_BETS_FALLBACK,
    min_precision: float = MIN_PRECISION,
) -> list[str]:
    """Return list of violation strings. Empty list = market passes.

    Args:
        market_name: Market identifier (for logging context only).
        market_config: Per-market config dict from sniper_deployment.json.
        max_ece: Maximum allowed ECE (default MAX_ECE).
        min_bets_fallback: Fallback minimum bets when MinTRL is unavailable.
        min_precision: Minimum holdout precision (default MIN_PRECISION).

    Returns:
        List of human-readable violation strings. Empty means the market
        passes all gates.
    """
    violations: list[str] = []

    # Gate 0: holdout_metrics must exist and be a dict
    hm = market_config.get("holdout_metrics")
    if not isinstance(hm, dict):
        violations.append("no holdout_metrics dict")
        return violations

    # Gate 1: saved_models must be non-empty
    saved = market_config.get("saved_models") or []
    if not saved:
        violations.append("no saved_models")

    # Gate 2: ECE is REQUIRED and must be < max_ece
    ece: Optional[float] = hm.get("ece")
    if ece is None:
        # Fallback to top-level ece (legacy format)
        ece = market_config.get("ece")
    if ece is None:
        violations.append("ECE missing (required)")
    elif ece > max_ece:
        violations.append(f"ECE {ece:.3f} > {max_ece}")

    # Gate 3: n_bets >= MinTRL (preferred) or >= fallback
    n_bets = hm.get("n_bets")
    if n_bets is None:
        violations.append("n_bets missing in holdout_metrics")
    else:
        mintrl = hm.get("min_track_record_length")
        if mintrl is not None:
            if n_bets < mintrl:
                violations.append(f"n_bets {n_bets} < MinTRL {mintrl}")
        elif n_bets < min_bets_fallback:
            violations.append(
                f"n_bets {n_bets} < fallback min {min_bets_fallback}"
            )

    # Gate 4: ensemble strategies require >= 2 saved_models
    wf = market_config.get("walkforward") or {}
    strategy = (wf.get("best_model_wf") or "").lower()
    if strategy in ENSEMBLE_STRATEGIES:
        n_models = len(saved)
        if n_models < 2:
            violations.append(
                f"strategy '{strategy}' requires ≥2 models but has {n_models}"
            )

    # Gate 5: precision >= min_precision (when available)
    precision: Optional[float] = hm.get("precision")
    if precision is not None and precision < min_precision:
        violations.append(f"precision {precision:.3f} < {min_precision}")

    return violations
