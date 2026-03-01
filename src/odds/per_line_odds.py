"""Generate per-line odds columns from default-line odds using Poisson CDF ratios.

Used to create match-specific per-line odds for training data so that training
and inference see consistent odds columns. At inference time, real per-line odds
come from The Odds API; for historical training data, we estimate them using
Poisson CDF ratio scaling from whatever default-line odds are available.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.utils.line_plausibility import compute_league_stat_averages

logger = logging.getLogger(__name__)

# Default lines for Poisson estimation (same as generate_daily_recommendations.py)
POISSON_ESTIMATION_LINES: Dict[str, float] = {
    "corners": 9.5,
    "cards": 4.5,
    "shots": 27.5,
    "fouls": 24.5,
}

# Maps stat → league average column name in features parquet
STAT_LEAGUE_COL: Dict[str, str] = {
    "corners": "total_corners",
    "cards": "total_cards",
    "shots": "total_shots",
    "fouls": "total_fouls",
}

# Candidate default-line odds columns (tried in order, first found wins)
DEFAULT_ODDS_CANDIDATES: Dict[str, Dict[str, List[str]]] = {
    "cards": {
        "over": ["theodds_cards_over_odds", "cards_over_avg"],
        "under": ["theodds_cards_under_odds", "cards_under_odds", "cards_under_avg"],
    },
    "corners": {
        "over": ["theodds_corners_over_odds", "corners_over_avg"],
        "under": ["theodds_corners_under_odds", "corners_under_odds", "corners_under_avg"],
    },
    "shots": {
        "over": ["theodds_shots_over_odds", "shots_over_avg"],
        "under": ["theodds_shots_under_odds", "shots_under_odds", "shots_under_avg"],
    },
    "fouls": {
        "over": ["fouls_over_odds", "fouls_over_avg"],
        "under": ["fouls_under_odds", "fouls_under_avg"],
    },
}

# Target per-line markets to generate
PER_LINE_TARGETS: Dict[str, Dict[str, List[float]]] = {
    "cards": {
        "over": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        "under": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    },
    "corners": {
        "over": [8.5, 9.5, 10.5, 11.5],
        "under": [8.5, 9.5, 10.5, 11.5],
    },
    "shots": {
        "over": [24.5, 25.5, 26.5, 27.5, 28.5, 29.5],
        "under": [24.5, 25.5, 26.5, 27.5, 28.5, 29.5],
    },
    "fouls": {
        "over": [22.5, 23.5, 24.5, 25.5, 26.5, 27.5],
        "under": [22.5, 23.5, 24.5, 25.5, 26.5, 27.5],
    },
}

# Vig to apply (matches fill_estimated_line_odds)
VIG = 0.05


def _line_to_col_suffix(line: float) -> str:
    """Convert line value to column suffix: 1.5 -> '15', 8.5 -> '85', 24.5 -> '245'."""
    return str(int(line * 10))


def _find_default_odds_col(
    df: pd.DataFrame, stat: str, direction: str
) -> Optional[str]:
    """Find the best available default-line odds column for a stat/direction."""
    candidates = DEFAULT_ODDS_CANDIDATES.get(stat, {}).get(direction, [])
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def generate_per_line_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-line odds columns from default-line odds using Poisson CDF ratios.

    For each niche stat (cards, corners, shots, fouls):
    - If default-line odds exist, use ratio scaling to estimate per-line odds
      (preserves match-specific information from bookmaker pricing)
    - If no default-line odds, use pure Poisson from league average lambda

    Adds columns like: cards_under_avg_25, cards_over_avg_35, corners_over_avg_85, etc.

    Args:
        df: Features DataFrame with 'league' column and optionally default-line odds.

    Returns:
        DataFrame with new per-line odds columns added.
    """
    if "league" not in df.columns:
        logger.warning("No 'league' column — skipping per-line odds generation")
        return df

    # Compute per-league lambda values
    league_stats = compute_league_stat_averages(df)
    if not league_stats:
        logger.warning("Could not compute league stats — skipping per-line odds")
        return df

    total_cols_added = 0

    for stat, default_line in POISSON_ESTIMATION_LINES.items():
        stat_col = STAT_LEAGUE_COL[stat]

        # Build per-row lambda from league averages
        lambdas = df["league"].map(
            lambda l, sc=stat_col: league_stats.get(l, {}).get(sc, np.nan)
        )
        valid_lambda = lambdas.notna() & (lambdas > 0)

        if valid_lambda.sum() == 0:
            logger.warning(f"No valid lambda for {stat} — skipping")
            continue

        # Poisson CDF for the default line
        default_floor = int(default_line)
        p_over_default = np.where(
            valid_lambda,
            1 - poisson.cdf(default_floor, np.where(valid_lambda, lambdas, 1)),
            np.nan,
        )
        p_under_default = np.where(
            valid_lambda,
            poisson.cdf(default_floor, np.where(valid_lambda, lambdas, 1)),
            np.nan,
        )

        # Guard against zero denominators
        p_over_default = np.where(
            np.abs(p_over_default) < 1e-10, np.nan, p_over_default
        )
        p_under_default = np.where(
            np.abs(p_under_default) < 1e-10, np.nan, p_under_default
        )

        # Find default-line odds columns
        over_odds_col = _find_default_odds_col(df, stat, "over")
        under_odds_col = _find_default_odds_col(df, stat, "under")

        has_both_odds = over_odds_col is not None and under_odds_col is not None
        if has_both_odds:
            over_odds = df[over_odds_col].astype(float)
            under_odds = df[under_odds_col].astype(float)
            has_odds_mask = (
                over_odds.notna()
                & (over_odds > 1.0)
                & under_odds.notna()
                & (under_odds > 1.0)
                & valid_lambda
            )

            # Vectorized remove_vig_2way
            implied_over = 1.0 / over_odds
            implied_under = 1.0 / under_odds
            total_implied = implied_over + implied_under
            # Guard against total <= 0
            total_implied = total_implied.where(total_implied > 0, np.nan)
            fair_over = implied_over / total_implied
            fair_under = implied_under / total_implied
        else:
            has_odds_mask = pd.Series(False, index=df.index)
            fair_over = pd.Series(np.nan, index=df.index)
            fair_under = pd.Series(np.nan, index=df.index)
            if over_odds_col:
                logger.info(
                    f"  {stat}: only over odds ({over_odds_col}), no under — using pure Poisson"
                )
            elif under_odds_col:
                logger.info(
                    f"  {stat}: only under odds ({under_odds_col}), no over — using pure Poisson"
                )

        # Generate per-line columns
        targets = PER_LINE_TARGETS.get(stat, {})
        stat_cols_added = 0

        for direction in ["over", "under"]:
            lines = targets.get(direction, [])
            for target_line in lines:
                col_name = f"{stat}_{direction}_avg_{_line_to_col_suffix(target_line)}"

                target_floor = int(target_line)
                lam_safe = np.where(valid_lambda, lambdas, 1)

                if direction == "over":
                    p_target = 1 - poisson.cdf(target_floor, lam_safe)
                    # Ratio scaling where default odds exist
                    ratio_prob = np.where(
                        has_odds_mask,
                        fair_over * (p_target / p_over_default),
                        np.nan,
                    )
                else:  # under
                    p_target = poisson.cdf(target_floor, lam_safe)
                    ratio_prob = np.where(
                        has_odds_mask,
                        fair_under * (p_target / p_under_default),
                        np.nan,
                    )

                # Pure Poisson fallback where no default odds
                pure_poisson_prob = np.where(valid_lambda, p_target, np.nan)

                # Combine: ratio scaling where available, pure Poisson otherwise
                fair_prob = np.where(has_odds_mask, ratio_prob, pure_poisson_prob)

                # Clamp to [0.02, 0.98]
                fair_prob = np.clip(fair_prob, 0.02, 0.98)

                # Convert to decimal odds with vig
                estimated_odds = 1.0 / (fair_prob * (1 + VIG))

                # Only write where valid
                estimated_odds = np.where(valid_lambda, estimated_odds, np.nan)
                df[col_name] = estimated_odds
                stat_cols_added += 1

        if stat_cols_added > 0:
            n_with_ratio = has_odds_mask.sum() if has_both_odds else 0
            n_pure_poisson = valid_lambda.sum() - n_with_ratio
            logger.info(
                f"  {stat}: {stat_cols_added} per-line columns generated "
                f"({n_with_ratio} ratio-scaled, {n_pure_poisson} pure Poisson)"
            )
            total_cols_added += stat_cols_added

    logger.info(f"Per-line odds: {total_cols_added} columns added to {len(df)} rows")
    return df


def get_expected_odds(
    stat: str, direction: str, line: float, typical_lambda: Optional[float] = None
) -> float:
    """Compute expected odds for a market using Poisson CDF.

    Useful for determining appropriate min_odds/max_odds search ranges.

    Args:
        stat: Market stat (cards, corners, shots, fouls).
        direction: 'over' or 'under'.
        line: The line value (e.g., 2.5, 8.5).
        typical_lambda: Override lambda. Defaults to typical league average.

    Returns:
        Expected decimal odds with ~5% vig.
    """
    default_lambdas = {"cards": 4.0, "corners": 10.5, "shots": 27.0, "fouls": 24.0}
    lam = typical_lambda or default_lambdas.get(stat, 5.0)
    floor = int(line)

    if direction == "over":
        prob = 1 - poisson.cdf(floor, lam)
    else:
        prob = poisson.cdf(floor, lam)

    prob = max(0.02, min(0.98, prob))
    return round(1.0 / (prob * (1 + VIG)), 2)


def compute_odds_search_ranges(
    stat: str, direction: str, line: float
) -> Tuple[List[float], List[float]]:
    """Compute appropriate min_odds_search and max_odds_search for a market.

    Returns search ranges centered around the expected odds for this line.

    Args:
        stat: Market stat.
        direction: 'over' or 'under'.
        line: Line value.

    Returns:
        (min_odds_search, max_odds_search) lists for Optuna.
    """
    expected = get_expected_odds(stat, direction, line)

    if expected < 1.3:
        return [1.01, 1.05, 1.10, 1.15], [1.3, 1.5, 1.8]
    elif expected < 1.8:
        return [1.05, 1.10, 1.20, 1.30], [1.6, 2.0, 2.5]
    elif expected < 2.5:
        return [1.10, 1.20, 1.40, 1.60], [2.5, 3.0, 3.5]
    elif expected < 4.0:
        return [1.40, 1.80, 2.00, 2.50], [3.5, 4.5, 5.5]
    elif expected < 8.0:
        return [2.00, 2.50, 3.00, 4.00], [6.0, 8.0, 12.0]
    else:
        return [3.00, 4.00, 5.00, 7.00], [10.0, 15.0, 20.0]
