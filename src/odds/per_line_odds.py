"""Generate per-line odds columns from default-line odds using count distribution CDF ratios.

Used to create match-specific per-line odds for training data so that training
and inference see consistent odds columns. At inference time, real per-line odds
come from The Odds API; for historical training data, we estimate them using
CDF ratio scaling from whatever default-line odds are available.

Uses Negative Binomial CDF for overdispersed stats (cards, corners, shots, fouls)
and Poisson CDF as fallback for stats with dispersion ratio <= 1.0.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from src.odds.count_distribution import match_varying_dispersion, overdispersed_cdf
from src.utils.line_plausibility import compute_league_stat_averages

logger = logging.getLogger(__name__)

# Default lines for Poisson estimation (same as generate_daily_recommendations.py)
POISSON_ESTIMATION_LINES: Dict[str, float] = {
    "corners": 9.5,
    "cards": 4.5,
    "shots": 27.5,
    "fouls": 24.5,
    "shots_on_target": 8.5,
    "offsides": 4.5,
    "booking_points": 40.5,
}

# Maps stat → league average column name in features parquet
STAT_LEAGUE_COL: Dict[str, str] = {
    "corners": "total_corners",
    "cards": "total_cards",
    "shots": "total_shots",
    "fouls": "total_fouls",
    "shots_on_target": "total_shots_on_target",
    "offsides": "total_offsides",
    "booking_points": "booking_points",
}

# Candidate default-line odds columns (tried in order, first found wins)
DEFAULT_ODDS_CANDIDATES: Dict[str, Dict[str, List[str]]] = {
    "cards": {
        "over": ["theodds_cards_over_odds", "cards_over_avg", "sm_cards_over_odds"],
        "under": ["theodds_cards_under_odds", "cards_under_odds", "cards_under_avg", "sm_cards_under_odds"],
    },
    "corners": {
        "over": ["theodds_corners_over_odds", "corners_over_avg", "sm_corners_over_odds"],
        "under": ["theodds_corners_under_odds", "corners_under_odds", "corners_under_avg", "sm_corners_under_odds"],
    },
    "shots": {
        "over": ["theodds_shots_over_odds", "shots_over_avg", "sm_shots_over_odds"],
        "under": ["theodds_shots_under_odds", "shots_under_odds", "shots_under_avg", "sm_shots_under_odds"],
    },
    "fouls": {
        "over": ["fouls_over_odds", "fouls_over_avg"],
        "under": ["fouls_under_odds", "fouls_under_avg"],
    },
    "shots_on_target": {
        "over": ["sot_over_odds"],
        "under": ["sot_under_odds"],
    },
    "offsides": {
        "over": ["offsides_over_odds"],
        "under": ["offsides_under_odds"],
    },
    "booking_points": {
        "over": ["bookpts_over_odds"],
        "under": ["bookpts_under_odds"],
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
        "over": [22.5, 23.5, 24.5, 25.5, 26.5, 27.5],
        "under": [22.5, 23.5, 24.5, 25.5, 26.5, 27.5],
    },
    "fouls": {
        "over": [19.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5],
        "under": [19.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5],
    },
    "shots_on_target": {
        "over": [7.5, 8.5, 9.5, 10.5],
        "under": [7.5, 8.5, 9.5, 10.5],
    },
    "offsides": {
        "over": [3.5, 4.5, 5.5],
        "under": [3.5, 4.5, 5.5],
    },
    "booking_points": {
        "over": [30.5, 40.5, 50.5],
        "under": [30.5, 40.5, 50.5],
    },
}

# Vig to apply (matches fill_estimated_line_odds)
VIG = 0.05

# --- Per-team markets (home/away stat totals) ---

# Per-team stat columns used as lambda source
PER_TEAM_STAT_COL: Dict[str, str] = {
    "home_corners": "home_corners",
    "away_corners": "away_corners",
    "home_shots": "home_shots",
    "away_shots": "away_shots",
    "home_fouls": "home_fouls",
    "away_fouls": "away_fouls",
    "home_yellow_cards": "home_yellow_cards",
    "away_yellow_cards": "away_yellow_cards",
}

# Default lines for per-team NegBin estimation (~median values)
PER_TEAM_ESTIMATION_LINES: Dict[str, float] = {
    "home_corners": 4.5,
    "away_corners": 3.5,
    "home_shots": 12.5,
    "away_shots": 10.5,
    "home_fouls": 11.5,
    "away_fouls": 11.5,
    "home_yellow_cards": 1.5,
    "away_yellow_cards": 1.5,
}

# Target per-line markets for per-team stats
PER_TEAM_LINE_TARGETS: Dict[str, Dict[str, List[float]]] = {
    "home_corners": {"over": [3.5, 4.5, 5.5, 6.5], "under": [3.5, 4.5, 5.5, 6.5]},
    "away_corners": {"over": [2.5, 3.5, 4.5, 5.5], "under": [2.5, 3.5, 4.5, 5.5]},
    "home_shots": {"over": [10.5, 12.5, 14.5, 16.5], "under": [10.5, 12.5, 14.5, 16.5]},
    "away_shots": {"over": [8.5, 10.5, 12.5, 14.5], "under": [8.5, 10.5, 12.5, 14.5]},
    "home_fouls": {"over": [10.5, 12.5, 14.5], "under": [10.5, 12.5, 14.5]},
    "away_fouls": {"over": [10.5, 12.5, 14.5], "under": [10.5, 12.5, 14.5]},
    "home_yellow_cards": {"over": [1.5, 2.5, 3.5], "under": [1.5, 2.5, 3.5]},
    "away_yellow_cards": {"over": [1.5, 2.5, 3.5], "under": [1.5, 2.5, 3.5]},
}

# Domain defaults for per-team stats (used when no history available)
PER_TEAM_DOMAIN_DEFAULTS: Dict[str, float] = {
    "home_corners": 5.4, "away_corners": 4.5,
    "home_shots": 13.7, "away_shots": 11.3,
    "home_fouls": 12.0, "away_fouls": 12.2,
    "home_yellow_cards": 2.0, "away_yellow_cards": 2.3,
}


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

    Uses per-league expanding mean with shift(1) for lambda computation to
    prevent data leakage (each match only sees past league averages).

    Adds columns like: cards_under_avg_25, cards_over_avg_35, corners_over_avg_85, etc.

    Args:
        df: Features DataFrame with 'league' and 'date' columns, plus
            optionally default-line odds.

    Returns:
        DataFrame with new per-line odds columns added.
    """
    if "league" not in df.columns:
        logger.warning("No 'league' column — skipping per-line odds generation")
        return df

    # Sort by date for correct expanding mean computation
    had_sort = False
    if "date" in df.columns:
        original_index = df.index.copy()
        df = df.sort_values("date").reset_index(drop=True)
        had_sort = True

    total_cols_added = 0

    # Compute match-varying dispersion from goal supremacy if available
    has_supremacy = "abs_goal_supremacy" in df.columns
    if has_supremacy:
        sup_values = df["abs_goal_supremacy"].fillna(0).values
        logger.info("Per-line odds: using match-varying dispersion from abs_goal_supremacy")

    for stat, default_line in POISSON_ESTIMATION_LINES.items():
        stat_col = STAT_LEAGUE_COL[stat]

        if stat_col not in df.columns:
            logger.warning(f"No {stat_col} column — skipping {stat}")
            continue

        # Per-match dispersion array (or None for fixed dispersion)
        d_array = match_varying_dispersion(stat, sup_values) if has_supremacy else None

        # Per-row lambda using expanding mean with shift(1) to prevent leakage.
        # Each match sees only the average of PAST matches in its league.
        lambdas = (
            df.groupby("league")[stat_col]
            .transform(lambda x: x.expanding().mean().shift(1))
        )

        # Fill NaN (first match per league) with global expanding mean
        global_expanding = df[stat_col].expanding().mean().shift(1)
        lambdas = lambdas.fillna(global_expanding)
        # Fill any remaining NaN (very first row) with domain default
        DOMAIN_DEFAULTS = {
            "corners": 10.5, "cards": 4.0, "shots": 27.0, "fouls": 24.0,
            "shots_on_target": 9.0, "offsides": 4.5, "booking_points": 40.0,
        }
        lambdas = lambdas.fillna(DOMAIN_DEFAULTS.get(stat, 10.0))

        valid_lambda = lambdas.notna() & (lambdas > 0)

        if valid_lambda.sum() == 0:
            logger.warning(f"No valid lambda for {stat} — skipping")
            continue

        # Count distribution CDF for the default line
        default_floor = int(default_line)
        lam_for_cdf = np.where(valid_lambda, lambdas, 1)
        p_over_default = np.where(
            valid_lambda,
            1 - overdispersed_cdf(default_floor, lam_for_cdf, stat, dispersion=d_array),
            np.nan,
        )
        p_under_default = np.where(
            valid_lambda,
            overdispersed_cdf(default_floor, lam_for_cdf, stat, dispersion=d_array),
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
                    p_target = 1 - overdispersed_cdf(target_floor, lam_safe, stat, dispersion=d_array)
                    # Ratio scaling where default odds exist
                    ratio_prob = np.where(
                        has_odds_mask,
                        fair_over * (p_target / p_over_default),
                        np.nan,
                    )
                else:  # under
                    p_target = overdispersed_cdf(target_floor, lam_safe, stat, dispersion=d_array)
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

                if col_name in df.columns:
                    # Only fill NaN — preserve pre-existing real bookmaker odds
                    mask = df[col_name].isna()
                    if mask.any():
                        df.loc[mask, col_name] = pd.Series(
                            estimated_odds, index=df.index
                        )[mask]
                else:
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

    # Restore original order if we sorted
    if had_sort and "date" in df.columns:
        df = df.set_index(original_index).sort_index()

    return df


def get_expected_odds(
    stat: str, direction: str, line: float, typical_lambda: Optional[float] = None
) -> float:
    """Compute expected odds for a market using count distribution CDF.

    Uses Negative Binomial for overdispersed stats, Poisson otherwise.
    Useful for determining appropriate min_odds/max_odds search ranges.

    Args:
        stat: Market stat (cards, corners, shots, fouls).
        direction: 'over' or 'under'.
        line: The line value (e.g., 2.5, 8.5).
        typical_lambda: Override lambda. Defaults to typical league average.

    Returns:
        Expected decimal odds with ~5% vig.
    """
    default_lambdas = {
        "cards": 4.0, "corners": 10.5, "shots": 27.0, "fouls": 24.0,
        "shots_on_target": 9.0, "offsides": 4.5, "booking_points": 40.0,
    }
    lam = typical_lambda or default_lambdas.get(stat, 5.0)
    floor = int(line)

    if direction == "over":
        prob = float(1 - overdispersed_cdf(floor, lam, stat))
    else:
        prob = float(overdispersed_cdf(floor, lam, stat))

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


def generate_per_team_line_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-line odds for per-team stat markets using NegBin CDF.

    For each per-team stat (home_corners, away_shots, etc.):
    - Computes per-league expanding mean with shift(1) as lambda
    - Uses PER_TEAM_DISPERSION_RATIOS for NegBin CDF (Poisson fallback for d <= 1.0)
    - Generates columns like: home_corners_over_avg_35, away_shots_under_avg_105

    No real bookmaker odds exist for these markets, so this is always pure NegBin.

    Args:
        df: Features DataFrame with 'league', 'date', and per-team stat columns.

    Returns:
        DataFrame with new per-team per-line odds columns added.
    """
    from src.odds.count_distribution import PER_TEAM_DISPERSION_RATIOS

    if "league" not in df.columns:
        logger.warning("No 'league' column — skipping per-team line odds generation")
        return df

    had_sort = False
    if "date" in df.columns:
        original_index = df.index.copy()
        df = df.sort_values("date").reset_index(drop=True)
        had_sort = True

    total_cols_added = 0

    for stat, stat_col in PER_TEAM_STAT_COL.items():
        if stat_col not in df.columns:
            logger.warning(f"No {stat_col} column — skipping {stat}")
            continue

        # Per-team dispersion ratio
        d = PER_TEAM_DISPERSION_RATIOS.get(stat, 1.0)

        # Per-row lambda from league expanding mean with shift(1)
        lambdas = (
            df.groupby("league")[stat_col]
            .transform(lambda x: x.expanding().mean().shift(1))
        )

        # Fill NaN (first match per league) with global expanding mean
        global_expanding = df[stat_col].expanding().mean().shift(1)
        lambdas = lambdas.fillna(global_expanding)
        # Fill remaining NaN with domain default
        lambdas = lambdas.fillna(PER_TEAM_DOMAIN_DEFAULTS.get(stat, 5.0))

        valid_lambda = lambdas.notna() & (lambdas > 0)
        if valid_lambda.sum() == 0:
            logger.warning(f"No valid lambda for {stat} — skipping")
            continue

        # Generate per-line columns (pure NegBin, no ratio scaling)
        targets = PER_TEAM_LINE_TARGETS.get(stat, {})
        stat_cols_added = 0
        lam_safe = np.where(valid_lambda, lambdas, 1)

        for direction in ["over", "under"]:
            lines = targets.get(direction, [])
            for target_line in lines:
                col_name = f"{stat}_{direction}_avg_{_line_to_col_suffix(target_line)}"

                target_floor = int(target_line)

                if direction == "over":
                    fair_prob = 1 - overdispersed_cdf(target_floor, lam_safe, stat, dispersion=d)
                else:
                    fair_prob = overdispersed_cdf(target_floor, lam_safe, stat, dispersion=d)

                # Clamp to [0.02, 0.98]
                fair_prob = np.clip(fair_prob, 0.02, 0.98)

                # Convert to decimal odds with vig
                estimated_odds = 1.0 / (fair_prob * (1 + VIG))
                estimated_odds = np.where(valid_lambda, estimated_odds, np.nan)

                if col_name in df.columns:
                    mask = df[col_name].isna()
                    if mask.any():
                        df.loc[mask, col_name] = pd.Series(
                            estimated_odds, index=df.index
                        )[mask]
                else:
                    df[col_name] = estimated_odds
                stat_cols_added += 1

        if stat_cols_added > 0:
            logger.info(
                f"  {stat}: {stat_cols_added} per-team line columns generated "
                f"(pure NegBin, d={d:.2f})"
            )
            total_cols_added += stat_cols_added

    logger.info(f"Per-team line odds: {total_cols_added} columns added to {len(df)} rows")

    if had_sort and "date" in df.columns:
        df = df.set_index(original_index).sort_index()

    return df
