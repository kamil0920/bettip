"""
Forecastability Diagnostics for Betting Markets

Computes per-market forecastability scorecards using information-theoretic measures:
- Permutation Entropy (PE) on residuals: primary measure of exploitable structure
- Sample Entropy (SampEn) on continuous series: pattern recurrence
- ACF lag-1: linear autocorrelation in residuals
- Fano bound (Pi_max): theoretical accuracy ceiling

Usage:
    python experiments/forecastability_analysis.py \
        --data data/03-features/features_all_5leagues_with_odds.parquet

Source: Manokhin, "Mastering Modern Time Series Forecasting", Chapters 2-3.
"""

import argparse
import logging
from collections import defaultdict
from itertools import permutations
from math import factorial, log
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Markets and their target + odds columns for implied probability
H2H_MARKETS = {
    "home_win": {"target": "home_win", "odds_col": "avg_home_close"},
    "away_win": {"target": "away_win", "odds_col": "avg_away_close"},
    "over25": {"target": "over25", "odds_col": "avg_over25_close"},
    "under25": {"target": "under25", "odds_col": "avg_under25_close"},
    "btts": {"target": "btts", "odds_col": "btts_yes_odds"},
}

NICHE_MARKETS = {
    "fouls": {"target": "total_fouls", "line": 24.5},
    "shots": {"target": "total_shots", "line": 24.5},
    "corners": {"target": "total_corners", "line": 9.5},
    "cards": {"target": "total_cards", "line": 4.5},
}

MIN_SERIES_LENGTH = 100  # Minimum matches per team for PE order=3


def permutation_entropy(
    x: np.ndarray, order: int = 3, delay: int = 1
) -> float:
    """Normalized permutation entropy [0, 1].

    0 = fully deterministic, 1 = completely random.
    Manual implementation — no external dependencies.

    Args:
        x: 1D array of values (continuous or discrete).
        order: Embedding dimension (pattern length). order=3 gives 6 patterns.
        delay: Time delay between elements.

    Returns:
        Normalized PE in [0, 1]. Returns NaN if series too short.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_patterns = n - (order - 1) * delay

    if n_patterns < order:
        return np.nan

    # Count ordinal patterns
    pattern_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    for i in range(n_patterns):
        indices = [i + j * delay for j in range(order)]
        window = x[indices]
        # Rank the values (argsort of argsort gives ranks)
        pattern = tuple(np.argsort(np.argsort(window)))
        pattern_counts[pattern] += 1

    # Shannon entropy of pattern distribution
    total = sum(pattern_counts.values())
    probs = np.array([c / total for c in pattern_counts.values()])
    probs = probs[probs > 0]
    h = -np.sum(probs * np.log2(probs))

    # Normalize by maximum entropy (log2 of order!)
    h_max = log(factorial(order), 2)
    return h / h_max if h_max > 0 else 0.0


def sample_entropy(
    x: np.ndarray, m: int = 2, r: Optional[float] = None
) -> float:
    """Sample entropy for continuous time series.

    Measures pattern recurrence excluding self-matches.
    Low = predictable, high = random.

    WARNING: Degenerates on binary {0,1} sequences. Use only on continuous series
    (rolling accuracy, cumulative PnL, residuals).

    Args:
        x: 1D array of continuous values.
        m: Embedding dimension (template length).
        r: Tolerance threshold. Defaults to 0.2 * std(x).

    Returns:
        SampEn value (non-negative). Returns NaN if computation fails.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < m + 2:
        return np.nan

    if r is None:
        std = np.std(x)
        if std == 0:
            return 0.0  # Constant series
        r = 0.2 * std

    def _count_matches(template_len: int) -> int:
        """Count template matches using Chebyshev distance."""
        count = 0
        templates = np.array(
            [x[i : i + template_len] for i in range(n - template_len)]
        )
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count

    a = _count_matches(m + 1)
    b = _count_matches(m)

    if b == 0:
        return np.nan

    return -log(a / b) if a > 0 else np.nan


def acf_lag1(x: np.ndarray) -> float:
    """First-order autocorrelation.

    Significant ACF at lag-1 indicates linear exploitable structure.

    Args:
        x: 1D array of values.

    Returns:
        ACF at lag 1. Returns NaN if series too short or constant.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    mean = np.mean(x)
    var = np.var(x)
    if var == 0:
        return 0.0
    cov = np.mean((x[:-1] - mean) * (x[1:] - mean))
    return cov / var


def fano_bound(pe: float) -> float:
    """Fano's inequality bound on maximum prediction accuracy.

    For binary classification: Pi_max = 1 - PE.
    This is the theoretical ceiling — no model can exceed this.

    Args:
        pe: Permutation entropy [0, 1].

    Returns:
        Maximum achievable accuracy [0, 1]. Returns NaN if PE is NaN.
    """
    if np.isnan(pe):
        return np.nan
    return max(0.0, 1.0 - pe)


def compute_implied_probability(
    odds: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Convert bookmaker odds to implied probability.

    Args:
        odds: Array of decimal odds.
        normalize: If True, normalize to remove overround (not used for single-market).

    Returns:
        Array of implied probabilities [0, 1].
    """
    odds = np.asarray(odds, dtype=float)
    implied = np.where(odds > 0, 1.0 / odds, np.nan)
    return np.clip(implied, 0.01, 0.99)


def forecastability_scorecard(
    target_series: np.ndarray,
    residual_series: np.ndarray,
    market_name: str,
) -> Dict:
    """Compute full forecastability scorecard for one team-market series.

    Args:
        target_series: Binary outcomes {0, 1} chronologically ordered.
        residual_series: Continuous residuals (actual - implied_prob).
        market_name: Name for logging.

    Returns:
        Dict with PE, SampEn, ACF1, Pi_max metrics.
    """
    # PE on residuals (primary — measures exploitable structure beyond market pricing)
    pe_residual = permutation_entropy(residual_series, order=3, delay=1)

    # PE on raw targets (secondary — expected to be high for binary)
    pe_target = permutation_entropy(target_series, order=3, delay=1)

    # SampEn on residuals (continuous series, appropriate for SampEn)
    sampen_residual = sample_entropy(residual_series, m=2)

    # ACF lag-1 on residuals
    acf1 = acf_lag1(residual_series)

    # Fano bound from residual PE
    pi_max = fano_bound(pe_residual)

    return {
        "market": market_name,
        "pe_residual": pe_residual,
        "pe_target": pe_target,
        "sampen_residual": sampen_residual,
        "acf1_residual": acf1,
        "pi_max": pi_max,
        "series_length": len(target_series),
    }


def build_team_series(
    df: pd.DataFrame,
    target_col: str,
    odds_col: Optional[str] = None,
    target_line: Optional[float] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build chronological per-team series from features DataFrame.

    Args:
        df: Features DataFrame with home_team, away_team, fixture_date, target, odds.
        target_col: Column name for the target variable.
        odds_col: Column name for bookmaker odds (for implied prob computation).
        target_line: For niche markets, the line to convert continuous → binary.

    Returns:
        Dict mapping team_name -> {"targets": array, "residuals": array}
    """
    # Sort chronologically
    date_col = next((c for c in ["fixture_date", "date"] if c in df.columns), None)
    if date_col is None:
        logger.warning("No date column found")
        return {}

    df_sorted = df.sort_values(date_col).copy()

    # Compute binary target (derive if needed)
    if target_col not in df_sorted.columns:
        # Derive targets that don't exist as columns
        if target_col == "over25" and "total_goals" in df_sorted.columns:
            binary_target = (df_sorted["total_goals"] > 2.5).astype(float)
        elif target_col == "under25" and "total_goals" in df_sorted.columns:
            binary_target = (df_sorted["total_goals"] <= 2.5).astype(float)
        elif target_col == "btts":
            if "goals_home" in df_sorted.columns and "goals_away" in df_sorted.columns:
                binary_target = ((df_sorted["goals_home"] > 0) & (df_sorted["goals_away"] > 0)).astype(float)
            elif "total_goals" in df_sorted.columns and "goal_difference" in df_sorted.columns:
                home_g = (df_sorted["total_goals"] + df_sorted["goal_difference"]) / 2
                away_g = (df_sorted["total_goals"] - df_sorted["goal_difference"]) / 2
                binary_target = ((home_g > 0) & (away_g > 0)).astype(float)
            else:
                logger.warning(f"Cannot derive '{target_col}' — missing source columns")
                return {}
        else:
            logger.warning(f"Target column '{target_col}' not found and cannot be derived")
            return {}
    elif target_line is not None:
        binary_target = (df_sorted[target_col] > target_line).astype(float)
    else:
        binary_target = df_sorted[target_col].astype(float)

    # Compute implied probability
    has_odds = odds_col and odds_col in df_sorted.columns
    if has_odds:
        implied_prob = compute_implied_probability(df_sorted[odds_col].values)
    else:
        # Use base rate as fallback (no residual analysis possible)
        implied_prob = np.full(len(df_sorted), binary_target.mean())

    residuals = binary_target.values - implied_prob

    # Build per-team series (each team appears as both home and away)
    team_series: Dict[str, Dict[str, List]] = defaultdict(
        lambda: {"targets": [], "residuals": []}
    )

    home_col = next(
        (c for c in ["home_team", "home_team_name"] if c in df_sorted.columns), None
    )
    away_col = next(
        (c for c in ["away_team", "away_team_name"] if c in df_sorted.columns), None
    )

    if not home_col or not away_col:
        logger.warning("No home_team/away_team columns found")
        return {}

    for i in range(len(df_sorted)):
        if np.isnan(binary_target.iloc[i]) or np.isnan(residuals[i]):
            continue
        home = df_sorted[home_col].iloc[i]
        away = df_sorted[away_col].iloc[i]

        team_series[home]["targets"].append(binary_target.iloc[i])
        team_series[home]["residuals"].append(residuals[i])
        team_series[away]["targets"].append(binary_target.iloc[i])
        team_series[away]["residuals"].append(residuals[i])

    # Convert to arrays and filter by minimum length
    result = {}
    for team, series in team_series.items():
        if len(series["targets"]) >= MIN_SERIES_LENGTH:
            result[team] = {
                "targets": np.array(series["targets"]),
                "residuals": np.array(series["residuals"]),
            }

    return result


def analyze_market(
    df: pd.DataFrame,
    market_name: str,
    target_col: str,
    odds_col: Optional[str] = None,
    target_line: Optional[float] = None,
) -> Dict:
    """Run forecastability analysis for a single market.

    Args:
        df: Features DataFrame.
        market_name: Market name for output.
        target_col: Target column name.
        odds_col: Odds column for implied prob.
        target_line: Line for niche market binary conversion.

    Returns:
        Dict with aggregated scorecard metrics.
    """
    team_series = build_team_series(df, target_col, odds_col, target_line)

    if not team_series:
        logger.warning(f"No qualifying teams for {market_name}")
        return {"market": market_name, "n_teams": 0, "status": "no_data"}

    scorecards = []
    for team, series in team_series.items():
        sc = forecastability_scorecard(
            series["targets"], series["residuals"], f"{market_name}_{team}"
        )
        scorecards.append(sc)

    # Aggregate across teams
    def _safe_nanmean(vals):
        clean = [v for v in vals if not np.isnan(v)]
        return np.mean(clean) if clean else np.nan

    def _safe_nanmedian(vals):
        clean = [v for v in vals if not np.isnan(v)]
        return np.median(clean) if clean else np.nan

    has_real_odds = odds_col and odds_col in df.columns and df[odds_col].notna().mean() > 0.3

    return {
        "market": market_name,
        "n_teams": len(team_series),
        "mean_pe_residual": _safe_nanmean([s["pe_residual"] for s in scorecards]),
        "median_pe_residual": _safe_nanmedian([s["pe_residual"] for s in scorecards]),
        "mean_pe_target": _safe_nanmean([s["pe_target"] for s in scorecards]),
        "mean_sampen_residual": _safe_nanmean([s["sampen_residual"] for s in scorecards]),
        "mean_acf1_residual": _safe_nanmean([s["acf1_residual"] for s in scorecards]),
        "mean_pi_max": _safe_nanmean([s["pi_max"] for s in scorecards]),
        "has_real_odds": has_real_odds,
        "status": "ok",
    }


def run_analysis(data_path: str) -> pd.DataFrame:
    """Run full forecastability analysis across all markets.

    Args:
        data_path: Path to features parquet file.

    Returns:
        DataFrame with per-market forecastability scorecards.
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    results = []

    # H2H markets (have bookmaker odds for residual computation)
    for name, config in H2H_MARKETS.items():
        logger.info(f"Analyzing {name}...")
        result = analyze_market(
            df, name, config["target"], config["odds_col"]
        )
        results.append(result)

    # Niche markets (no real odds — uses base-rate residuals)
    for name, config in NICHE_MARKETS.items():
        logger.info(f"Analyzing {name} (niche, base-rate residuals)...")
        result = analyze_market(
            df, name, config["target"], target_line=config["line"]
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    # Sort by forecastability (lower PE = more forecastable)
    if "mean_pe_residual" in results_df.columns:
        results_df = results_df.sort_values("mean_pe_residual", ascending=True)

    return results_df


def print_scorecard(results_df: pd.DataFrame) -> None:
    """Print formatted forecastability scorecard."""
    print("\n" + "=" * 90)
    print("FORECASTABILITY SCORECARD")
    print("=" * 90)
    print(
        f"{'Market':<15} {'Teams':>5} {'PE(res)':>8} {'PE(tgt)':>8} "
        f"{'SampEn':>8} {'ACF1':>8} {'Pi_max':>8} {'Odds':>5}"
    )
    print("-" * 90)

    for _, row in results_df.iterrows():
        if row.get("status") != "ok":
            print(f"{row['market']:<15} {'N/A':>5} — {row.get('status', 'error')}")
            continue

        pe_r = f"{row['mean_pe_residual']:.4f}" if not np.isnan(row['mean_pe_residual']) else "N/A"
        pe_t = f"{row['mean_pe_target']:.4f}" if not np.isnan(row['mean_pe_target']) else "N/A"
        se = f"{row['mean_sampen_residual']:.4f}" if not np.isnan(row['mean_sampen_residual']) else "N/A"
        acf = f"{row['mean_acf1_residual']:+.4f}" if not np.isnan(row['mean_acf1_residual']) else "N/A"
        pi = f"{row['mean_pi_max']:.4f}" if not np.isnan(row['mean_pi_max']) else "N/A"
        odds = "real" if row.get("has_real_odds") else "base"

        print(
            f"{row['market']:<15} {row['n_teams']:>5} {pe_r:>8} {pe_t:>8} "
            f"{se:>8} {acf:>8} {pi:>8} {odds:>5}"
        )

    print("-" * 90)
    print("PE(res) = Permutation Entropy on residuals (lower = more forecastable)")
    print("PE(tgt) = Permutation Entropy on raw target (high for all binary markets)")
    print("SampEn  = Sample Entropy on residuals (lower = more pattern recurrence)")
    print("ACF1    = Autocorrelation at lag 1 (significant = exploitable structure)")
    print("Pi_max  = Fano bound: theoretical accuracy ceiling = 1 - PE(res)")
    print("Odds    = 'real' if bookmaker odds available, 'base' if using base-rate fallback")
    print("=" * 90)


def _set_min_series_length(value: int) -> None:
    """Update the global MIN_SERIES_LENGTH."""
    global MIN_SERIES_LENGTH
    MIN_SERIES_LENGTH = value


def main():
    parser = argparse.ArgumentParser(description="Forecastability analysis for betting markets")
    parser.add_argument(
        "--data",
        default="data/03-features/features_all_5leagues_with_odds.parquet",
        help="Path to features parquet",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path for results",
    )
    parser.add_argument(
        "--min-series",
        type=int,
        default=MIN_SERIES_LENGTH,
        help=f"Minimum matches per team (default: {MIN_SERIES_LENGTH})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    _set_min_series_length(args.min_series)

    results_df = run_analysis(args.data)
    print_scorecard(results_df)

    if args.output:
        results_df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
