"""NegBin baseline probability for niche market edge estimation.

Replaces the flat 0.50 baseline with a per-match NegBin probability
computed from expected_total_{stat}. The NegBin knows only the expected
count + overdispersion. The ML model adds match context (referee, form,
ELO, weather). The gap = genuine ML skill.

H2H markets (home_win, away_win, over25, under25, btts) are unaffected —
they use real bookmaker odds for edge.
"""

from typing import Optional, Tuple

import numpy as np

from src.odds.count_distribution import negbin_over_probability
from src.utils.line_plausibility import parse_market_line

# Mapping from base stat → expected_total column in features
EXPECTED_TOTAL_COLUMNS = {
    "fouls": "expected_total_fouls",
    "cards": "expected_total_cards",
    "shots": "expected_total_shots",
    "corners": "expected_total_corners",
}

# Default lines for base markets (no line suffix)
BASE_MARKET_LINES = {
    "fouls": 24.5,
    "cards": 4.5,
    "shots": 24.5,
    "corners": 9.5,
}


def resolve_negbin_params(
    market: str,
) -> Optional[Tuple[str, float, str, str]]:
    """Map market name to NegBin computation parameters.

    Args:
        market: Market name, e.g. 'fouls_over_245', 'corners', 'cards_under_35'.

    Returns:
        Tuple of (stat, line, direction, expected_total_col) for niche markets,
        or None for H2H / non-niche markets.
    """
    # Try line variant first (e.g. corners_over_85 → corners, 8.5, over)
    parsed = parse_market_line(market)
    if parsed is not None:
        stat, line, direction = parsed
        col = EXPECTED_TOTAL_COLUMNS.get(stat)
        if col is not None:
            return stat, line, direction, col
        return None

    # Base niche market (e.g. "corners", "fouls")
    if market in BASE_MARKET_LINES:
        stat = market
        line = BASE_MARKET_LINES[stat]
        col = EXPECTED_TOTAL_COLUMNS[stat]
        return stat, line, "over", col

    return None


def compute_negbin_baseline(
    market: str,
    expected_total: np.ndarray,
) -> np.ndarray:
    """Compute per-match NegBin probability for a niche market.

    Args:
        market: Market name (e.g. 'fouls_over_245', 'corners_under_95').
        expected_total: Array of expected total stat values per match.
            NaN entries produce NaN output.

    Returns:
        Array of NegBin probabilities, clipped to [0.02, 0.98].
        NaN where expected_total is NaN.

    Raises:
        ValueError: If market is not a niche market (resolve_negbin_params returns None).
    """
    params = resolve_negbin_params(market)
    if params is None:
        raise ValueError(f"Not a niche market: {market}")

    stat, line, direction, _ = params
    expected_total = np.asarray(expected_total, dtype=float)

    # Compute P(X > line) via NegBin
    over_prob = negbin_over_probability(expected_total, line, stat)

    if direction == "under":
        prob = 1.0 - over_prob
    else:
        prob = over_prob

    # Preserve NaN from input
    nan_mask = np.isnan(expected_total)
    prob = np.where(nan_mask, np.nan, prob)

    # Clip valid values
    return np.clip(prob, 0.02, 0.98)
