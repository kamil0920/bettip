"""Market classification utilities for notification formatting.

Single source of truth for which markets have real bookmaker odds vs estimated odds.
"""

from __future__ import annotations

from typing import Any

# Markets with real bookmaker odds from football-data.co.uk / The Odds API.
# ROI, edge, and odds are meaningful ONLY for these markets.
REAL_ODDS_MARKETS: frozenset[str] = frozenset(
    {
        "home_win",
        "away_win",
        "over25",
        "under25",
        "btts",
        "home_win_h1",
        "away_win_h1",
    }
)


def is_real_odds_market(market: str) -> bool:
    """Check if a market has real bookmaker odds (not Poisson-estimated).

    Args:
        market: Market name (e.g. "home_win", "corners_over_95").

    Returns:
        True if ROI/edge/odds are trustworthy for this market.
    """
    return market in REAL_ODDS_MARKETS


def classify_recommendations(
    recs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split recommendations into real-odds and model-only lists.

    Uses the ``odds_verified`` column if present; falls back to market name.

    Args:
        recs: List of recommendation dicts (from CSV rows or DataFrames).

    Returns:
        (real_odds_recs, model_only_recs) — each sorted by their primary
        metric (edge desc for real-odds, probability desc for model-only).
    """
    real_odds: list[dict[str, Any]] = []
    model_only: list[dict[str, Any]] = []

    for rec in recs:
        # Primary signal: explicit odds_verified column
        odds_verified = rec.get("odds_verified")
        if odds_verified is not None:
            is_real = str(odds_verified).lower() in ("true", "1", "yes")
        else:
            # Fallback: market name lookup
            market = rec.get("market", rec.get("bet_type", ""))
            is_real = is_real_odds_market(str(market))

        if is_real:
            real_odds.append(rec)
        else:
            model_only.append(rec)

    # Sort: real-odds by edge descending, model-only by probability descending
    real_odds.sort(key=lambda r: float(r.get("edge", 0)), reverse=True)
    model_only.sort(key=lambda r: float(r.get("probability", 0)), reverse=True)

    return real_odds, model_only
