"""League-aware prior adjustment for niche market calibration.

Shifts model probabilities toward league-specific base rates in logit space.
This corrects overconfidence caused by pooling leagues with vastly different
base rates (e.g., cards U3.5 ranges from 22.7% in Portuguese Liga to 63.5%
in Eredivisie). Only applied to high-variance markets (cards, fouls, shots).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Markets where league heterogeneity causes miscalibration.
# Each entry: (target_column, line, direction)
# "under" means P(total < line), "over" means P(total > line)
MARKET_TARGET_MAP: Dict[str, Tuple[str, float, str]] = {
    # Cards UNDER
    "cards_under_15": ("total_cards", 1.5, "under"),
    "cards_under_25": ("total_cards", 2.5, "under"),
    "cards_under_35": ("total_cards", 3.5, "under"),
    "cards_under_45": ("total_cards", 4.5, "under"),
    "cards_under_55": ("total_cards", 5.5, "under"),
    "cards_under_65": ("total_cards", 6.5, "under"),
    # Cards OVER
    "cards_over_15": ("total_cards", 1.5, "over"),
    "cards_over_25": ("total_cards", 2.5, "over"),
    "cards_over_35": ("total_cards", 3.5, "over"),
    "cards_over_45": ("total_cards", 4.5, "over"),
    "cards_over_55": ("total_cards", 5.5, "over"),
    "cards_over_65": ("total_cards", 6.5, "over"),
    "cards": ("total_cards", 4.5, "over"),
    # Fouls UNDER
    "fouls_under_235": ("total_fouls", 23.5, "under"),
    "fouls_under_245": ("total_fouls", 24.5, "under"),
    "fouls_under_255": ("total_fouls", 25.5, "under"),
    "fouls_under_265": ("total_fouls", 26.5, "under"),
    "fouls_under_275": ("total_fouls", 27.5, "under"),
    # Fouls OVER
    "fouls_over_225": ("total_fouls", 22.5, "over"),
    "fouls_over_235": ("total_fouls", 23.5, "over"),
    "fouls_over_245": ("total_fouls", 24.5, "over"),
    "fouls_over_255": ("total_fouls", 25.5, "over"),
    "fouls_over_265": ("total_fouls", 26.5, "over"),
    "fouls": ("total_fouls", 24.5, "over"),
    # Shots UNDER
    "shots_under_255": ("total_shots", 25.5, "under"),
    "shots_under_265": ("total_shots", 26.5, "under"),
    "shots_under_275": ("total_shots", 27.5, "under"),
    "shots_under_285": ("total_shots", 28.5, "under"),
    "shots_under_295": ("total_shots", 29.5, "under"),
    # Shots OVER
    "shots_over_255": ("total_shots", 25.5, "over"),
    "shots_over_265": ("total_shots", 26.5, "over"),
    "shots_over_275": ("total_shots", 27.5, "over"),
    "shots_over_285": ("total_shots", 28.5, "over"),
    "shots_over_295": ("total_shots", 29.5, "over"),
    "shots": ("total_shots", 24.5, "over"),
}

# Only adjust if max-min league spread exceeds this threshold
MIN_VARIANCE_PP = 0.15

# Clamp probabilities to avoid logit explosion
_EPS = 1e-6


def _logit(p: float) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class LeaguePriorAdjuster:
    """Computes and applies league-specific base rate adjustments."""

    _instance: Optional["LeaguePriorAdjuster"] = None

    def __init__(self, features_path: Optional[str] = None):
        if features_path is None:
            features_path = "data/03-features/features_all_5leagues_with_odds.parquet"
        self._league_rates: Dict[str, Dict[str, float]] = {}  # market_key -> {league: rate}
        self._overall_rates: Dict[str, float] = {}  # market_key -> overall rate
        self._high_variance: Dict[str, bool] = {}  # market_key -> should adjust
        self._load(features_path)

    def _load(self, path: str) -> None:
        """Load features and compute per-league base rates."""
        try:
            df = pd.read_parquet(path, columns=["league", "total_cards", "total_fouls", "total_shots", "total_corners"])
        except Exception as e:
            logger.warning("LeaguePriorAdjuster: cannot load features (%s), adjustments disabled", e)
            return

        if "league" not in df.columns:
            logger.warning("LeaguePriorAdjuster: no league column, adjustments disabled")
            return

        # Pre-compute under-rates for each unique (column, line) pair
        seen_pairs: Dict[Tuple[str, float], Tuple[Dict[str, float], float]] = {}
        for market_key, (col, line, direction) in MARKET_TARGET_MAP.items():
            pair = (col, line)
            if pair not in seen_pairs:
                valid = df[df[col].notna()]
                if len(valid) == 0:
                    continue
                under_mask = (valid[col] < line).astype(float)
                league_rates = under_mask.groupby(valid["league"]).mean().to_dict()
                overall = float(under_mask.mean())
                seen_pairs[pair] = (league_rates, overall)

            if pair not in seen_pairs:
                continue

            league_under_rates, overall_under = seen_pairs[pair]

            if direction == "under":
                self._league_rates[market_key] = league_under_rates
                self._overall_rates[market_key] = overall_under
            else:  # over
                self._league_rates[market_key] = {k: 1.0 - v for k, v in league_under_rates.items()}
                self._overall_rates[market_key] = 1.0 - overall_under

            # High variance check
            rates = list(self._league_rates[market_key].values())
            if len(rates) >= 2:
                spread = max(rates) - min(rates)
                self._high_variance[market_key] = spread >= MIN_VARIANCE_PP
            else:
                self._high_variance[market_key] = False

        n_markets = sum(1 for v in self._high_variance.values() if v)
        logger.info("LeaguePriorAdjuster: loaded %d high-variance markets from %d total", n_markets, len(self._high_variance))

    def adjust(self, prob: float, market: str, league: str, strength: float = 0.5) -> float:
        """Adjust probability using league prior in logit space.

        Args:
            prob: Model's predicted probability
            market: Market key (e.g., "cards_under_35", "fouls_over_245")
            league: League identifier (e.g., "premier_league", "ligue_1")
            strength: Blending strength (0=no adjustment, 1=full league prior shift)

        Returns:
            Adjusted probability, or original if market/league not applicable.
        """
        if market not in self._high_variance or not self._high_variance[market]:
            return prob

        league_rates = self._league_rates.get(market, {})
        if league not in league_rates:
            return prob

        overall = self._overall_rates[market]
        league_rate = league_rates[league]

        # Avoid degenerate base rates
        if overall <= _EPS or overall >= 1 - _EPS:
            return prob
        if league_rate <= _EPS or league_rate >= 1 - _EPS:
            return prob

        model_logit = _logit(prob)
        league_logit = _logit(league_rate)
        overall_logit = _logit(overall)

        adjusted_logit = model_logit + strength * (league_logit - overall_logit)
        return _sigmoid(adjusted_logit)

    @classmethod
    def get_instance(cls, features_path: Optional[str] = None) -> "LeaguePriorAdjuster":
        if cls._instance is None:
            cls._instance = cls(features_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None


def adjust_for_league(prob: float, market: str, league: str, strength: float = 0.5) -> float:
    """Module-level convenience function for league prior adjustment."""
    return LeaguePriorAdjuster.get_instance().adjust(prob, market, league, strength)
