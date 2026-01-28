"""
Fouls Totals Odds Loader

Provides estimated odds for fouls over/under markets.

IMPORTANT: No major odds provider (The Odds API, SportMonks, BetsAPI) offers
real bookmaker odds for fouls totals. This loader uses estimated odds based on:
- Team fouls EMA statistics
- Historical averages
- League-specific adjustments

Market typical odds (for 22.5 fouls line):
- Over 22.5 Fouls: 1.85-1.95
- Under 22.5 Fouls: 1.85-1.95

Usage:
    loader = FoulsOddsLoader()

    # Get estimated odds for historical/upcoming matches
    odds_df = loader.estimate_historical_odds(matches_df)

Note: Predictions using estimated odds should be marked as lower confidence
in the output since they don't reflect actual bookmaker assessments.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Common fouls lines (based on historical averages)
COMMON_FOUL_LINES = [20.5, 21.5, 22.5, 23.5, 24.5, 25.5]
DEFAULT_FOUL_LINE = 22.5

# Default odds - slightly tighter margin for estimated market
DEFAULT_OVER_ODDS = 1.90
DEFAULT_UNDER_ODDS = 1.90

# League-specific average fouls (used for adjustment)
LEAGUE_FOUL_AVERAGES = {
    "premier_league": 21.5,
    "la_liga": 24.5,
    "serie_a": 25.0,
    "bundesliga": 22.0,
    "ligue_1": 23.5,
}

ODDS_SOURCE_ESTIMATED = "estimated"


class FoulsOddsLoader:
    """
    Estimate fouls totals odds for betting markets.

    Since no major odds provider offers fouls markets, this loader uses
    statistical estimation based on:
    1. Historical fouls rate for the league
    2. Team fouls EMAs
    3. Expected total fouls features from ML pipeline

    Usage:
        loader = FoulsOddsLoader()

        # Get estimated odds for historical matches
        historical = loader.estimate_historical_odds(matches_df)

        # Get odds for specific line
        odds = loader.estimate_historical_odds(matches_df, target_line=23.5)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize loader.

        Args:
            cache_dir: Directory to cache computed odds
        """
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path("data/fouls_odds_cache")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def estimate_historical_odds(
        self,
        matches_df: pd.DataFrame,
        total_fouls_col: str = "total_fouls",
        target_line: float = DEFAULT_FOUL_LINE,
        use_market_efficiency: bool = True,
    ) -> pd.DataFrame:
        """
        Estimate fouls odds for matches based on statistical patterns.

        Uses team fouls statistics to generate realistic odds estimates.

        Args:
            matches_df: DataFrame with match data including total fouls
            total_fouls_col: Column name for total fouls outcome
            target_line: Fouls line for odds estimation
            use_market_efficiency: If True, adjust odds based on actual over rate

        Returns:
            DataFrame with estimated fouls odds
        """
        df = matches_df.copy()

        base_over = DEFAULT_OVER_ODDS
        base_under = DEFAULT_UNDER_ODDS

        if use_market_efficiency and total_fouls_col in df.columns:
            # Calculate over rate from historical data
            over_rate = (df[total_fouls_col] > target_line).mean()

            # Avoid edge cases
            if np.isnan(over_rate) or over_rate <= 0 or over_rate >= 1:
                logger.warning(
                    f"Invalid over_rate: {over_rate}, using default odds"
                )
                df["fouls_over_odds"] = base_over
                df["fouls_under_odds"] = base_under
            else:
                # Calculate implied probabilities with typical margin
                implied_over = 1 / base_over
                implied_under = 1 / base_under
                total_implied = implied_over + implied_under
                margin = total_implied - 1

                true_over = over_rate
                true_under = 1 - over_rate

                # Apply margin proportionally
                adj_over = 1 / (
                    true_over + margin * true_over / (true_over + true_under)
                )
                adj_under = 1 / (
                    true_under + margin * true_under / (true_over + true_under)
                )

                logger.info(f"Fouls over {target_line} rate: {over_rate:.1%}")
                logger.info(
                    f"Estimated odds: Over={adj_over:.2f}, Under={adj_under:.2f}"
                )

                df["fouls_over_odds"] = adj_over
                df["fouls_under_odds"] = adj_under
        else:
            df["fouls_over_odds"] = base_over
            df["fouls_under_odds"] = base_under

        # Adjust based on team fouls EMAs if available
        if "home_fouls_ema" in df.columns and "away_fouls_ema" in df.columns:
            # Teams with higher fouls EMAs -> higher over odds implied prob
            expected_fouls = df["home_fouls_ema"] + df["away_fouls_ema"]
            fouls_factor = expected_fouls / (target_line * 0.95)  # Slight adjustment
            fouls_factor = fouls_factor.clip(0.85, 1.15)

            df["fouls_over_odds"] = df["fouls_over_odds"] / fouls_factor
            df["fouls_under_odds"] = df["fouls_under_odds"] * fouls_factor
        elif (
            "home_avg_fouls" in df.columns and "away_avg_fouls" in df.columns
        ):
            # Alternative column names
            expected_fouls = df["home_avg_fouls"] + df["away_avg_fouls"]
            fouls_factor = expected_fouls / (target_line * 0.95)
            fouls_factor = fouls_factor.clip(0.85, 1.15)

            df["fouls_over_odds"] = df["fouls_over_odds"] / fouls_factor
            df["fouls_under_odds"] = df["fouls_under_odds"] * fouls_factor

        # Adjust based on expected total fouls feature if available
        if "expected_total_fouls" in df.columns:
            deviation = (df["expected_total_fouls"] - target_line) / target_line
            deviation = deviation.clip(-0.2, 0.2)

            df["fouls_over_odds"] = df["fouls_over_odds"] * (1 - deviation * 0.15)
            df["fouls_under_odds"] = df["fouls_under_odds"] * (1 + deviation * 0.15)

        # League-specific adjustment
        if "league" in df.columns:
            for league, avg_fouls in LEAGUE_FOUL_AVERAGES.items():
                mask = df["league"] == league
                if mask.any():
                    league_factor = avg_fouls / target_line
                    league_factor = np.clip(league_factor, 0.9, 1.1)

                    df.loc[mask, "fouls_over_odds"] = (
                        df.loc[mask, "fouls_over_odds"] / league_factor
                    )
                    df.loc[mask, "fouls_under_odds"] = (
                        df.loc[mask, "fouls_under_odds"] * league_factor
                    )

        # Ensure odds are within reasonable bounds
        df["fouls_over_odds"] = df["fouls_over_odds"].clip(1.50, 2.50)
        df["fouls_under_odds"] = df["fouls_under_odds"].clip(1.50, 2.50)

        df["fouls_line"] = target_line
        df["odds_source"] = ODDS_SOURCE_ESTIMATED

        return df

    def estimate_for_upcoming(
        self,
        fixtures_df: pd.DataFrame,
        target_line: float = DEFAULT_FOUL_LINE,
    ) -> pd.DataFrame:
        """
        Estimate fouls odds for upcoming fixtures.

        Uses team fouls EMAs and league averages to generate odds.

        Args:
            fixtures_df: DataFrame with upcoming fixtures and team stats
            target_line: Fouls line for odds estimation

        Returns:
            DataFrame with estimated fouls odds
        """
        df = fixtures_df.copy()

        # Start with base odds
        df["fouls_over_odds"] = DEFAULT_OVER_ODDS
        df["fouls_under_odds"] = DEFAULT_UNDER_ODDS

        # Adjust based on team fouls tendencies
        if "home_fouls_ema" in df.columns and "away_fouls_ema" in df.columns:
            expected_fouls = df["home_fouls_ema"] + df["away_fouls_ema"]
            fouls_factor = expected_fouls / target_line
            fouls_factor = fouls_factor.clip(0.85, 1.15)

            df["fouls_over_odds"] = df["fouls_over_odds"] / fouls_factor
            df["fouls_under_odds"] = df["fouls_under_odds"] * fouls_factor

        # League adjustment
        if "league" in df.columns:
            for league, avg_fouls in LEAGUE_FOUL_AVERAGES.items():
                mask = df["league"] == league
                if mask.any():
                    league_factor = avg_fouls / target_line
                    league_factor = np.clip(league_factor, 0.9, 1.1)

                    df.loc[mask, "fouls_over_odds"] = (
                        df.loc[mask, "fouls_over_odds"] / league_factor
                    )
                    df.loc[mask, "fouls_under_odds"] = (
                        df.loc[mask, "fouls_under_odds"] * league_factor
                    )

        # Ensure reasonable bounds
        df["fouls_over_odds"] = df["fouls_over_odds"].clip(1.50, 2.50)
        df["fouls_under_odds"] = df["fouls_under_odds"].clip(1.50, 2.50)

        df["fouls_line"] = target_line
        df["odds_source"] = ODDS_SOURCE_ESTIMATED

        return df

    def get_league_baseline(self, league: str) -> Dict:
        """
        Get baseline fouls statistics for a league.

        Args:
            league: League name

        Returns:
            Dict with average fouls and suggested lines
        """
        avg_fouls = LEAGUE_FOUL_AVERAGES.get(league, 22.5)

        # Suggest lines around the average
        suggested_lines = [
            round(avg_fouls - 1.5, 1),
            round(avg_fouls - 0.5, 1),
            round(avg_fouls + 0.5, 1),
            round(avg_fouls + 1.5, 1),
        ]

        return {
            "league": league,
            "average_fouls": avg_fouls,
            "suggested_lines": suggested_lines,
            "default_line": round(avg_fouls, 1),
        }


def add_fouls_odds_to_features(
    features_df: pd.DataFrame,
    total_fouls_col: str = "total_fouls",
    target_line: float = DEFAULT_FOUL_LINE,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Add fouls odds to existing features DataFrame.

    Args:
        features_df: Features DataFrame with total fouls
        total_fouls_col: Column name for total fouls
        target_line: Fouls line for odds
        output_path: Where to save result (optional)

    Returns:
        Features with fouls odds added
    """
    loader = FoulsOddsLoader()
    result = loader.estimate_historical_odds(
        features_df, total_fouls_col, target_line
    )

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = FoulsOddsLoader()

    # Show league baselines
    print("League Fouls Baselines:")
    for league in LEAGUE_FOUL_AVERAGES.keys():
        baseline = loader.get_league_baseline(league)
        print(f"  {league}: avg={baseline['average_fouls']}, lines={baseline['suggested_lines']}")

    # Demo with sample data
    print("\nEstimated Odds Example:")
    sample = pd.DataFrame(
        {
            "total_fouls": [20, 25, 22, 24, 19],
            "home_fouls_ema": [10.5, 12.0, 11.0, 11.5, 9.0],
            "away_fouls_ema": [11.0, 11.5, 10.5, 12.0, 10.0],
            "league": ["premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1"],
        }
    )
    result = loader.estimate_historical_odds(sample, target_line=22.5)
    print(
        result[
            ["total_fouls", "fouls_over_odds", "fouls_under_odds", "fouls_line", "league"]
        ]
    )
