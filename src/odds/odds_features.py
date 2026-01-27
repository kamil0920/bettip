"""
Feature engineering for betting odds data.

Creates features from raw odds including:
- Implied probabilities
- Odds movement (opening to closing)
- Market consensus indicators
- Value indicators
"""
import logging
from typing import List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class OddsFeatureEngineer:
    """
    Transform raw odds into ML features.

    Features created:
    1. Implied probabilities (normalized)
    2. Odds movement (close - open)
    3. Market consensus (deviation from mean)
    4. Overround (market margin)
    5. Favorite indicators
    """

    def __init__(self, use_closing_odds: bool = True):
        """
        Initialize feature engineer.

        Args:
            use_closing_odds: If True, use closing odds as primary.
                             If False, use opening odds.
        """
        self.use_closing_odds = use_closing_odds

    def _odds_to_probability(self, odds: pd.Series) -> pd.Series:
        """Convert decimal odds to implied probability."""
        return 1 / odds

    def _normalize_probabilities(
        self,
        prob_home: pd.Series,
        prob_draw: pd.Series,
        prob_away: pd.Series
    ) -> tuple:
        """
        Normalize probabilities to sum to 1 (remove overround).

        Raw implied probabilities sum to >1 due to bookmaker margin.
        """
        total = prob_home + prob_draw + prob_away
        return (
            prob_home / total,
            prob_draw / total,
            prob_away / total
        )

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all odds-based features.

        Args:
            df: DataFrame with raw odds columns

        Returns:
            DataFrame with odds features added
        """
        df = df.copy()

        # Determine which odds columns to use as primary
        if self.use_closing_odds:
            home_col = self._find_column(df, ['avg_home_close', 'b365_home_close', 'avg_home_open'])
            draw_col = self._find_column(df, ['avg_draw_close', 'b365_draw_close', 'avg_draw_open'])
            away_col = self._find_column(df, ['avg_away_close', 'b365_away_close', 'avg_away_open'])
        else:
            home_col = self._find_column(df, ['avg_home_open', 'b365_home_open'])
            draw_col = self._find_column(df, ['avg_draw_open', 'b365_draw_open'])
            away_col = self._find_column(df, ['avg_away_open', 'b365_away_open'])

        if not all([home_col, draw_col, away_col]):
            logger.warning("Missing required odds columns, returning empty features")
            return df

        # 1. Raw implied probabilities
        prob_home_raw = self._odds_to_probability(df[home_col])
        prob_draw_raw = self._odds_to_probability(df[draw_col])
        prob_away_raw = self._odds_to_probability(df[away_col])

        # 2. Normalized probabilities (sum to 1)
        prob_home, prob_draw, prob_away = self._normalize_probabilities(
            prob_home_raw, prob_draw_raw, prob_away_raw
        )

        df['odds_home_prob'] = prob_home
        df['odds_draw_prob'] = prob_draw
        df['odds_away_prob'] = prob_away

        # 3. Overround (market margin indicator)
        df['odds_overround'] = prob_home_raw + prob_draw_raw + prob_away_raw

        # 4. Odds movement features (if both opening and closing available)
        df = self._create_movement_features(df)

        # 5. Favorite/underdog indicators
        df['odds_home_favorite'] = (prob_home > prob_away).astype(int)
        df['odds_prob_diff'] = prob_home - prob_away  # Positive = home favored
        df['odds_prob_max'] = df[['odds_home_prob', 'odds_draw_prob', 'odds_away_prob']].max(axis=1)

        # 6. Market certainty (how confident is the market)
        # Higher entropy = more uncertain
        df['odds_entropy'] = self._calculate_entropy(prob_home, prob_draw, prob_away)

        # 7. Upset potential (when underdog prob is meaningful)
        df['odds_upset_potential'] = df[['odds_home_prob', 'odds_away_prob']].min(axis=1)

        # 8. Draw likelihood relative to others
        df['odds_draw_relative'] = prob_draw / (prob_home + prob_away)

        # 9. Over/Under 2.5 goals features
        df = self._create_ou_features(df)

        # 10. Match statistics totals (for niche market targets)
        df = self._create_match_stats_features(df)

        logger.info(f"Created {len([c for c in df.columns if c.startswith('odds_')])} odds features")

        return df

    def _create_match_stats_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create match statistics totals for niche market targets.

        Creates:
        - total_corners, total_cards, total_fouls, total_shots
        - btts (Both Teams To Score)
        """
        # Total corners
        if 'home_corners' in df.columns and 'away_corners' in df.columns:
            df['total_corners'] = df['home_corners'] + df['away_corners']

        # Total cards (yellows + reds)
        if 'home_yellows' in df.columns and 'away_yellows' in df.columns:
            df['total_yellows'] = df['home_yellows'] + df['away_yellows']
            df['total_reds'] = df.get('home_reds', 0) + df.get('away_reds', 0)
            df['total_cards'] = df['total_yellows'] + df['total_reds']

        # Total fouls
        if 'home_fouls' in df.columns and 'away_fouls' in df.columns:
            df['total_fouls'] = df['home_fouls'] + df['away_fouls']

        # Total shots
        if 'home_shots' in df.columns and 'away_shots' in df.columns:
            df['total_shots'] = df['home_shots'] + df['away_shots']

        if 'home_shots_on_target' in df.columns and 'away_shots_on_target' in df.columns:
            df['total_shots_on_target'] = df['home_shots_on_target'] + df['away_shots_on_target']

        # BTTS (Both Teams To Score) - if goals data available
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            valid_goals = df['home_goals'].notna() & df['away_goals'].notna()
            df.loc[valid_goals, 'btts'] = (
                (df.loc[valid_goals, 'home_goals'] > 0) &
                (df.loc[valid_goals, 'away_goals'] > 0)
            ).astype(int)

        return df

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first available column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _create_movement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create odds movement features (opening → closing).

        Movement indicates where "smart money" is going.
        Research shows large late moves in betting markets are reliable outcome indicators.
        """
        # Check if we have both opening and closing odds
        has_opening = 'avg_home_open' in df.columns or 'b365_home_open' in df.columns
        has_closing = 'avg_home_close' in df.columns or 'b365_home_close' in df.columns

        if not (has_opening and has_closing):
            logger.info("Opening or closing odds not available, skipping movement features")
            return df

        # Use average odds preferably, fallback to Bet365
        open_home = df.get('avg_home_open', df.get('b365_home_open'))
        open_draw = df.get('avg_draw_open', df.get('b365_draw_open'))
        open_away = df.get('avg_away_open', df.get('b365_away_open'))

        close_home = df.get('avg_home_close', df.get('b365_home_close'))
        close_draw = df.get('avg_draw_close', df.get('b365_draw_close'))
        close_away = df.get('avg_away_close', df.get('b365_away_close'))

        if open_home is not None and close_home is not None:
            # Raw odds movement (positive = odds shortened = more money on this outcome)
            df['odds_move_home'] = open_home - close_home
            df['odds_move_draw'] = open_draw - close_draw
            df['odds_move_away'] = open_away - close_away

            # Probability movement (more intuitive)
            prob_open_home = 1 / open_home
            prob_close_home = 1 / close_home
            prob_open_away = 1 / open_away
            prob_close_away = 1 / close_away
            prob_open_draw = 1 / open_draw
            prob_close_draw = 1 / close_draw

            df['odds_prob_move_home'] = prob_close_home - prob_open_home
            df['odds_prob_move_away'] = prob_close_away - prob_open_away

            # Relative movement (% change in odds)
            df['odds_move_home_pct'] = (open_home - close_home) / open_home
            df['odds_move_away_pct'] = (open_away - close_away) / open_away

            # Steam move indicator (significant movement > 5%)
            df['odds_steam_home'] = (df['odds_move_home_pct'].abs() > 0.05).astype(int)
            df['odds_steam_away'] = (df['odds_move_away_pct'].abs() > 0.05).astype(int)

            # === ENHANCED LINE MOVEMENT FEATURES ===

            # 1. Total movement magnitude (market activity indicator)
            # Higher = more information came into market
            df['line_movement_magnitude'] = (
                df['odds_move_home_pct'].abs() +
                df['odds_move_away_pct'].abs()
            )

            # 2. Movement consistency check
            # Normal: home shortens → away drifts (or vice versa)
            # Unusual: both move same direction (draw getting action)
            home_direction = np.sign(df['odds_move_home'])  # +1 = shortened, -1 = drifted
            away_direction = np.sign(df['odds_move_away'])
            df['movement_consistent'] = (home_direction != away_direction).astype(int)

            # 3. Sharp money direction indicator
            # +1 = sharp money on home, -1 = sharp money on away, 0 = mixed
            df['sharp_money_direction'] = np.sign(
                df['odds_prob_move_home'] - df['odds_prob_move_away']
            )

            # 4. Overround change (market efficiency indicator)
            # Decreasing overround = market becoming more efficient/confident
            overround_open = prob_open_home + prob_open_draw + prob_open_away
            overround_close = prob_close_home + prob_close_draw + prob_close_away
            df['overround_change'] = overround_close - overround_open

            # 5. Tiered movement strength (more granular than binary steam)
            # 0 = minimal (<2%), 1 = small (2-5%), 2 = medium (5-10%), 3 = large (>10%)
            abs_move_home = df['odds_move_home_pct'].abs()
            df['movement_tier_home'] = pd.cut(
                abs_move_home,
                bins=[-np.inf, 0.02, 0.05, 0.10, np.inf],
                labels=[0, 1, 2, 3]
            ).astype(float)

            abs_move_away = df['odds_move_away_pct'].abs()
            df['movement_tier_away'] = pd.cut(
                abs_move_away,
                bins=[-np.inf, 0.02, 0.05, 0.10, np.inf],
                labels=[0, 1, 2, 3]
            ).astype(float)

            # 6. Big mover flag (>10% movement - strong sharp signal)
            df['big_mover_home'] = (abs_move_home > 0.10).astype(int)
            df['big_mover_away'] = (abs_move_away > 0.10).astype(int)

            # 7. Favorite drift indicator (favorite odds getting longer)
            # This often indicates sharp money against the favorite
            is_home_favorite = close_home < close_away
            df['favorite_drifting'] = (
                (is_home_favorite & (df['odds_move_home'] < 0)) |  # Home fav drifting
                (~is_home_favorite & (df['odds_move_away'] < 0))   # Away fav drifting
            ).astype(int)

            # 8. Draw movement (often overlooked but can be predictive)
            df['odds_move_draw_pct'] = (open_draw - close_draw) / open_draw
            df['draw_steam'] = (df['odds_move_draw_pct'].abs() > 0.05).astype(int)

            # 9. Combined sharp indicator
            # High confidence sharp signal when multiple indicators align
            df['sharp_confidence'] = (
                (df['line_movement_magnitude'] > 0.08).astype(int) +  # Significant total move
                df['movement_consistent'] +  # Normal pattern
                (df['big_mover_home'] | df['big_mover_away']).astype(int)  # Large single move
            )

            # === VELOCITY FEATURES (movement / time) ===
            # These require time delta between opening and closing odds
            df = self._create_velocity_features(df)

        return df

    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create odds velocity features (movement rate over time).

        Velocity = movement percentage / time elapsed (hours)

        Fast velocity indicates sharp money entering close to kickoff,
        which is often more predictive than early movements.
        """
        # Check for time columns
        time_col = None
        open_time_col = None
        close_time_col = None

        # Try to find time delta information
        for col in ['odds_time_delta_hours', 'time_to_kickoff_hours', 'hours_before_kickoff']:
            if col in df.columns:
                time_col = col
                break

        # Also check for separate opening/closing timestamps
        if 'odds_open_time' in df.columns and 'odds_close_time' in df.columns:
            open_time_col = 'odds_open_time'
            close_time_col = 'odds_close_time'

        # Calculate time delta if we have timestamps
        time_delta_hours = None

        if open_time_col and close_time_col:
            try:
                open_time = pd.to_datetime(df[open_time_col])
                close_time = pd.to_datetime(df[close_time_col])
                time_delta_hours = (close_time - open_time).dt.total_seconds() / 3600
            except Exception:
                pass

        elif time_col:
            time_delta_hours = df[time_col]

        # Default assumption: 24-48 hours between opening and closing
        # Most bookmakers open markets 2-3 days before kickoff
        if time_delta_hours is None:
            # Use default 36 hours if no time data available
            default_hours = 36.0
            time_delta_hours = pd.Series([default_hours] * len(df), index=df.index)
            logger.debug("Using default 36h time delta for velocity calculation")

        # Avoid division by zero
        time_delta_hours = pd.Series(time_delta_hours).clip(lower=1.0)

        # Calculate velocity features
        if 'odds_move_home_pct' in df.columns:
            # Velocity = movement% / hours
            df['odds_velocity_home'] = df['odds_move_home_pct'] / time_delta_hours
            df['odds_velocity_away'] = df['odds_move_away_pct'] / time_delta_hours

            # Absolute velocity (magnitude of change rate)
            df['odds_velocity_abs_home'] = df['odds_velocity_home'].abs()
            df['odds_velocity_abs_away'] = df['odds_velocity_away'].abs()

            # High velocity indicator (rapid movement - usually sharp money)
            # Threshold: > 0.2% per hour movement rate
            velocity_threshold = 0.002  # 0.2% per hour
            df['high_velocity_home'] = (df['odds_velocity_abs_home'] > velocity_threshold).astype(int)
            df['high_velocity_away'] = (df['odds_velocity_abs_away'] > velocity_threshold).astype(int)

            # Velocity direction (which side is moving faster)
            # +1 = home moving faster toward shorter odds
            # -1 = away moving faster toward shorter odds
            df['velocity_direction'] = np.sign(
                df['odds_velocity_home'] - df['odds_velocity_away']
            )

        # Late surge detection
        # This requires early vs late movement data (e.g., 24h out vs 1h out)
        # If available, compare movement rates in different time windows
        df = self._create_late_surge_features(df)

        return df

    def _create_late_surge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect late surges in odds movement (movement accelerating near kickoff).

        Late movement (within 2-4 hours of kickoff) is considered more predictive
        as it reflects information from team news, weather, and sharp money.

        This requires multiple odds snapshots or explicit early/late movement columns.
        """
        # Check for multi-snapshot odds data
        has_late_data = any(
            col in df.columns for col in [
                'odds_move_home_late', 'odds_home_1h', 'odds_home_4h',
                'late_move_home_pct', 'early_move_home_pct'
            ]
        )

        if not has_late_data:
            # Estimate late surge from overall movement characteristics
            # Higher absolute movement with high velocity suggests late surge
            if 'odds_move_home_pct' in df.columns and 'odds_velocity_abs_home' in df.columns:
                # Late surge proxy: large movement AND high velocity
                # This approximates the pattern of concentrated late betting activity
                large_move_home = df['odds_move_home_pct'].abs() > 0.05
                high_velocity_home = df['odds_velocity_abs_home'] > 0.002

                df['late_surge_proxy_home'] = (large_move_home & high_velocity_home).astype(int)

                large_move_away = df['odds_move_away_pct'].abs() > 0.05
                high_velocity_away = df['odds_velocity_abs_away'] > 0.002

                df['late_surge_proxy_away'] = (large_move_away & high_velocity_away).astype(int)

                # Combined late activity indicator
                df['late_market_activity'] = (
                    df['late_surge_proxy_home'] | df['late_surge_proxy_away']
                ).astype(int)

        else:
            # Use actual early vs late movement data if available
            if 'early_move_home_pct' in df.columns and 'late_move_home_pct' in df.columns:
                # Late surge = late movement > early movement
                df['late_surge_home'] = (
                    df['late_move_home_pct'].abs() > df['early_move_home_pct'].abs()
                ).astype(int)

                df['late_surge_away'] = (
                    df['late_move_away_pct'].abs() > df['early_move_away_pct'].abs()
                ).astype(int)

                # Late movement dominance ratio
                df['late_dominance_home'] = (
                    df['late_move_home_pct'].abs() /
                    (df['early_move_home_pct'].abs() + 0.001)
                )

        return df

    def _create_ou_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Over/Under 2.5 goals features."""

        ou_col = self._find_column(df, ['avg_over25_close', 'b365_over25_close', 'avg_over25', 'b365_over25'])

        if ou_col is None:
            return df

        under_col = ou_col.replace('over', 'under')
        if under_col not in df.columns:
            return df

        over_odds = df[ou_col]
        under_odds = df[under_col]

        # Implied probability of over 2.5 goals
        prob_over = 1 / over_odds
        prob_under = 1 / under_odds
        total = prob_over + prob_under

        df['odds_over25_prob'] = prob_over / total
        df['odds_under25_prob'] = prob_under / total

        # Expected goals indicator (higher over25 prob = more goals expected)
        df['odds_goals_expectation'] = df['odds_over25_prob']

        return df

    def _calculate_entropy(
        self,
        prob_home: pd.Series,
        prob_draw: pd.Series,
        prob_away: pd.Series
    ) -> pd.Series:
        """
        Calculate Shannon entropy of probability distribution.

        Higher entropy = more uncertainty in outcome.
        Max entropy = log(3) ≈ 1.1 when all outcomes equally likely.
        """
        # Avoid log(0)
        eps = 1e-10
        probs = pd.concat([prob_home, prob_draw, prob_away], axis=1)
        probs = probs.clip(lower=eps)

        entropy = -np.sum(probs.values * np.log(probs.values), axis=1)
        return pd.Series(entropy, index=prob_home.index)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names created by this engineer."""
        return [
            # Probabilities
            'odds_home_prob',
            'odds_draw_prob',
            'odds_away_prob',
            # Market indicators
            'odds_overround',
            'odds_home_favorite',
            'odds_prob_diff',
            'odds_prob_max',
            'odds_entropy',
            'odds_upset_potential',
            'odds_draw_relative',
            # Movement (if available)
            'odds_move_home',
            'odds_move_draw',
            'odds_move_away',
            'odds_prob_move_home',
            'odds_prob_move_away',
            'odds_move_home_pct',
            'odds_move_away_pct',
            'odds_steam_home',
            'odds_steam_away',
            # Enhanced line movement features
            'line_movement_magnitude',
            'movement_consistent',
            'sharp_money_direction',
            'overround_change',
            'movement_tier_home',
            'movement_tier_away',
            'big_mover_home',
            'big_mover_away',
            'favorite_drifting',
            'odds_move_draw_pct',
            'draw_steam',
            'sharp_confidence',
            # Velocity features (movement / time)
            'odds_velocity_home',
            'odds_velocity_away',
            'odds_velocity_abs_home',
            'odds_velocity_abs_away',
            'high_velocity_home',
            'high_velocity_away',
            'velocity_direction',
            # Late surge features
            'late_surge_proxy_home',
            'late_surge_proxy_away',
            'late_market_activity',
            'late_surge_home',
            'late_surge_away',
            'late_dominance_home',
            # Over/Under
            'odds_over25_prob',
            'odds_under25_prob',
            'odds_goals_expectation',
        ]
