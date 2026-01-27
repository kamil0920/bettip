"""
CLV (Closing Line Value) Diagnostic Feature Engineering.

Creates features based on historical CLV patterns to identify
teams that are consistently mispriced by the market.

The closing line is considered the most efficient estimate of true probability,
so teams with positive historical CLV (consistently better than closing odds)
represent sustainable value.

CRITICAL: All features use HISTORICAL data only - no future leakage.
We use shift(1) on cumulative CLV metrics to ensure we only see past data.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer


class CLVDiagnosticEngineer(BaseFeatureEngineer):
    """
    Creates CLV diagnostic features from historical betting performance.

    CLV (Closing Line Value) measures edge vs closing line:
    - Positive CLV = Bet was placed at better odds than close
    - Negative CLV = Bet was placed at worse odds than close

    Teams with consistently positive CLV may be systematically mispriced.

    Features created:
    - Team's historical average CLV (rolling window)
    - CLV trend (improving or declining mispricing)
    - CLV consistency (variance of historical CLV)
    - CLV by match type (home/away specific patterns)
    """

    def __init__(
        self,
        lookback_matches: int = 20,
        min_matches: int = 5,
        ema_span: int = 10,
    ):
        """
        Initialize CLV diagnostic engineer.

        Args:
            lookback_matches: Number of historical matches to consider
            min_matches: Minimum matches needed for reliable CLV estimate
            ema_span: Span for exponential moving average of CLV
        """
        self.lookback_matches = lookback_matches
        self.min_matches = min_matches
        self.ema_span = ema_span

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create CLV diagnostic features.

        Args:
            data: Dict with 'matches' DataFrame containing:
                - fixture_id, date, home_team_id, away_team_id
                - Opening and closing odds columns
                - Match outcomes (home_win, away_win)

        Returns:
            DataFrame with CLV diagnostic features
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Calculate historical CLV for each match
        matches = self._calculate_match_clv(matches)

        # Create team-level CLV history features
        features_list = self._create_team_clv_features(matches)

        print(f"Created {len(features_list)} CLV diagnostic features")
        return pd.DataFrame(features_list)

    def _calculate_match_clv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CLV for each match based on opening vs closing odds.

        CLV = (closing_prob - opening_prob) / opening_prob
        Positive CLV = opening odds were better value than close
        """
        # Try to find opening and closing odds columns
        open_home = df.get('avg_home_open', df.get('b365_home_open'))
        close_home = df.get('avg_home_close', df.get('b365_home_close'))
        open_away = df.get('avg_away_open', df.get('b365_away_open'))
        close_away = df.get('avg_away_close', df.get('b365_away_close'))

        if open_home is None or close_home is None:
            # If no opening/closing data, we can't calculate CLV
            # Return empty CLV columns
            df['clv_home'] = 0.0
            df['clv_away'] = 0.0
            df['clv_draw'] = 0.0
            return df

        # Calculate implied probabilities
        open_prob_home = 1 / open_home
        close_prob_home = 1 / close_home
        open_prob_away = 1 / open_away
        close_prob_away = 1 / close_away

        # CLV = (close_prob - open_prob) / open_prob
        # Positive = closing probability higher = opening odds were value
        df['clv_home'] = (close_prob_home - open_prob_home) / (open_prob_home + 1e-10)
        df['clv_away'] = (close_prob_away - open_prob_away) / (open_prob_away + 1e-10)

        # Handle draw if available
        open_draw = df.get('avg_draw_open', df.get('b365_draw_open'))
        close_draw = df.get('avg_draw_close', df.get('b365_draw_close'))

        if open_draw is not None and close_draw is not None:
            open_prob_draw = 1 / open_draw
            close_prob_draw = 1 / close_draw
            df['clv_draw'] = (close_prob_draw - open_prob_draw) / (open_prob_draw + 1e-10)
        else:
            df['clv_draw'] = 0.0

        return df

    def _create_team_clv_features(self, df: pd.DataFrame) -> List[Dict]:
        """
        Create team-level CLV history features.

        For each match, calculates historical CLV metrics for both teams
        using only data from BEFORE the current match (no leakage).
        """
        all_teams = set(df['home_team_id'].unique()) | set(df['away_team_id'].unique())

        # Track CLV history for each team
        team_clv_history = {
            team_id: {
                'home_clv': [],  # CLV when playing at home
                'away_clv': [],  # CLV when playing away
                'dates': [],
            }
            for team_id in all_teams
        }

        features_list = []

        for idx, match in df.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get HISTORICAL CLV stats for both teams (before this match)
            home_features = self._get_team_clv_stats(
                team_clv_history[home_id],
                prefix='home',
                is_home_match=True,
            )

            away_features = self._get_team_clv_stats(
                team_clv_history[away_id],
                prefix='away',
                is_home_match=False,
            )

            # Combine features
            features = {'fixture_id': fixture_id}
            features.update(home_features)
            features.update(away_features)

            # Derived features
            features['clv_edge_diff'] = (
                home_features.get('home_avg_historical_clv', 0) -
                away_features.get('away_avg_historical_clv', 0)
            )

            # Both teams mispriced indicator
            features['both_positive_clv'] = int(
                home_features.get('home_positive_clv_rate', 0) > 0.5 and
                away_features.get('away_positive_clv_rate', 0) > 0.5
            )

            features_list.append(features)

            # Update CLV history AFTER recording features (no leakage)
            clv_home = match.get('clv_home', 0)
            clv_away = match.get('clv_away', 0)
            match_date = match.get('date')

            team_clv_history[home_id]['home_clv'].append(clv_home)
            team_clv_history[home_id]['dates'].append(match_date)

            team_clv_history[away_id]['away_clv'].append(clv_away)
            team_clv_history[away_id]['dates'].append(match_date)

        return features_list

    def _get_team_clv_stats(
        self,
        history: Dict,
        prefix: str,
        is_home_match: bool,
    ) -> Dict:
        """
        Calculate CLV statistics from team's historical data.

        Args:
            history: Dict with 'home_clv', 'away_clv', 'dates' lists
            prefix: Feature prefix ('home' or 'away')
            is_home_match: Whether current match is home or away

        Returns:
            Dict of CLV features
        """
        # Get relevant CLV history
        if is_home_match:
            clv_list = history['home_clv'][-self.lookback_matches:]
        else:
            clv_list = history['away_clv'][-self.lookback_matches:]

        # Also get overall CLV (both home and away combined)
        all_clv = history['home_clv'] + history['away_clv']
        all_clv = all_clv[-self.lookback_matches:]

        if len(clv_list) < self.min_matches:
            # Not enough data for reliable estimates
            return {
                f'{prefix}_avg_historical_clv': 0.0,
                f'{prefix}_clv_ema': 0.0,
                f'{prefix}_clv_trend': 0.0,
                f'{prefix}_clv_std': 0.0,
                f'{prefix}_positive_clv_rate': 0.5,
                f'{prefix}_clv_matches': len(clv_list),
            }

        clv_array = np.array(clv_list)

        # Average historical CLV
        avg_clv = np.mean(clv_array)

        # EMA of CLV (more weight on recent)
        clv_ema = self._calculate_ema(clv_array)

        # CLV trend (regression slope)
        clv_trend = self._calculate_trend(clv_array)

        # CLV consistency (lower std = more consistent)
        clv_std = np.std(clv_array)

        # Rate of positive CLV matches
        positive_rate = np.mean(clv_array > 0)

        return {
            f'{prefix}_avg_historical_clv': float(avg_clv),
            f'{prefix}_clv_ema': float(clv_ema),
            f'{prefix}_clv_trend': float(clv_trend),
            f'{prefix}_clv_std': float(clv_std),
            f'{prefix}_positive_clv_rate': float(positive_rate),
            f'{prefix}_clv_matches': len(clv_list),
        }

    def _calculate_ema(self, values: np.ndarray) -> float:
        """Calculate exponential moving average of CLV."""
        if len(values) == 0:
            return 0.0

        # Use pandas EMA calculation
        series = pd.Series(values)
        ema = series.ewm(span=self.ema_span, min_periods=1).mean()
        return ema.iloc[-1]

    def _calculate_trend(self, values: np.ndarray) -> float:
        """
        Calculate trend (slope) of CLV over time.

        Positive trend = CLV improving (team becoming more mispriced)
        Negative trend = CLV declining (market adjusting to team)
        """
        if len(values) < 3:
            return 0.0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)


class CLVOutcomeFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on historical relationship between CLV and outcomes.

    This engineer tracks whether positive CLV historically led to winning bets
    for each team, identifying teams where CLV is a reliable signal.
    """

    def __init__(
        self,
        lookback_matches: int = 30,
        min_matches: int = 10,
    ):
        self.lookback_matches = lookback_matches
        self.min_matches = min_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create CLV-outcome relationship features.
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Need match outcomes
        if 'home_win' not in matches.columns or 'away_win' not in matches.columns:
            print("No outcome columns found, skipping CLV-outcome features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Calculate CLV
        matches = self._calculate_match_clv(matches)

        # Track CLV vs outcome history
        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())

        team_clv_outcome = {
            team_id: {
                'positive_clv_wins': 0,
                'positive_clv_total': 0,
                'negative_clv_wins': 0,
                'negative_clv_total': 0,
            }
            for team_id in all_teams
        }

        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Calculate historical CLV-win correlation
            home_clv_reliability = self._get_clv_reliability(team_clv_outcome[home_id])
            away_clv_reliability = self._get_clv_reliability(team_clv_outcome[away_id])

            features = {
                'fixture_id': fixture_id,
                'home_clv_win_rate': home_clv_reliability['positive_clv_win_rate'],
                'away_clv_win_rate': away_clv_reliability['positive_clv_win_rate'],
                'home_clv_reliable': home_clv_reliability['is_reliable'],
                'away_clv_reliable': away_clv_reliability['is_reliable'],
            }

            features_list.append(features)

            # Update history AFTER recording features
            home_clv = match.get('clv_home', 0)
            away_clv = match.get('clv_away', 0)
            home_won = match.get('home_win', 0)
            away_won = match.get('away_win', 0)

            self._update_clv_outcome(team_clv_outcome[home_id], home_clv, home_won)
            self._update_clv_outcome(team_clv_outcome[away_id], away_clv, away_won)

        print(f"Created {len(features_list)} CLV-outcome features")
        return pd.DataFrame(features_list)

    def _calculate_match_clv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate CLV for each match."""
        open_home = df.get('avg_home_open', df.get('b365_home_open'))
        close_home = df.get('avg_home_close', df.get('b365_home_close'))
        open_away = df.get('avg_away_open', df.get('b365_away_open'))
        close_away = df.get('avg_away_close', df.get('b365_away_close'))

        if open_home is None or close_home is None:
            df['clv_home'] = 0.0
            df['clv_away'] = 0.0
            return df

        open_prob_home = 1 / open_home
        close_prob_home = 1 / close_home
        open_prob_away = 1 / open_away
        close_prob_away = 1 / close_away

        df['clv_home'] = (close_prob_home - open_prob_home) / (open_prob_home + 1e-10)
        df['clv_away'] = (close_prob_away - open_prob_away) / (open_prob_away + 1e-10)

        return df

    def _update_clv_outcome(self, history: Dict, clv: float, won: int):
        """Update CLV-outcome tracking."""
        if clv > 0:
            history['positive_clv_total'] += 1
            if won:
                history['positive_clv_wins'] += 1
        else:
            history['negative_clv_total'] += 1
            if won:
                history['negative_clv_wins'] += 1

    def _get_clv_reliability(self, history: Dict) -> Dict:
        """Calculate how reliable CLV is as a signal for this team."""
        pos_total = history['positive_clv_total']
        pos_wins = history['positive_clv_wins']
        neg_total = history['negative_clv_total']
        neg_wins = history['negative_clv_wins']

        # Win rate when CLV is positive
        pos_win_rate = pos_wins / pos_total if pos_total >= self.min_matches else 0.5

        # Win rate when CLV is negative
        neg_win_rate = neg_wins / neg_total if neg_total >= self.min_matches else 0.5

        # CLV is reliable if positive CLV has significantly higher win rate
        is_reliable = int(pos_win_rate > neg_win_rate + 0.1 and pos_total >= self.min_matches)

        return {
            'positive_clv_win_rate': pos_win_rate,
            'negative_clv_win_rate': neg_win_rate,
            'is_reliable': is_reliable,
        }
