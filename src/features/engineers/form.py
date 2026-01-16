"""Feature engineering - Team form and streak features."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer


class TeamFormFeatureEngineer(BaseFeatureEngineer):
    """Create features related to team form."""

    def __init__(self, n_matches: int = 5):
        self.n_matches = n_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Args:
            data: dict {name:DataFrame}

        Returns:
            new DataFrame
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date')
        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            match_date = match['date']

            home_form = self._calculate_team_form(matches, home_id, match_date, self.n_matches)
            away_form = self._calculate_team_form(matches, away_id, match_date, self.n_matches)

            features = {
                'fixture_id': match['fixture_id'],
                'home_wins_last_n': home_form['wins'],
                'home_draws_last_n': home_form['draws'],
                'home_losses_last_n': home_form['losses'],
                'home_goals_scored_last_n': home_form['goals_scored'],
                'home_goals_conceded_last_n': home_form['goals_conceded'],
                'home_points_last_n': home_form['points'],
                'away_wins_last_n': away_form['wins'],
                'away_draws_last_n': away_form['draws'],
                'away_losses_last_n': away_form['losses'],
                'away_goals_scored_last_n': away_form['goals_scored'],
                'away_goals_conceded_last_n': away_form['goals_conceded'],
                'away_points_last_n': away_form['points'],
            }

            features_list.append(features)

        print(f"Created {len(features_list)} team form features (last {self.n_matches} matches)")
        return pd.DataFrame(features_list)

    def _calculate_team_form(self, matches: pd.DataFrame, team_id: int, current_date: pd.Timestamp, n: int) -> Dict:
        """
        Calculate team form based on N last matches.

        Args:
            matches: DataFrame with matches
            team_id: team id
            current_date: matches date
            n: number of matches

        Returns:
            Dict with stats
        """
        past_matches = matches[matches['date'] < current_date]
        team_matches = past_matches[(past_matches['home_team_id'] == team_id) |(past_matches['away_team_id'] == team_id)].tail(n)

        if len(team_matches) == 0:
            return {
                'wins': 0, 'draws': 0, 'losses': 0,
                'goals_scored': 0, 'goals_conceded': 0, 'points': 0
            }

        wins = draws = losses = 0
        goals_scored = goals_conceded = 0

        for _, match in team_matches.iterrows():
            is_home = match['home_team_id'] == team_id

            if is_home:
                team_goals = match['ft_home']
                opponent_goals = match['ft_away']
            else:
                team_goals = match['ft_away']
                opponent_goals = match['ft_home']

            goals_scored += team_goals
            goals_conceded += opponent_goals

            if team_goals > opponent_goals:
                wins += 1
            elif team_goals == opponent_goals:
                draws += 1
            else:
                losses += 1

        points = wins * 3 + draws

        return {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'points': points
        }



class ExponentialMovingAverageFeatureEngineer(BaseFeatureEngineer):
    """
    Creates Exponential Moving Average (EMA) features for teams.
    EMA gives more weight to recent matches compared to simple moving average.
    """

    def __init__(self, span: int = 5):
        """
        Args:
            span: Number of periods for EMA calculation
                  alpha = 2 / (span + 1)
                  span=5 means alpha=0.333, giving 33% weight to newest value
        """
        self.span = span
        self.alpha = 2 / (span + 1)

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate EMA for key team statistics.

        Args:
            data: dict {name:DataFrame}

        Returns:
            DataFrame with EMA features for home and away teams
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        features_list = []

        # Get unique teams
        teams = pd.concat([
            matches[['home_team_id']].rename(columns={'home_team_id': 'team_id'}),
            matches[['away_team_id']].rename(columns={'away_team_id': 'team_id'})
        ]).drop_duplicates()['team_id'].unique()

        # Initialize EMA storage for each team
        team_ema = {
            team_id: {
                'goals_scored_ema': None,
                'goals_conceded_ema': None,
                'points_ema': None,
                'xg_ema': None,
            }
            for team_id in teams
        }

        # Calculate EMA for each match
        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get current EMA values (before this match)
            home_ema = self._get_team_ema(team_ema, home_id)
            away_ema = self._get_team_ema(team_ema, away_id)

            features = {
                'fixture_id': match['fixture_id'],
                'home_goals_scored_ema': home_ema['goals_scored_ema'],
                'home_goals_conceded_ema': home_ema['goals_conceded_ema'],
                'home_points_ema': home_ema['points_ema'],
                'away_goals_scored_ema': away_ema['goals_scored_ema'],
                'away_goals_conceded_ema': away_ema['goals_conceded_ema'],
                'away_points_ema': away_ema['points_ema'],
            }

            features_list.append(features)

            # Update EMA after this match
            self._update_team_ema(
                team_ema, home_id, match['ft_home'],
                match['ft_away'], is_home=True
            )
            self._update_team_ema(
                team_ema, away_id, match['ft_away'],
                match['ft_home'], is_home=False
            )

        print(f"Created {len(features_list)} EMA features (span={self.span}, alpha={self.alpha:.3f})")
        return pd.DataFrame(features_list)

    def _get_team_ema(self, team_ema: Dict, team_id: int) -> Dict:
        """
        Get current EMA values for team, return 0 if no history.

        Args:
            team_ema: Dictionary storing EMA values
            team_id: Team ID

        Returns:
            Dictionary with current EMA values
        """
        ema = team_ema[team_id]
        return {
            'goals_scored_ema': ema['goals_scored_ema'] if ema['goals_scored_ema'] is not None else 0,
            'goals_conceded_ema': ema['goals_conceded_ema'] if ema['goals_conceded_ema'] is not None else 0,
            'points_ema': ema['points_ema'] if ema['points_ema'] is not None else 0,
        }

    def _update_team_ema(
            self,
            team_ema: Dict,
            team_id: int,
            goals_scored: int,
            goals_conceded: int,
            is_home: bool
    ):
        """
        Update EMA values after a match using formula:
        EMA_new = alpha * value_new + (1 - alpha) * EMA_old

        Args:
            team_ema: Dictionary storing EMA values
            team_id: Team ID
            goals_scored: Goals scored in this match
            goals_conceded: Goals conceded in this match
            is_home: Whether team played at home
        """
        # Calculate points from this match
        if goals_scored > goals_conceded:
            points = 3
        elif goals_scored == goals_conceded:
            points = 1
        else:
            points = 0

        ema = team_ema[team_id]

        # Update EMA using exponential smoothing formula
        if ema['goals_scored_ema'] is None:
            # First match - initialize with actual values
            ema['goals_scored_ema'] = float(goals_scored)
            ema['goals_conceded_ema'] = float(goals_conceded)
            ema['points_ema'] = float(points)
        else:
            # Apply EMA formula: EMA_new = alpha * value + (1-alpha) * EMA_old
            ema['goals_scored_ema'] = (
                    self.alpha * goals_scored +
                    (1 - self.alpha) * ema['goals_scored_ema']
            )
            ema['goals_conceded_ema'] = (
                    self.alpha * goals_conceded +
                    (1 - self.alpha) * ema['goals_conceded_ema']
            )
            ema['points_ema'] = (
                    self.alpha * points +
                    (1 - self.alpha) * ema['points_ema']
            )



class HomeAwayFormFeatureEngineer(BaseFeatureEngineer):
    """
    Creates separate form features for home and away matches.

    Unlike TeamFormFeatureEngineer which calculates form from all matches,
    this engineer calculates:
    - Home team's form ONLY from their home matches
    - Away team's form ONLY from their away matches

    This is more predictive because team performance varies by venue.
    """

    def __init__(self, n_matches: int = 5):
        """
        Args:
            n_matches: Number of recent home/away matches to consider
        """
        self.n_matches = n_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate venue-specific form features.

        Args:
            data: dict with 'matches' DataFrame

        Returns:
            DataFrame with home-only and away-only form features
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            match_date = match['date']

            # Home team's form from HOME matches only
            home_form = self._calculate_venue_form(
                matches, home_id, match_date, is_home=True
            )

            # Away team's form from AWAY matches only
            away_form = self._calculate_venue_form(
                matches, away_id, match_date, is_home=False
            )

            features = {
                'fixture_id': match['fixture_id'],
                # Home team at home
                'home_home_wins': home_form['wins'],
                'home_home_draws': home_form['draws'],
                'home_home_losses': home_form['losses'],
                'home_home_goals_scored': home_form['goals_scored'],
                'home_home_goals_conceded': home_form['goals_conceded'],
                'home_home_points': home_form['points'],
                'home_home_ppg': home_form['ppg'],
                # Away team away
                'away_away_wins': away_form['wins'],
                'away_away_draws': away_form['draws'],
                'away_away_losses': away_form['losses'],
                'away_away_goals_scored': away_form['goals_scored'],
                'away_away_goals_conceded': away_form['goals_conceded'],
                'away_away_points': away_form['points'],
                'away_away_ppg': away_form['ppg'],
                # Derived features
                'home_away_ppg_diff': home_form['ppg'] - away_form['ppg'],
                'home_away_gd_diff': home_form['goal_diff'] - away_form['goal_diff'],
            }

            features_list.append(features)

        print(f"Created {len(features_list)} home/away form features (last {self.n_matches} venue-specific matches)")
        return pd.DataFrame(features_list)

    def _calculate_venue_form(
        self,
        matches: pd.DataFrame,
        team_id: int,
        current_date,
        is_home: bool
    ) -> Dict:
        """
        Calculate team form from venue-specific matches only.

        Args:
            matches: All matches DataFrame
            team_id: Team to calculate for
            current_date: Current match date (exclude this and future)
            is_home: If True, get home matches; if False, get away matches

        Returns:
            Dict with form statistics
        """
        past_matches = matches[matches['date'] < current_date]

        if is_home:
            venue_matches = past_matches[past_matches['home_team_id'] == team_id].tail(self.n_matches)
        else:
            venue_matches = past_matches[past_matches['away_team_id'] == team_id].tail(self.n_matches)

        if len(venue_matches) == 0:
            return {
                'wins': 0, 'draws': 0, 'losses': 0,
                'goals_scored': 0, 'goals_conceded': 0,
                'points': 0, 'ppg': 0.0, 'goal_diff': 0.0
            }

        wins = draws = losses = 0
        goals_scored = goals_conceded = 0

        for _, match in venue_matches.iterrows():
            if is_home:
                team_goals = match['ft_home']
                opponent_goals = match['ft_away']
            else:
                team_goals = match['ft_away']
                opponent_goals = match['ft_home']

            goals_scored += team_goals
            goals_conceded += opponent_goals

            if team_goals > opponent_goals:
                wins += 1
            elif team_goals == opponent_goals:
                draws += 1
            else:
                losses += 1

        points = wins * 3 + draws
        n_matches = len(venue_matches)
        ppg = points / n_matches if n_matches > 0 else 0.0
        goal_diff = (goals_scored - goals_conceded) / n_matches if n_matches > 0 else 0.0

        return {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'points': points,
            'ppg': ppg,
            'goal_diff': goal_diff
        }



class StreakFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on winning/losing/drawing streaks.

    Streaks capture momentum:
    - Winning streak = team is confident, on a roll
    - Losing streak = team is struggling, low morale
    - Unbeaten streak = consistent performance

    Also tracks clean sheet and scoring streaks.
    """

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate streak features.

        Args:
            data: dict with 'matches' DataFrame

        Returns:
            DataFrame with streak features
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Track streaks for each team
        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())
        team_streaks = {
            team_id: {
                'current_result': None,  # 'W', 'D', 'L'
                'streak_length': 0,
                'unbeaten_streak': 0,
                'winless_streak': 0,
                'scoring_streak': 0,
                'clean_sheet_streak': 0,
                'results_history': [],  # last N results for form string
            }
            for team_id in all_teams
        }

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get current streaks BEFORE this match
            home_streaks = self._get_streak_features(team_streaks, home_id)
            away_streaks = self._get_streak_features(team_streaks, away_id)

            features = {
                'fixture_id': match['fixture_id'],
                # Win/loss streaks (positive = winning, negative = losing)
                'home_streak': home_streaks['streak'],
                'away_streak': away_streaks['streak'],
                'streak_diff': home_streaks['streak'] - away_streaks['streak'],
                # Unbeaten and winless
                'home_unbeaten_streak': home_streaks['unbeaten'],
                'away_unbeaten_streak': away_streaks['unbeaten'],
                'home_winless_streak': home_streaks['winless'],
                'away_winless_streak': away_streaks['winless'],
                # Scoring and clean sheets
                'home_scoring_streak': home_streaks['scoring'],
                'away_scoring_streak': away_streaks['scoring'],
                'home_clean_sheet_streak': home_streaks['clean_sheet'],
                'away_clean_sheet_streak': away_streaks['clean_sheet'],
            }

            features_list.append(features)

            # Update streaks AFTER recording features
            home_goals = match['ft_home']
            away_goals = match['ft_away']

            self._update_streaks(team_streaks, home_id, home_goals, away_goals)
            self._update_streaks(team_streaks, away_id, away_goals, home_goals)

        print(f"Created {len(features_list)} streak features")
        return pd.DataFrame(features_list)

    def _get_streak_features(self, team_streaks: Dict, team_id: int) -> Dict:
        """Get current streak values for a team."""
        streaks = team_streaks[team_id]

        # Convert streak to signed value (positive = winning, negative = losing)
        if streaks['current_result'] == 'W':
            streak = streaks['streak_length']
        elif streaks['current_result'] == 'L':
            streak = -streaks['streak_length']
        else:
            streak = 0

        return {
            'streak': streak,
            'unbeaten': streaks['unbeaten_streak'],
            'winless': streaks['winless_streak'],
            'scoring': streaks['scoring_streak'],
            'clean_sheet': streaks['clean_sheet_streak'],
        }

    def _update_streaks(self, team_streaks: Dict, team_id: int, goals_for: int, goals_against: int):
        """Update team streaks after a match."""
        streaks = team_streaks[team_id]

        # Determine result
        if goals_for > goals_against:
            result = 'W'
        elif goals_for == goals_against:
            result = 'D'
        else:
            result = 'L'

        # Update main streak
        if result == streaks['current_result']:
            streaks['streak_length'] += 1
        else:
            streaks['current_result'] = result
            streaks['streak_length'] = 1

        # Update unbeaten streak
        if result in ['W', 'D']:
            streaks['unbeaten_streak'] += 1
        else:
            streaks['unbeaten_streak'] = 0

        # Update winless streak
        if result in ['D', 'L']:
            streaks['winless_streak'] += 1
        else:
            streaks['winless_streak'] = 0

        # Update scoring streak
        if goals_for > 0:
            streaks['scoring_streak'] += 1
        else:
            streaks['scoring_streak'] = 0

        # Update clean sheet streak
        if goals_against == 0:
            streaks['clean_sheet_streak'] += 1
        else:
            streaks['clean_sheet_streak'] = 0

        # Keep results history (last 5)
        streaks['results_history'].append(result)
        if len(streaks['results_history']) > 5:
            streaks['results_history'].pop(0)


# =============================================================================
# FEATURE ENGINEERING V4 - New Engineers
# =============================================================================


class DixonColesDecayFeatureEngineer(BaseFeatureEngineer):
    """
    Creates time-weighted features using Dixon-Coles exponential decay.

    Unlike standard EMA which uses fixed alpha per match, Dixon-Coles
    uses calendar time for decay:
        weight = exp(-λ * days_since_match)
        where λ = ln(2) / half_life_days

    This means:
    - A match 30 days ago has ~50% weight (if half_life=30)
    - A match 60 days ago has ~25% weight
    - More recent matches dominate regardless of match frequency

    Research (Dixon & Coles 1997) suggests half-life of 1-3 years for
    team strength, but for form features 20-60 days is more appropriate.
    """

    # Stats to track with time decay
    STATS_TO_TRACK = [
        'goals_scored', 'goals_conceded', 'points', 'xg_for', 'xg_against',
        'shots_total', 'shots_on_target', 'fouls_committed'
    ]

    def __init__(self, half_life_days: float = 60.0, min_matches: int = 3):
        """
        Args:
            half_life_days: Days for weight to decay to 50%
                           - 20 days: Aggressive, very reactive to recent form
                           - 30 days: Balanced, moderate reactivity
                           - 45 days: Conservative, smoother trends
                           - 60 days: Stable, best backtest performance (recommended)
            min_matches: Minimum matches required before outputting features
        """
        self.half_life_days = half_life_days
        self.lambda_decay = np.log(2) / half_life_days  # λ = ln(2) / half_life
        self.min_matches = min_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate time-weighted features using Dixon-Coles decay.

        Args:
            data: dict with 'matches' DataFrame (and optionally 'player_stats')

        Returns:
            DataFrame with time-decayed features
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(matches['date']):
            matches['date'] = pd.to_datetime(matches['date'])

        # Get unique teams
        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())

        # Track match history per team: list of (date, stats_dict)
        team_history: Dict[int, List[tuple]] = {team_id: [] for team_id in all_teams}

        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            match_date = match['date']

            # Calculate time-weighted averages BEFORE this match
            home_features = self._calculate_weighted_features(
                team_history[home_id], match_date, prefix='home'
            )
            away_features = self._calculate_weighted_features(
                team_history[away_id], match_date, prefix='away'
            )

            features = {'fixture_id': fixture_id}
            features.update(home_features)
            features.update(away_features)

            # Add derived features
            features['dc_goals_diff'] = (
                features.get('home_goals_scored_dc', 0) -
                features.get('away_goals_scored_dc', 0)
            )
            features['dc_xg_diff'] = (
                features.get('home_xg_for_dc', 0) -
                features.get('away_xg_for_dc', 0)
            )
            features['dc_points_diff'] = (
                features.get('home_points_dc', 0) -
                features.get('away_points_dc', 0)
            )

            features_list.append(features)

            # Update history AFTER recording features
            home_stats = self._extract_match_stats(match, is_home=True)
            away_stats = self._extract_match_stats(match, is_home=False)

            team_history[home_id].append((match_date, home_stats))
            team_history[away_id].append((match_date, away_stats))

        print(f"Created {len(features_list)} Dixon-Coles decay features "
              f"(half_life={self.half_life_days} days, λ={self.lambda_decay:.4f})")

        return pd.DataFrame(features_list)

    def _calculate_weighted_features(
        self,
        history: List[tuple],
        current_date: pd.Timestamp,
        prefix: str
    ) -> Dict[str, float]:
        """
        Calculate time-weighted average of historical stats.

        Args:
            history: List of (date, stats_dict) tuples
            current_date: Current match date
            prefix: 'home' or 'away' for feature naming

        Returns:
            Dict of weighted feature values
        """
        features = {}

        if len(history) < self.min_matches:
            # Not enough history - return NaN
            for stat in self.STATS_TO_TRACK:
                features[f'{prefix}_{stat}_dc'] = np.nan
            features[f'{prefix}_dc_weight_sum'] = 0.0
            features[f'{prefix}_dc_matches'] = len(history)
            return features

        # Calculate weights and weighted sums
        weights = []
        stat_sums = {stat: 0.0 for stat in self.STATS_TO_TRACK}

        for hist_date, stats in history:
            days_ago = (current_date - hist_date).days
            weight = np.exp(-self.lambda_decay * days_ago)
            weights.append(weight)

            for stat in self.STATS_TO_TRACK:
                value = stats.get(stat, 0)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = 0
                stat_sums[stat] += weight * value

        total_weight = sum(weights)

        # Normalize by total weight
        for stat in self.STATS_TO_TRACK:
            if total_weight > 0:
                features[f'{prefix}_{stat}_dc'] = stat_sums[stat] / total_weight
            else:
                features[f'{prefix}_{stat}_dc'] = np.nan

        # Add metadata
        features[f'{prefix}_dc_weight_sum'] = total_weight
        features[f'{prefix}_dc_matches'] = len(history)

        return features

    def _extract_match_stats(self, match: pd.Series, is_home: bool) -> Dict[str, float]:
        """
        Extract relevant stats from a match for history tracking.

        Args:
            match: Match row from DataFrame
            is_home: Whether extracting for home team

        Returns:
            Dict of stats for this match
        """
        if is_home:
            goals_scored = match.get('ft_home', 0)
            goals_conceded = match.get('ft_away', 0)
            xg_for = match.get('home_xg', match.get('xg_home', 0))
            xg_against = match.get('away_xg', match.get('xg_away', 0))
            shots_total = match.get('home_shots_total', 0)
            shots_on = match.get('home_shots_on', 0)
            fouls = match.get('home_fouls', 0)
        else:
            goals_scored = match.get('ft_away', 0)
            goals_conceded = match.get('ft_home', 0)
            xg_for = match.get('away_xg', match.get('xg_away', 0))
            xg_against = match.get('home_xg', match.get('xg_home', 0))
            shots_total = match.get('away_shots_total', 0)
            shots_on = match.get('away_shots_on', 0)
            fouls = match.get('away_fouls', 0)

        # Calculate points
        if goals_scored > goals_conceded:
            points = 3
        elif goals_scored == goals_conceded:
            points = 1
        else:
            points = 0

        return {
            'goals_scored': float(goals_scored) if pd.notna(goals_scored) else 0,
            'goals_conceded': float(goals_conceded) if pd.notna(goals_conceded) else 0,
            'points': float(points),
            'xg_for': float(xg_for) if pd.notna(xg_for) else 0,
            'xg_against': float(xg_against) if pd.notna(xg_against) else 0,
            'shots_total': float(shots_total) if pd.notna(shots_total) else 0,
            'shots_on_target': float(shots_on) if pd.notna(shots_on) else 0,
            'fouls_committed': float(fouls) if pd.notna(fouls) else 0,
        }

    def get_feature_names(self) -> List[str]:
        """Return list of feature names created by this engineer."""
        features = []
        for prefix in ['home', 'away']:
            for stat in self.STATS_TO_TRACK:
                features.append(f'{prefix}_{stat}_dc')
            features.append(f'{prefix}_dc_weight_sum')
            features.append(f'{prefix}_dc_matches')

        # Derived features
        features.extend(['dc_goals_diff', 'dc_xg_diff', 'dc_points_diff'])
        return features


