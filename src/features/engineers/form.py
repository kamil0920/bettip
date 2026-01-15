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


