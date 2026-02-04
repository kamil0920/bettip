"""Feature engineering - ELO ratings and Poisson-based features."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer


class ELORatingFeatureEngineer(BaseFeatureEngineer):
    """
    Creates ELO rating features for teams.

    ELO is a rating system that updates after each match based on:
    - Expected outcome vs actual outcome
    - K-factor (sensitivity to recent results)
    - Home advantage adjustment

    Standard ELO formula:
    R_new = R_old + K * (S - E)
    where:
    - S = actual score (1 for win, 0.5 for draw, 0 for loss)
    - E = expected score = 1 / (1 + 10^((R_opponent - R_self) / 400))
    """

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        home_advantage: float = 100.0,
    ):
        """
        Args:
            initial_rating: Starting ELO for new teams
            k_factor: How much ratings change per match (higher = more volatile)
            home_advantage: ELO points added for home team in expected calc
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate ELO ratings for each team before each match.

        Maintains three rating tracks per team:
        - Overall ELO (updated every match)
        - Home ELO (updated only on home matches)
        - Away ELO (updated only on away matches)

        Args:
            data: dict with 'matches' DataFrame

        Returns:
            DataFrame with ELO features
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Initialize ratings
        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())
        team_ratings = {team_id: self.initial_rating for team_id in all_teams}
        # Venue-specific ratings
        home_venue_ratings = {team_id: self.initial_rating for team_id in all_teams}
        away_venue_ratings = {team_id: self.initial_rating for team_id in all_teams}

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            home_elo = team_ratings[home_id]
            away_elo = team_ratings[away_id]

            home_expected = self._expected_score(home_elo + self.home_advantage, away_elo)
            away_expected = 1 - home_expected

            elo_diff = home_elo - away_elo

            # Venue-specific ratings (before this match)
            home_venue_elo = home_venue_ratings[home_id]
            away_venue_elo = away_venue_ratings[away_id]
            venue_elo_diff = home_venue_elo - away_venue_elo

            # Venue dependency: how much better/worse a team is at home vs away
            home_team_venue_gap = home_venue_ratings[home_id] - away_venue_ratings[home_id]
            away_team_venue_gap = home_venue_ratings[away_id] - away_venue_ratings[away_id]

            features = {
                'fixture_id': match['fixture_id'],
                'home_elo': home_elo,
                'away_elo': away_elo,
                'elo_diff': elo_diff,
                'home_win_prob_elo': home_expected,
                'away_win_prob_elo': away_expected,
                # Venue-specific features
                'home_venue_elo': home_venue_elo,
                'away_venue_elo': away_venue_elo,
                'venue_elo_diff': venue_elo_diff,
                'home_team_venue_gap': home_team_venue_gap,
                'away_team_venue_gap': away_team_venue_gap,
            }
            features_list.append(features)

            home_goals = match['ft_home']
            away_goals = match['ft_away']

            if home_goals > away_goals:
                home_actual, away_actual = 1.0, 0.0
            elif home_goals < away_goals:
                home_actual, away_actual = 0.0, 1.0
            else:
                home_actual, away_actual = 0.5, 0.5

            # Update overall ratings
            team_ratings[home_id] = home_elo + self.k_factor * (home_actual - home_expected)
            team_ratings[away_id] = away_elo + self.k_factor * (away_actual - away_expected)

            # Update venue-specific ratings
            venue_home_expected = self._expected_score(
                home_venue_elo + self.home_advantage, away_venue_elo
            )
            home_venue_ratings[home_id] = home_venue_elo + self.k_factor * (
                home_actual - venue_home_expected
            )
            away_venue_ratings[away_id] = away_venue_elo + self.k_factor * (
                away_actual - (1 - venue_home_expected)
            )

        print(f"Created {len(features_list)} ELO rating features (K={self.k_factor}, with venue-specific)")
        return pd.DataFrame(features_list)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for team A against team B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))



class PoissonFeatureEngineer(BaseFeatureEngineer):
    """
    Creates Poisson-based features for goal prediction.

    Uses historical goal-scoring rates to estimate:
    - Expected goals for each team
    - Match outcome probabilities based on Poisson distribution

    The Poisson model assumes:
    - Goals are independent events
    - Average rate (lambda) based on attack/defense strength
    """

    def __init__(self, lookback_matches: int = 10):
        """
        Args:
            lookback_matches: Number of recent matches to calculate rates
        """
        self.lookback_matches = lookback_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate Poisson-based features.

        Args:
            data: dict with 'matches' DataFrame

        Returns:
            DataFrame with Poisson features
        """
        from scipy.stats import poisson

        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Calculate league averages (will be updated as we go)
        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())

        # Track team stats
        team_stats = {
            team_id: {
                'goals_scored': [],
                'goals_conceded': [],
                'home_goals_scored': [],
                'home_goals_conceded': [],
                'away_goals_scored': [],
                'away_goals_conceded': [],
            }
            for team_id in all_teams
        }

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            home_attack = self._get_attack_strength(team_stats, home_id, is_home=True)
            home_defense = self._get_defense_strength(team_stats, home_id, is_home=True)
            away_attack = self._get_attack_strength(team_stats, away_id, is_home=False)
            away_defense = self._get_defense_strength(team_stats, away_id, is_home=False)

            # NOTE: Baselines calibrated to actual league averages
            # Feature-level xG underestimates slightly (-0.15), but this is acceptable
            # Overconfidence in predictions comes from ML model outputs, addressed via
            # calibration module in src/calibration/market_calibrator.py
            home_xg = max(0.5, home_attack - away_defense + 1.5)  # baseline ~1.5 goals
            away_xg = max(0.5, away_attack - home_defense + 1.0)   # away teams score less

            home_win_prob, draw_prob, away_win_prob = self._poisson_outcome_probs(home_xg, away_xg)

            features = {
                'fixture_id': match['fixture_id'],
                'home_xg_poisson': home_xg,
                'away_xg_poisson': away_xg,
                'xg_diff': home_xg - away_xg,
                'home_attack_strength': home_attack,
                'home_defense_strength': home_defense,
                'away_attack_strength': away_attack,
                'away_defense_strength': away_defense,
                'poisson_home_win_prob': home_win_prob,
                'poisson_draw_prob': draw_prob,
                'poisson_away_win_prob': away_win_prob,
            }
            features_list.append(features)

            home_goals = match['ft_home']
            away_goals = match['ft_away']

            team_stats[home_id]['goals_scored'].append(home_goals)
            team_stats[home_id]['goals_conceded'].append(away_goals)
            team_stats[home_id]['home_goals_scored'].append(home_goals)
            team_stats[home_id]['home_goals_conceded'].append(away_goals)

            team_stats[away_id]['goals_scored'].append(away_goals)
            team_stats[away_id]['goals_conceded'].append(home_goals)
            team_stats[away_id]['away_goals_scored'].append(away_goals)
            team_stats[away_id]['away_goals_conceded'].append(home_goals)

        print(f"Created {len(features_list)} Poisson features (lookback={self.lookback_matches})")
        return pd.DataFrame(features_list)

    def _get_attack_strength(self, team_stats: Dict, team_id: int, is_home: bool) -> float:
        """Calculate attack strength (goals scored per match - league avg)."""
        stats = team_stats[team_id]
        key = 'home_goals_scored' if is_home else 'away_goals_scored'
        goals = stats[key][-self.lookback_matches:] if stats[key] else []

        if not goals:
            return 0.0

        return np.mean(goals) - 1.3

    def _get_defense_strength(self, team_stats: Dict, team_id: int, is_home: bool) -> float:
        """Calculate defense strength (goals conceded per match - league avg)."""
        stats = team_stats[team_id]
        key = 'home_goals_conceded' if is_home else 'away_goals_conceded'
        goals = stats[key][-self.lookback_matches:] if stats[key] else []

        if not goals:
            return 0.0

        return np.mean(goals) - 1.3

    def _poisson_outcome_probs(self, home_xg: float, away_xg: float, max_goals: int = 7) -> tuple:
        """
        Calculate match outcome probabilities using Poisson distribution.

        Returns:
            (home_win_prob, draw_prob, away_win_prob)
        """
        from scipy.stats import poisson

        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0

        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                prob = poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)

                if home_goals > away_goals:
                    home_win_prob += prob
                elif home_goals == away_goals:
                    draw_prob += prob
                else:
                    away_win_prob += prob

        total = home_win_prob + draw_prob + away_win_prob
        if total > 0:
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total

        return home_win_prob, draw_prob, away_win_prob



class TeamRatingFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on average team ratings.

    Uses player ratings to calculate overall team strength.
    """

    def __init__(self, lookback_matches: int = 5):
        """
        Args:
            lookback_matches: Number of recent matches for rating calculation
        """
        self.lookback_matches = lookback_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate team rating features.
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)
        player_stats = data.get('player_stats')

        if player_stats is None or player_stats.empty:
            print("No player stats data available, skipping team rating features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Track team ratings history
        team_ratings_history = {}  # {team_id: [avg_ratings]}
        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get historical team ratings
            home_ratings = team_ratings_history.get(home_id, [])
            away_ratings = team_ratings_history.get(away_id, [])

            home_avg = sum(home_ratings) / len(home_ratings) if home_ratings else 6.5
            away_avg = sum(away_ratings) / len(away_ratings) if away_ratings else 6.5

            features = {
                'fixture_id': fixture_id,
                'home_team_avg_rating': home_avg,
                'away_team_avg_rating': away_avg,
                'team_rating_diff': home_avg - away_avg,
            }
            features_list.append(features)

            # Update team ratings from this match
            match_stats = player_stats[
                (player_stats['fixture_id'] == fixture_id) &
                (player_stats['rating'].notna()) &
                (player_stats['rating'] > 0)
            ]

            for team_id in [home_id, away_id]:
                team_match_ratings = match_stats[
                    match_stats['team_id'] == team_id
                ]['rating'].tolist()

                if team_match_ratings:
                    avg_rating = sum(team_match_ratings) / len(team_match_ratings)

                    if team_id not in team_ratings_history:
                        team_ratings_history[team_id] = []
                    team_ratings_history[team_id].append(avg_rating)

                    if len(team_ratings_history[team_id]) > self.lookback_matches:
                        team_ratings_history[team_id].pop(0)

        print(f"Created {len(features_list)} team rating features")
        return pd.DataFrame(features_list)


