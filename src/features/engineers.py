"""Feature engineering implementations."""
from typing import Dict, List

import numpy as np
import pandas as pd

from src.features.interfaces import IFeatureEngineer


class BaseFeatureEngineer(IFeatureEngineer):
    """Base class for feature engineers."""

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Template method pattern."""
        raise NotImplementedError


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


class TeamStatsFeatureEngineer(BaseFeatureEngineer):
    """
    Create EMA (Exponential Moving Average) features from player stats.

    Uses EMA instead of simple average because:
    - Recent matches are more predictive of current form
    - EMA naturally handles the "recency bias" in sports
    - Standard approach in sports analytics

    EMA formula: EMA_new = alpha * value_new + (1 - alpha) * EMA_old
    where alpha = 2 / (span + 1)
    """

    # Stats to track with EMA
    STATS_TO_TRACK = [
        'rating', 'shots_total', 'shots_on', 'passes_total',
        'passes_key', 'passes_accuracy', 'tackles_total', 'fouls_committed'
    ]

    def __init__(self, span: int = 5):
        """
        Args:
            span: EMA span (similar to "last N matches" but with decay)
                  span=5 means alpha=0.333, giving 33% weight to newest value
        """
        self.span = span
        self.alpha = 2 / (span + 1)

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate EMA of player stats for each team.

        For each match, returns the EMA values BEFORE that match
        (no data leakage - we don't use current match stats).

        Args:
            data: dict with 'matches' and 'player_stats' DataFrames

        Returns:
            DataFrame with EMA team stats features
        """
        if 'player_stats' not in data:
            print("Warning: player_stats not found, skipping TeamStatsFeatureEngineer")
            return pd.DataFrame()

        player_stats = data['player_stats'].copy()
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Aggregate player stats per fixture and team
        fixture_team_stats = player_stats.groupby(['fixture_id', 'team_id']).agg({
            'rating': 'mean',
            'shots_total': 'sum',
            'shots_on': 'sum',
            'passes_total': 'sum',
            'passes_key': 'sum',
            'passes_accuracy': 'mean',
            'tackles_total': 'sum',
            'fouls_committed': 'sum',
        }).reset_index()

        # Merge with matches to get dates
        fixture_team_stats = fixture_team_stats.merge(
            matches[['fixture_id', 'date']],
            on='fixture_id',
            how='left'
        ).sort_values('date')

        # Get unique teams
        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())

        # Initialize EMA storage for each team
        team_ema = {
            team_id: {stat: None for stat in self.STATS_TO_TRACK}
            for team_id in all_teams
        }

        # Build lookup: fixture_id -> team_id -> stats
        fixture_stats_lookup = {}
        for _, row in fixture_team_stats.iterrows():
            fid = row['fixture_id']
            tid = row['team_id']
            if fid not in fixture_stats_lookup:
                fixture_stats_lookup[fid] = {}
            fixture_stats_lookup[fid][tid] = {
                stat: row[stat] for stat in self.STATS_TO_TRACK
            }

        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get CURRENT EMA values (before this match)
            home_ema = self._get_current_ema(team_ema, home_id)
            away_ema = self._get_current_ema(team_ema, away_id)

            # Build features
            features = {'fixture_id': fixture_id}
            for stat in self.STATS_TO_TRACK:
                features[f'home_{stat}_ema'] = home_ema[stat]
                features[f'away_{stat}_ema'] = away_ema[stat]

            features_list.append(features)

            # Update EMA AFTER recording features (for next iteration)
            if fixture_id in fixture_stats_lookup:
                if home_id in fixture_stats_lookup[fixture_id]:
                    self._update_ema(team_ema, home_id, fixture_stats_lookup[fixture_id][home_id])
                if away_id in fixture_stats_lookup[fixture_id]:
                    self._update_ema(team_ema, away_id, fixture_stats_lookup[fixture_id][away_id])

        print(f"Created {len(features_list)} team stats EMA features (span={self.span}, alpha={self.alpha:.3f})")
        return pd.DataFrame(features_list)

    def _get_current_ema(self, team_ema: Dict, team_id: int) -> Dict:
        """Get current EMA values for a team (0 if no history)."""
        ema = team_ema.get(team_id, {})
        return {
            stat: (ema.get(stat) if ema.get(stat) is not None else 0.0)
            for stat in self.STATS_TO_TRACK
        }

    def _update_ema(self, team_ema: Dict, team_id: int, new_stats: Dict) -> None:
        """
        Update EMA values after a match.

        EMA formula: EMA_new = alpha * value + (1 - alpha) * EMA_old
        """
        ema = team_ema[team_id]

        for stat in self.STATS_TO_TRACK:
            new_value = new_stats.get(stat, 0)
            if new_value is None or (isinstance(new_value, float) and np.isnan(new_value)):
                new_value = 0

            if ema[stat] is None:
                # First match - initialize with actual value
                ema[stat] = float(new_value)
            else:
                # Apply EMA formula
                ema[stat] = self.alpha * new_value + (1 - self.alpha) * ema[stat]


class MatchOutcomeFeatureEngineer(BaseFeatureEngineer):
    """Creates target variables for prediction."""

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Args:
            data: dict {name:DataFrame}

        Returns:
            DataFrame with target variables
        """
        matches = data['matches'].copy()

        matches['match_result'] = np.sign(matches['ft_home'] - matches['ft_away'])
        matches['home_win'] = (matches['ft_home'] > matches['ft_away']).astype(int)
        matches['draw'] = (matches['ft_home'] == matches['ft_away']).astype(int)
        matches['away_win'] = (matches['ft_home'] < matches['ft_away']).astype(int)
        matches['total_goals'] = matches['ft_home'] + matches['ft_away']
        matches['goal_difference'] = matches['ft_home'] - matches['ft_away']

        target_cols = [
            'fixture_id', 'match_result', 'home_win', 'draw',
            'away_win', 'total_goals', 'goal_difference'
        ]

        print(f"Created target variables")
        return matches[target_cols]


class HeadToHeadFeatureEngineer(BaseFeatureEngineer):
    """Creates features related to history of face-to-face matches."""

    def __init__(self, n_h2h: int = 3):
        """
        Args:
            n_h2h: how many matches take
        """
        self.n_h2h = n_h2h

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Creates features head-to-head.

        Args:
            data: dict {name:DataFrame}

        Returns:
            new DataFrame with H2H features
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date')

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            match_date = match['date']

            h2h_matches = matches[
                (matches['date'] < match_date) &
                (
                        ((matches['home_team_id'] == home_id) & (matches['away_team_id'] == away_id)) |
                        ((matches['home_team_id'] == away_id) & (matches['away_team_id'] == home_id))
                )
                ].tail(self.n_h2h)

            if len(h2h_matches) == 0:
                h2h_home_wins = h2h_draws = h2h_away_wins = 0
                h2h_avg_goals = 0
            else:
                h2h_home_wins = h2h_draws = h2h_away_wins = 0
                total_goals = 0

                for _, h2h in h2h_matches.iterrows():
                    if h2h['home_team_id'] == home_id:
                        if h2h['ft_home'] > h2h['ft_away']:
                            h2h_home_wins += 1
                        elif h2h['ft_home'] == h2h['ft_away']:
                            h2h_draws += 1
                        else:
                            h2h_away_wins += 1
                    else:
                        if h2h['ft_away'] > h2h['ft_home']:
                            h2h_home_wins += 1
                        elif h2h['ft_away'] == h2h['ft_home']:
                            h2h_draws += 1
                        else:
                            h2h_away_wins += 1

                    total_goals += h2h['ft_home'] + h2h['ft_away']

                h2h_avg_goals = total_goals / len(h2h_matches)

            features = {
                'fixture_id': match['fixture_id'],
                'h2h_home_wins': h2h_home_wins,
                'h2h_draws': h2h_draws,
                'h2h_away_wins': h2h_away_wins,
                'h2h_avg_goals': h2h_avg_goals
            }

            features_list.append(features)

        print(f"Created {len(features_list)} head-to-head features (last {self.n_h2h} matches)")
        return pd.DataFrame(features_list)


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

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            home_elo = team_ratings[home_id]
            away_elo = team_ratings[away_id]

            home_expected = self._expected_score(home_elo + self.home_advantage, away_elo)
            away_expected = 1 - home_expected

            elo_diff = home_elo - away_elo

            features = {
                'fixture_id': match['fixture_id'],
                'home_elo': home_elo,
                'away_elo': away_elo,
                'elo_diff': elo_diff,
                'home_win_prob_elo': home_expected,
                'away_win_prob_elo': away_expected,
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

            team_ratings[home_id] = home_elo + self.k_factor * (home_actual - home_expected)
            team_ratings[away_id] = away_elo + self.k_factor * (away_actual - away_expected)

        print(f"Created {len(features_list)} ELO rating features (K={self.k_factor})")
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

            home_xg = max(0.5, home_attack - away_defense + 1.5)  # baseline ~1.5 goals
            away_xg = max(0.5, away_attack - home_defense + 1.0)  # away teams score less

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


class GoalDifferenceFeatureEngineer(BaseFeatureEngineer):
    """
    Creates goal difference based features.

    Goal difference is a strong predictor of team quality and
    is used in league standings (tiebreaker).
    """

    def __init__(self, lookback_matches: int = 5):
        self.lookback_matches = lookback_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate goal difference features."""
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())
        team_gd = {team_id: [] for team_id in all_teams}

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            home_recent_gd = team_gd[home_id][-self.lookback_matches:]
            away_recent_gd = team_gd[away_id][-self.lookback_matches:]

            home_avg_gd = np.mean(home_recent_gd) if home_recent_gd else 0.0
            away_avg_gd = np.mean(away_recent_gd) if away_recent_gd else 0.0

            home_total_gd = sum(home_recent_gd) if home_recent_gd else 0
            away_total_gd = sum(away_recent_gd) if away_recent_gd else 0

            features = {
                'fixture_id': match['fixture_id'],
                'home_avg_goal_diff': home_avg_gd,
                'away_avg_goal_diff': away_avg_gd,
                'home_total_goal_diff': home_total_gd,
                'away_total_goal_diff': away_total_gd,
                'goal_diff_advantage': home_avg_gd - away_avg_gd,
            }
            features_list.append(features)

            home_goals = match['ft_home']
            away_goals = match['ft_away']
            team_gd[home_id].append(home_goals - away_goals)
            team_gd[away_id].append(away_goals - home_goals)

        print(f"Created {len(features_list)} goal difference features")
        return pd.DataFrame(features_list)


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


class RestDaysFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features related to rest days between matches.

    Rest days can significantly impact performance:
    - Too few days (fatigue, no recovery)
    - Too many days (rust, loss of match rhythm)
    - Ideal is typically 4-7 days

    Also calculates relative rest advantage.
    """

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate rest days features for each team.

        Args:
            data: dict with 'matches' DataFrame

        Returns:
            DataFrame with rest days features
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Convert date to datetime if needed
        if matches['date'].dtype == 'object':
            matches['date'] = pd.to_datetime(matches['date'])

        # Track last match date for each team
        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())
        last_match_date = {team_id: None for team_id in all_teams}

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            match_date = match['date']

            # Calculate rest days
            home_rest = self._calculate_rest_days(last_match_date, home_id, match_date)
            away_rest = self._calculate_rest_days(last_match_date, away_id, match_date)

            # Rest advantage (positive = home team more rested)
            rest_advantage = home_rest - away_rest if (home_rest is not None and away_rest is not None) else 0

            # Categorize rest (1 = short <4 days, 2 = normal 4-7 days, 3 = long >7 days)
            home_rest_category = self._categorize_rest(home_rest)
            away_rest_category = self._categorize_rest(away_rest)

            features = {
                'fixture_id': match['fixture_id'],
                'home_rest_days': home_rest if home_rest is not None else 7,  # default to normal
                'away_rest_days': away_rest if away_rest is not None else 7,
                'rest_days_diff': rest_advantage,
                'home_short_rest': 1 if home_rest_category == 1 else 0,
                'away_short_rest': 1 if away_rest_category == 1 else 0,
                'home_long_rest': 1 if home_rest_category == 3 else 0,
                'away_long_rest': 1 if away_rest_category == 3 else 0,
            }

            features_list.append(features)

            # Update last match dates
            last_match_date[home_id] = match_date
            last_match_date[away_id] = match_date

        print(f"Created {len(features_list)} rest days features")
        return pd.DataFrame(features_list)

    def _calculate_rest_days(self, last_match_date: Dict, team_id: int, current_date) -> int:
        """Calculate days since last match for a team."""
        last_date = last_match_date.get(team_id)

        if last_date is None:
            return None

        delta = current_date - last_date
        return delta.days

    def _categorize_rest(self, rest_days: int) -> int:
        """
        Categorize rest days.

        1 = Short rest (< 4 days) - potential fatigue
        2 = Normal rest (4-7 days) - optimal
        3 = Long rest (> 7 days) - potential rust
        """
        if rest_days is None:
            return 2  # assume normal

        if rest_days < 4:
            return 1
        elif rest_days <= 7:
            return 2
        else:
            return 3


class LeaguePositionFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on current league table position.

    League position is a strong indicator of team quality and
    can predict outcomes (top teams beat bottom teams).

    Features:
    - Current position
    - Points
    - Points per game
    - Goal difference in season
    """

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate league position features.

        Args:
            data: dict with 'matches' DataFrame

        Returns:
            DataFrame with league position features
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Track season stats for each team
        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())
        team_stats = {
            team_id: {
                'points': 0,
                'played': 0,
                'goals_for': 0,
                'goals_against': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
            }
            for team_id in all_teams
        }

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get current standings BEFORE this match
            standings = self._calculate_standings(team_stats)

            home_position = standings.get(home_id, {'position': 10, 'ppg': 0, 'gd': 0})
            away_position = standings.get(away_id, {'position': 10, 'ppg': 0, 'gd': 0})

            features = {
                'fixture_id': match['fixture_id'],
                'home_league_position': home_position['position'],
                'away_league_position': away_position['position'],
                'position_diff': away_position['position'] - home_position['position'],  # positive = home higher
                'home_season_ppg': home_position['ppg'],
                'away_season_ppg': away_position['ppg'],
                'home_season_gd': home_position['gd'],
                'away_season_gd': away_position['gd'],
                'ppg_diff': home_position['ppg'] - away_position['ppg'],
                'season_gd_diff': home_position['gd'] - away_position['gd'],
            }

            features_list.append(features)

            # Update stats AFTER recording features
            home_goals = match['ft_home']
            away_goals = match['ft_away']

            self._update_team_stats(team_stats, home_id, home_goals, away_goals)
            self._update_team_stats(team_stats, away_id, away_goals, home_goals)

        print(f"Created {len(features_list)} league position features")
        return pd.DataFrame(features_list)

    def _update_team_stats(self, team_stats: Dict, team_id: int, goals_for: int, goals_against: int):
        """Update team stats after a match."""
        stats = team_stats[team_id]
        stats['played'] += 1
        stats['goals_for'] += goals_for
        stats['goals_against'] += goals_against

        if goals_for > goals_against:
            stats['points'] += 3
            stats['wins'] += 1
        elif goals_for == goals_against:
            stats['points'] += 1
            stats['draws'] += 1
        else:
            stats['losses'] += 1

    def _calculate_standings(self, team_stats: Dict) -> Dict:
        """
        Calculate current league standings.

        Returns dict with position, ppg, and goal difference for each team.
        """
        standings_data = []

        for team_id, stats in team_stats.items():
            played = stats['played']
            if played == 0:
                ppg = 0.0
                gd = 0.0
            else:
                ppg = stats['points'] / played
                gd = (stats['goals_for'] - stats['goals_against']) / played

            standings_data.append({
                'team_id': team_id,
                'points': stats['points'],
                'gd_total': stats['goals_for'] - stats['goals_against'],
                'gf': stats['goals_for'],
                'ppg': ppg,
                'gd': gd,
                'played': played,
            })

        # Sort by points, then goal difference, then goals for
        standings_data.sort(key=lambda x: (-x['points'], -x['gd_total'], -x['gf']))

        result = {}
        for pos, team in enumerate(standings_data, 1):
            result[team['team_id']] = {
                'position': pos,
                'ppg': team['ppg'],
                'gd': team['gd'],
            }

        return result


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


class FormationFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on team formations.

    Formations can indicate playing style:
    - 4-3-3, 4-2-3-1: Attacking
    - 5-3-2, 5-4-1: Defensive
    - 3-5-2: Balanced/wing play

    Also encodes formation matchups (attacking vs defensive).
    """

    # Formation categories
    ATTACKING_FORMATIONS = ['4-3-3', '4-2-3-1', '3-4-3', '4-1-4-1']
    DEFENSIVE_FORMATIONS = ['5-3-2', '5-4-1', '4-5-1', '5-2-3']
    BALANCED_FORMATIONS = ['4-4-2', '3-5-2', '4-4-1-1', '4-1-2-1-2']

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create formation-based features.

        Args:
            data: dict with 'matches' and 'lineups' DataFrames

        Returns:
            DataFrame with formation features
        """
        matches = data['matches'].copy()
        lineups = data.get('lineups')

        if lineups is None or lineups.empty:
            print("No lineups data available, skipping formation features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Get formation per team per match
        formations = lineups[lineups['formation'].notna()][
            ['fixture_id', 'team_id', 'formation']
        ].drop_duplicates()

        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get formations for this match
            match_formations = formations[formations['fixture_id'] == fixture_id]

            home_formation = match_formations[
                match_formations['team_id'] == home_id
            ]['formation'].values
            away_formation = match_formations[
                match_formations['team_id'] == away_id
            ]['formation'].values

            home_form = home_formation[0] if len(home_formation) > 0 else None
            away_form = away_formation[0] if len(away_formation) > 0 else None

            features = {
                'fixture_id': fixture_id,
                'home_formation_attacking': 1 if home_form in self.ATTACKING_FORMATIONS else 0,
                'home_formation_defensive': 1 if home_form in self.DEFENSIVE_FORMATIONS else 0,
                'away_formation_attacking': 1 if away_form in self.ATTACKING_FORMATIONS else 0,
                'away_formation_defensive': 1 if away_form in self.DEFENSIVE_FORMATIONS else 0,
                # Matchup: 1 if home attacking vs away defensive, -1 if opposite
                'formation_matchup': self._calculate_matchup(home_form, away_form),
                # Number of defenders (from formation string)
                'home_defenders': self._count_defenders(home_form),
                'away_defenders': self._count_defenders(away_form),
            }

            features_list.append(features)

        print(f"Created {len(features_list)} formation features")
        return pd.DataFrame(features_list)

    def _calculate_matchup(self, home_form: str, away_form: str) -> int:
        """Calculate formation matchup advantage."""
        if home_form is None or away_form is None:
            return 0

        home_attacking = home_form in self.ATTACKING_FORMATIONS
        home_defensive = home_form in self.DEFENSIVE_FORMATIONS
        away_attacking = away_form in self.ATTACKING_FORMATIONS
        away_defensive = away_form in self.DEFENSIVE_FORMATIONS

        # Attacking vs Defensive = advantage
        if home_attacking and away_defensive:
            return 1
        elif home_defensive and away_attacking:
            return -1
        return 0

    def _count_defenders(self, formation: str) -> int:
        """Extract number of defenders from formation string."""
        if formation is None:
            return 4  # default

        try:
            parts = formation.split('-')
            return int(parts[0])
        except (IndexError, ValueError):
            return 4


class CoachFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on coaches/managers.

    Features:
    - Coach change indicator (new coach in last N matches)
    - Coach tenure (how long has coach been at club)
    """

    def __init__(self, lookback_matches: int = 5):
        """
        Args:
            lookback_matches: Number of matches to look back for coach changes
        """
        self.lookback_matches = lookback_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create coach-based features.

        Note: Coach data would need to be extracted from lineups.
        For now, we'll use a simplified approach based on available data.
        """
        matches = data['matches'].copy()
        lineups = data.get('lineups')

        # If no lineups data, return empty features
        if lineups is None or lineups.empty:
            print("No lineups data available, skipping coach features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Coach features would require coach_id in lineups
        # For now, return placeholder
        features_list = []
        for idx, match in matches.iterrows():
            features = {
                'fixture_id': match['fixture_id'],
                # Placeholder - would need coach data
                'home_coach_change_recent': 0,
                'away_coach_change_recent': 0,
            }
            features_list.append(features)

        print(f"Created {len(features_list)} coach features (placeholder)")
        return pd.DataFrame(features_list)


class LineupStabilityFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on lineup stability.

    Teams with stable lineups often perform better due to:
    - Better understanding between players
    - Established partnerships
    - Fewer injury disruptions
    """

    def __init__(self, lookback_matches: int = 3):
        """
        Args:
            lookback_matches: Number of recent matches to compare lineups
        """
        self.lookback_matches = lookback_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate lineup stability features.
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)
        lineups = data.get('lineups')

        if lineups is None or lineups.empty:
            print("No lineups data available, skipping lineup stability features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Get starting XI per team per match
        starters = lineups[lineups['starting'] == True][
            ['fixture_id', 'team_id', 'player_id']
        ].copy()

        # Track lineups history per team
        team_lineup_history = {}
        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get current match lineups
            match_starters = starters[starters['fixture_id'] == fixture_id]
            home_starters = set(
                match_starters[match_starters['team_id'] == home_id]['player_id'].tolist()
            )
            away_starters = set(
                match_starters[match_starters['team_id'] == away_id]['player_id'].tolist()
            )

            # Calculate stability vs recent matches
            home_stability = self._calculate_stability(
                team_lineup_history.get(home_id, []), home_starters
            )
            away_stability = self._calculate_stability(
                team_lineup_history.get(away_id, []), away_starters
            )

            features = {
                'fixture_id': fixture_id,
                'home_lineup_stability': home_stability,
                'away_lineup_stability': away_stability,
                'lineup_stability_diff': home_stability - away_stability,
            }
            features_list.append(features)

            # Update history
            if home_id not in team_lineup_history:
                team_lineup_history[home_id] = []
            if away_id not in team_lineup_history:
                team_lineup_history[away_id] = []

            if home_starters:
                team_lineup_history[home_id].append(home_starters)
                if len(team_lineup_history[home_id]) > self.lookback_matches:
                    team_lineup_history[home_id].pop(0)

            if away_starters:
                team_lineup_history[away_id].append(away_starters)
                if len(team_lineup_history[away_id]) > self.lookback_matches:
                    team_lineup_history[away_id].pop(0)

        print(f"Created {len(features_list)} lineup stability features")
        return pd.DataFrame(features_list)

    def _calculate_stability(self, history: List[set], current: set) -> float:
        """
        Calculate lineup stability as average overlap with recent lineups.

        Returns:
            Float from 0 to 1 (1 = same lineup as recent matches)
        """
        if not history or not current:
            return 0.5  # default neutral

        overlaps = []
        for past_lineup in history:
            if past_lineup:
                overlap = len(current & past_lineup) / max(len(current), 1)
                overlaps.append(overlap)

        return sum(overlaps) / len(overlaps) if overlaps else 0.5


class StarPlayerFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on whether star players are playing.

    Star players are defined as top N players by average rating.
    """

    def __init__(self, top_n: int = 3, min_matches: int = 5):
        """
        Args:
            top_n: Number of top players to consider as "stars"
            min_matches: Minimum matches to establish player rating
        """
        self.top_n = top_n
        self.min_matches = min_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create star player features.
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)
        player_stats = data.get('player_stats')
        lineups = data.get('lineups')

        if player_stats is None or player_stats.empty:
            print("No player stats data available, skipping star player features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Calculate running average ratings per player per team
        player_ratings = {}  # {team_id: {player_id: [ratings]}}
        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get star players for each team based on historical ratings
            home_stars = self._get_star_players(player_ratings.get(home_id, {}))
            away_stars = self._get_star_players(player_ratings.get(away_id, {}))

            # Check if stars are starting in this match
            if lineups is not None and not lineups.empty:
                match_starters = lineups[
                    (lineups['fixture_id'] == fixture_id) &
                    (lineups['starting'] == True)
                ]
                home_starters = set(
                    match_starters[match_starters['team_id'] == home_id]['player_id'].tolist()
                )
                away_starters = set(
                    match_starters[match_starters['team_id'] == away_id]['player_id'].tolist()
                )

                home_stars_playing = len(home_stars & home_starters) if home_stars else 0
                away_stars_playing = len(away_stars & away_starters) if away_stars else 0
            else:
                home_stars_playing = self.top_n
                away_stars_playing = self.top_n

            features = {
                'fixture_id': fixture_id,
                'home_stars_playing': home_stars_playing,
                'away_stars_playing': away_stars_playing,
                'home_stars_ratio': home_stars_playing / self.top_n,
                'away_stars_ratio': away_stars_playing / self.top_n,
                'stars_advantage': home_stars_playing - away_stars_playing,
            }
            features_list.append(features)

            # Update player ratings from this match
            match_stats = player_stats[player_stats['fixture_id'] == fixture_id]
            for _, player in match_stats.iterrows():
                team_id = player['team_id']
                player_id = player['player_id']
                rating = player.get('rating')

                if pd.notna(rating) and rating > 0:
                    if team_id not in player_ratings:
                        player_ratings[team_id] = {}
                    if player_id not in player_ratings[team_id]:
                        player_ratings[team_id][player_id] = []
                    player_ratings[team_id][player_id].append(float(rating))

        print(f"Created {len(features_list)} star player features")
        return pd.DataFrame(features_list)

    def _get_star_players(self, team_ratings: Dict) -> set:
        """Get top N players by average rating."""
        if not team_ratings:
            return set()

        # Calculate average rating per player (only if enough matches)
        avg_ratings = []
        for player_id, ratings in team_ratings.items():
            if len(ratings) >= self.min_matches:
                avg_ratings.append((player_id, sum(ratings) / len(ratings)))

        # Sort and get top N
        avg_ratings.sort(key=lambda x: x[1], reverse=True)
        return set(p[0] for p in avg_ratings[:self.top_n])


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


class KeyPlayerAbsenceFeatureEngineer(BaseFeatureEngineer):
    """
    Detects when key players are missing from lineup.

    Key players defined as those who played most minutes in recent matches.
    """

    def __init__(self, top_n: int = 5, lookback_matches: int = 5):
        """
        Args:
            top_n: Number of top players by minutes to consider "key"
            lookback_matches: Matches to look back for establishing key players
        """
        self.top_n = top_n
        self.lookback_matches = lookback_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate key player absence features.
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)
        player_stats = data.get('player_stats')
        lineups = data.get('lineups')

        if player_stats is None or lineups is None:
            print("Missing player stats or lineups, skipping key player absence features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Track minutes per player per team
        player_minutes = {}  # {team_id: {player_id: total_minutes}}
        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get key players based on historical minutes
            home_key = self._get_key_players(player_minutes.get(home_id, {}))
            away_key = self._get_key_players(player_minutes.get(away_id, {}))

            # Check who's in starting lineup
            match_starters = lineups[
                (lineups['fixture_id'] == fixture_id) &
                (lineups['starting'] == True)
            ]
            home_starters = set(
                match_starters[match_starters['team_id'] == home_id]['player_id'].dropna().tolist()
            )
            away_starters = set(
                match_starters[match_starters['team_id'] == away_id]['player_id'].dropna().tolist()
            )

            # Count missing key players
            home_missing = len(home_key - home_starters) if home_key else 0
            away_missing = len(away_key - away_starters) if away_key else 0

            features = {
                'fixture_id': fixture_id,
                'home_key_players_missing': home_missing,
                'away_key_players_missing': away_missing,
                'key_player_advantage': away_missing - home_missing,  # positive = home advantage
            }
            features_list.append(features)

            # Update player minutes from this match
            match_stats = player_stats[player_stats['fixture_id'] == fixture_id]
            for _, player in match_stats.iterrows():
                team_id = player['team_id']
                player_id = player['player_id']
                minutes = player.get('minutes', 0)

                if pd.notna(minutes) and minutes > 0:
                    if team_id not in player_minutes:
                        player_minutes[team_id] = {}
                    if player_id not in player_minutes[team_id]:
                        player_minutes[team_id][player_id] = 0
                    player_minutes[team_id][player_id] += minutes

        print(f"Created {len(features_list)} key player absence features")
        return pd.DataFrame(features_list)

    def _get_key_players(self, team_minutes: Dict) -> set:
        """Get top N players by total minutes played."""
        if not team_minutes:
            return set()

        sorted_players = sorted(
            team_minutes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return set(p[0] for p in sorted_players[:self.top_n])


class DisciplineFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on team discipline (cards).

    Disciplinary record can indicate:
    - Aggressive play style
    - Likelihood of red cards
    - Impact on match flow
    """

    def __init__(self, lookback_matches: int = 5):
        """
        Args:
            lookback_matches: Number of recent matches for discipline calculation
        """
        self.lookback_matches = lookback_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate discipline features from events data.
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)
        events = data.get('events')

        if events is None or events.empty:
            print("No events data available, skipping discipline features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Filter to card events only
        cards = events[events['type'] == 'Card'].copy()

        # Track card history per team
        team_cards_history = {}  # {team_id: [(yellows, reds)]}
        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get historical discipline
            home_history = team_cards_history.get(home_id, [])
            away_history = team_cards_history.get(away_id, [])

            home_avg_yellows = sum(h[0] for h in home_history) / len(home_history) if home_history else 1.5
            home_avg_reds = sum(h[1] for h in home_history) / len(home_history) if home_history else 0.05
            away_avg_yellows = sum(h[0] for h in away_history) / len(away_history) if away_history else 1.5
            away_avg_reds = sum(h[1] for h in away_history) / len(away_history) if away_history else 0.05

            features = {
                'fixture_id': fixture_id,
                'home_avg_yellows': home_avg_yellows,
                'away_avg_yellows': away_avg_yellows,
                'home_avg_reds': home_avg_reds,
                'away_avg_reds': away_avg_reds,
                'discipline_diff': (home_avg_yellows + home_avg_reds * 3) - (away_avg_yellows + away_avg_reds * 3),
            }
            features_list.append(features)

            # Update discipline from this match
            match_cards = cards[cards['fixture_id'] == fixture_id]

            for team_id in [home_id, away_id]:
                team_cards = match_cards[match_cards['team_id'] == team_id]
                yellows = len(team_cards[team_cards['detail'] == 'Yellow Card'])
                reds = len(team_cards[team_cards['detail'] == 'Red Card'])

                if team_id not in team_cards_history:
                    team_cards_history[team_id] = []
                team_cards_history[team_id].append((yellows, reds))

                if len(team_cards_history[team_id]) > self.lookback_matches:
                    team_cards_history[team_id].pop(0)

        print(f"Created {len(features_list)} discipline features")
        return pd.DataFrame(features_list)


class GoalTimingFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on when teams score/concede goals.

    Timing patterns:
    - Early goals (< 30 min): Fast starters
    - Late goals (> 75 min): Strong finishers
    - First half vs Second half distribution
    """

    def __init__(self, lookback_matches: int = 10):
        """
        Args:
            lookback_matches: Number of recent matches for timing calculation
        """
        self.lookback_matches = lookback_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate goal timing features from events data.
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)
        events = data.get('events')

        if events is None or events.empty:
            print("No events data available, skipping goal timing features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Filter to goal events only
        goals = events[events['type'] == 'Goal'].copy()

        # Track goal timing history per team
        team_goal_timing = {}  # {team_id: {'early': N, 'late': N, '1h': N, '2h': N, 'total': N}}
        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get historical timing patterns
            home_timing = team_goal_timing.get(home_id, {})
            away_timing = team_goal_timing.get(away_id, {})

            features = {
                'fixture_id': fixture_id,
                'home_early_goal_rate': self._get_rate(home_timing, 'early'),
                'home_late_goal_rate': self._get_rate(home_timing, 'late'),
                'home_first_half_rate': self._get_rate(home_timing, '1h'),
                'away_early_goal_rate': self._get_rate(away_timing, 'early'),
                'away_late_goal_rate': self._get_rate(away_timing, 'late'),
                'away_first_half_rate': self._get_rate(away_timing, '1h'),
            }
            features_list.append(features)

            # Update timing from this match
            match_goals = goals[goals['fixture_id'] == fixture_id]

            for team_id in [home_id, away_id]:
                team_goals = match_goals[match_goals['team_id'] == team_id]

                if team_id not in team_goal_timing:
                    team_goal_timing[team_id] = {
                        'early': 0, 'late': 0, '1h': 0, '2h': 0, 'total': 0
                    }

                for _, goal in team_goals.iterrows():
                    time = goal.get('time_elapsed', 45)
                    team_goal_timing[team_id]['total'] += 1

                    if time <= 30:
                        team_goal_timing[team_id]['early'] += 1
                    if time >= 75:
                        team_goal_timing[team_id]['late'] += 1
                    if time <= 45:
                        team_goal_timing[team_id]['1h'] += 1
                    else:
                        team_goal_timing[team_id]['2h'] += 1

        print(f"Created {len(features_list)} goal timing features")
        return pd.DataFrame(features_list)

    def _get_rate(self, timing: Dict, key: str) -> float:
        """Calculate rate of goals in given period."""
        if not timing or timing.get('total', 0) == 0:
            return 0.33  # default
        return timing.get(key, 0) / timing['total']


class SeasonPhaseFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on season phase.

    Different phases have different dynamics:
    - Start (rounds 1-10): Teams finding form
    - Middle (rounds 11-28): Settled patterns
    - End (rounds 29-38): Pressure, motivation varies
    """

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate season phase features.
        """
        matches = data['matches'].copy()

        # Extract round number
        matches['round_num'] = matches['round'].str.extract(r'(\d+)').astype(float)

        features_list = []

        for idx, match in matches.iterrows():
            round_num = match.get('round_num', 19)  # default mid-season

            # Determine phase
            if round_num <= 10:
                phase = 'start'
                phase_encoded = 0
            elif round_num <= 28:
                phase = 'middle'
                phase_encoded = 1
            else:
                phase = 'end'
                phase_encoded = 2

            features = {
                'fixture_id': match['fixture_id'],
                'season_phase': phase_encoded,
                'is_season_start': 1 if phase == 'start' else 0,
                'is_season_end': 1 if phase == 'end' else 0,
                'round_number': round_num if pd.notna(round_num) else 19,
            }
            features_list.append(features)

        print(f"Created {len(features_list)} season phase features")
        return pd.DataFrame(features_list)


class DerbyFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features for local derby matches.

    Derbies are high-stakes matches where:
    - Form often goes out the window
    - Home advantage is amplified
    - Results are less predictable
    """

    # Premier League derbies
    DERBY_PAIRS = {
        # Manchester Derby
        ('Manchester United', 'Manchester City'),
        ('Manchester City', 'Manchester United'),
        # North London Derby
        ('Arsenal', 'Tottenham'),
        ('Tottenham', 'Arsenal'),
        # Merseyside Derby
        ('Liverpool', 'Everton'),
        ('Everton', 'Liverpool'),
        # North West Derby
        ('Liverpool', 'Manchester United'),
        ('Manchester United', 'Liverpool'),
        # London Derbies
        ('Chelsea', 'Arsenal'),
        ('Arsenal', 'Chelsea'),
        ('Chelsea', 'Tottenham'),
        ('Tottenham', 'Chelsea'),
        ('West Ham', 'Tottenham'),
        ('Tottenham', 'West Ham'),
        ('Chelsea', 'West Ham'),
        ('West Ham', 'Chelsea'),
        # Yorkshire
        ('Leeds', 'Sheffield'),
        ('Sheffield', 'Leeds'),
        # Midlands
        ('Aston Villa', 'Birmingham'),
        ('Birmingham', 'Aston Villa'),
        ('Aston Villa', 'Wolves'),
        ('Wolves', 'Aston Villa'),
        # Others
        ('Newcastle', 'Sunderland'),
        ('Sunderland', 'Newcastle'),
        ('Brighton', 'Crystal Palace'),
        ('Crystal Palace', 'Brighton'),
    }

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate derby features.
        """
        matches = data['matches'].copy()
        features_list = []

        for idx, match in matches.iterrows():
            home_name = match['home_team_name']
            away_name = match['away_team_name']

            # Check if derby
            is_derby = (home_name, away_name) in self.DERBY_PAIRS

            # Check for same-city teams (simplified)
            same_city = self._same_city(home_name, away_name)

            features = {
                'fixture_id': match['fixture_id'],
                'is_derby': 1 if is_derby else 0,
                'is_same_city': 1 if same_city else 0,
                'is_rivalry': 1 if (is_derby or same_city) else 0,
            }
            features_list.append(features)

        derby_count = sum(1 for f in features_list if f['is_derby'] == 1)
        print(f"Created {len(features_list)} derby features ({derby_count} derbies found)")
        return pd.DataFrame(features_list)

    def _same_city(self, home: str, away: str) -> bool:
        """Check if teams are from the same city."""
        city_teams = {
            'London': ['Arsenal', 'Chelsea', 'Tottenham', 'West Ham', 'Crystal Palace',
                       'Fulham', 'Brentford', 'QPR', 'Charlton', 'Millwall'],
            'Manchester': ['Manchester United', 'Manchester City'],
            'Liverpool': ['Liverpool', 'Everton'],
            'Birmingham': ['Aston Villa', 'Birmingham', 'West Brom', 'Wolves'],
            'Sheffield': ['Sheffield United', 'Sheffield Wednesday'],
        }

        for city, teams in city_teams.items():
            if home in teams and away in teams:
                return True
        return False
