"""Feature engineering implementations."""
from typing import Dict

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
