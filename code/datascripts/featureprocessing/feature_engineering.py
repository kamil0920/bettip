from typing import Dict

import numpy as np
import pandas as pd

from interfaces import IFeatureEngineer


class BaseFeatureEngineer(IFeatureEngineer):

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Template method pattern"""
        raise NotImplementedError


class TeamFormFeatureEngineer(BaseFeatureEngineer):
    """Create features related to team form"""

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

            home_form = self._calculate_team_form(
                matches, home_id, match_date, self.n_matches
            )
            away_form = self._calculate_team_form(
                matches, away_id, match_date, self.n_matches
            )

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

        print(f"✓ Created {len(features_list)} team features form (last {self.n_matches} matches)")
        return pd.DataFrame(features_list)

    def _calculate_team_form(self, matches: pd.DataFrame, team_id: int,
                            current_date: pd.Timestamp, n: int) -> Dict:
        """
        Calculate team form based on N last matches

        Args:
            matches: DataFrame with matches
            team_id: team id
            current_date: matches date
            n: number of matches

        Returns:
            Dict with stats
        """
        past_matches = matches[matches['date'] < current_date]

        team_matches = past_matches[
            (past_matches['home_team_id'] == team_id) |
            (past_matches['away_team_id'] == team_id)
        ].tail(n)

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
    """Create aggregated features from players stats"""

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregates player stats to team lvl

        Args:
            data: dict {name:DataFrame}

        Returns:
            DataFrame with aggregated stats
        """
        player_stats = data['player_stats'].copy()

        team_stats = player_stats.groupby(['fixture_id', 'team_id']).agg({
            'rating': 'mean',
            'goals': 'sum',
            'assists': 'sum',
            'shots_total': 'sum',
            'shots_on': 'sum',
            'passes_total': 'sum',
            'passes_key': 'sum',
            'passes_accuracy': 'mean',
            'tackles_total': 'sum',
            'fouls_committed': 'sum',
            'yellow_cards': 'sum',
            'red_cards': 'sum'
        }).reset_index()

        team_stats.columns = ['fixture_id', 'team_id', 'avg_rating', 'total_goals',
                              'total_assists', 'total_shots', 'total_shots_on',
                              'total_passes', 'total_key_passes', 'avg_pass_accuracy',
                              'total_tackles', 'total_fouls', 'total_yellows', 'total_reds']

        print(f"✓ Created {len(team_stats)} aggregated team stats")
        return team_stats


class MatchOutcomeFeatureEngineer(BaseFeatureEngineer):
    """Creates target variables for prediction"""

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
        target_cols = ['fixture_id', 'match_result', 'home_win', 'draw','away_win', 'total_goals', 'goal_difference']

        print(f"✓ Created target variables")
        return matches[target_cols]


class HeadToHeadFeatureEngineer(BaseFeatureEngineer):
    """Creates features related to history of face-to-face matches"""

    def __init__(self, n_h2h: int = 3):
        """
        Args:
            n_h2h: how many matches take
        """
        self.n_h2h = n_h2h

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Creates features head-to-head

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

        print(f"✓ Created {len(features_list)} features head-to-head (last {self.n_h2h} matches)")
        return pd.DataFrame(features_list)