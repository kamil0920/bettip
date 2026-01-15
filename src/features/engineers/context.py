"""Feature engineering - Match context features (rest days, position, importance)."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer


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

        matches = calculate_goal_diff_form(matches)

        target_cols = [
            'fixture_id', 'match_result', 'home_win', 'draw',
            'away_win', 'total_goals', 'goal_difference', 'gd_form_diff'
        ]

        print(f"Created target variables")
        return matches[target_cols]

def calculate_goal_diff_form(df):
    df['match_gd'] = df['ft_home'] - df['ft_away']

    home_stats = df[['date', 'home_team', 'match_gd']].rename(
        columns={'home_team': 'team', 'match_gd': 'gd'}
    )
    away_stats = df[['date', 'away_team', 'match_gd']].rename(
        columns={'away_team': 'team', 'match_gd': 'gd'}
    )
    away_stats['gd'] = -away_stats['gd']

    all_stats = pd.concat([home_stats, away_stats]).sort_values(['team', 'date'])

    all_stats['avg_gd_last_5'] = all_stats.groupby('team')['gd'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

    df = df.merge(
        all_stats[['date', 'team', 'avg_gd_last_5']].rename(columns={'avg_gd_last_5': 'home_gd_form'}),
        left_on=['date', 'home_team'],
        right_on=['date', 'team'],
        how='left'
    ).drop(columns=['team'])

    df = df.merge(
        all_stats[['date', 'team', 'avg_gd_last_5']].rename(columns={'avg_gd_last_5': 'away_gd_form'}),
        left_on=['date', 'away_team'],
        right_on=['date', 'team'],
        how='left'
    ).drop(columns=['team'])

    df['gd_form_diff'] = df['home_gd_form'] - df['away_gd_form']

    df = df.drop(columns=['match_gd'])

    return df


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



class MatchImportanceFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on match importance.

    Match importance considers:
    - Title race (close to 1st place)
    - Champions League race (close to 4th place)
    - Relegation battle (close to 18th place)
    - Mid-table (no significant stakes)

    High-stakes matches often have different dynamics than
    matches with nothing to play for.
    """

    def __init__(
        self,
        cl_spots: int = 4,
        europa_spots: int = 6,
        relegation_spots: int = 3,
        title_threshold_pts: int = 9,
        cl_threshold_pts: int = 6,
        relegation_threshold_pts: int = 6,
    ):
        """
        Initialize with league-specific parameters.

        Args:
            cl_spots: Number of CL qualification spots (default 4)
            europa_spots: Number of Europa League spots (default 6)
            relegation_spots: Number of relegation spots (default 3)
            title_threshold_pts: Points gap to consider in title race
            cl_threshold_pts: Points gap to consider in CL race
            relegation_threshold_pts: Points gap from safety
        """
        self.cl_spots = cl_spots
        self.europa_spots = europa_spots
        self.relegation_spots = relegation_spots
        self.title_threshold_pts = title_threshold_pts
        self.cl_threshold_pts = cl_threshold_pts
        self.relegation_threshold_pts = relegation_threshold_pts

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate match importance features.

        Uses running standings to determine each team's position
        and distance to key thresholds at the time of the match.
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())
        num_teams = len(all_teams)

        relegation_line = num_teams - self.relegation_spots + 1  # e.g., 18th in 20-team league

        team_stats = {
            team_id: {
                'points': 0,
                'played': 0,
                'goals_for': 0,
                'goals_against': 0,
            }
            for team_id in all_teams
        }

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            standings = self._calculate_full_standings(team_stats)

            home_data = standings.get(home_id, self._default_position_data(num_teams // 2))
            away_data = standings.get(away_id, self._default_position_data(num_teams // 2))

            features = self._calculate_importance_features(
                match['fixture_id'],
                home_data,
                away_data,
                standings,
                relegation_line,
            )

            features_list.append(features)

            home_goals = match['ft_home']
            away_goals = match['ft_away']

            self._update_team_stats(team_stats, home_id, home_goals, away_goals)
            self._update_team_stats(team_stats, away_id, away_goals, home_goals)

        print(f"Created {len(features_list)} match importance features")
        return pd.DataFrame(features_list)

    def _default_position_data(self, default_pos: int) -> Dict:
        """Return default position data for teams with no history."""
        return {
            'position': default_pos,
            'points': 0,
            'played': 0,
            'gd': 0,
        }

    def _update_team_stats(self, team_stats: Dict, team_id: int, goals_for: int, goals_against: int):
        """Update team stats after a match."""
        stats = team_stats[team_id]
        stats['played'] += 1
        stats['goals_for'] += goals_for
        stats['goals_against'] += goals_against

        if goals_for > goals_against:
            stats['points'] += 3
        elif goals_for == goals_against:
            stats['points'] += 1

    def _calculate_full_standings(self, team_stats: Dict) -> Dict:
        """
        Calculate current league standings with points.

        Returns dict with position, points, played, gd for each team.
        """
        standings_data = []

        for team_id, stats in team_stats.items():
            standings_data.append({
                'team_id': team_id,
                'points': stats['points'],
                'gd': stats['goals_for'] - stats['goals_against'],
                'gf': stats['goals_for'],
                'played': stats['played'],
            })

        # Sort by points, then goal difference, then goals for
        standings_data.sort(key=lambda x: (-x['points'], -x['gd'], -x['gf']))

        result = {}
        for pos, team in enumerate(standings_data, 1):
            result[team['team_id']] = {
                'position': pos,
                'points': team['points'],
                'played': team['played'],
                'gd': team['gd'],
            }

        return result

    def _calculate_importance_features(
        self,
        fixture_id: int,
        home_data: Dict,
        away_data: Dict,
        standings: Dict,
        relegation_line: int,
    ) -> Dict:
        """Calculate importance features for a single match."""
        all_positions = list(standings.values())
        if not all_positions or all(p['played'] == 0 for p in all_positions):
            return self._default_importance_features(fixture_id)

        sorted_standings = sorted(all_positions, key=lambda x: -x['points'])

        leader_pts = sorted_standings[0]['points'] if sorted_standings else 0
        cl_line_pts = sorted_standings[self.cl_spots - 1]['points'] if len(sorted_standings) >= self.cl_spots else 0
        europa_line_pts = sorted_standings[self.europa_spots - 1]['points'] if len(sorted_standings) >= self.europa_spots else 0
        safety_line_pts = sorted_standings[relegation_line - 2]['points'] if len(sorted_standings) >= relegation_line - 1 else 0

        home_pos = home_data['position']
        away_pos = away_data['position']
        home_pts = home_data['points']
        away_pts = away_data['points']

        home_pts_to_leader = leader_pts - home_pts
        away_pts_to_leader = leader_pts - away_pts
        home_pts_to_cl = cl_line_pts - home_pts if home_pos > self.cl_spots else 0
        away_pts_to_cl = cl_line_pts - away_pts if away_pos > self.cl_spots else 0
        home_pts_to_safety = home_pts - safety_line_pts if home_pos >= relegation_line else 0
        away_pts_to_safety = away_pts - safety_line_pts if away_pos >= relegation_line else 0

        home_in_title_race = home_pts_to_leader <= self.title_threshold_pts and home_data['played'] >= 5
        away_in_title_race = away_pts_to_leader <= self.title_threshold_pts and away_data['played'] >= 5
        home_in_cl_race = home_pos <= self.cl_spots + 2 and home_pts_to_cl <= self.cl_threshold_pts
        away_in_cl_race = away_pos <= self.cl_spots + 2 and away_pts_to_cl <= self.cl_threshold_pts
        home_in_relegation_zone = home_pos >= relegation_line
        away_in_relegation_zone = away_pos >= relegation_line
        home_relegation_battle = home_pos >= relegation_line - 3 and home_pts_to_safety <= self.relegation_threshold_pts
        away_relegation_battle = away_pos >= relegation_line - 3 and away_pts_to_safety <= self.relegation_threshold_pts

        home_importance = self._calculate_team_importance(
            home_in_title_race, home_in_cl_race, home_in_relegation_zone, home_relegation_battle
        )
        away_importance = self._calculate_team_importance(
            away_in_title_race, away_in_cl_race, away_in_relegation_zone, away_relegation_battle
        )
        match_importance = (home_importance + away_importance) / 2

        return {
            'fixture_id': fixture_id,
            'home_pts_to_leader': home_pts_to_leader,
            'away_pts_to_leader': away_pts_to_leader,
            'home_pts_to_cl': home_pts_to_cl,
            'away_pts_to_cl': away_pts_to_cl,
            'home_pts_to_safety': home_pts_to_safety,
            'away_pts_to_safety': away_pts_to_safety,
            'home_in_title_race': 1 if home_in_title_race else 0,
            'away_in_title_race': 1 if away_in_title_race else 0,
            'home_in_cl_race': 1 if home_in_cl_race else 0,
            'away_in_cl_race': 1 if away_in_cl_race else 0,
            'home_in_relegation_zone': 1 if home_in_relegation_zone else 0,
            'away_in_relegation_zone': 1 if away_in_relegation_zone else 0,
            'home_relegation_battle': 1 if home_relegation_battle else 0,
            'away_relegation_battle': 1 if away_relegation_battle else 0,
            'home_importance': home_importance,
            'away_importance': away_importance,
            'match_importance': match_importance,
            'importance_diff': home_importance - away_importance,
            'is_title_decider': 1 if (home_in_title_race and away_in_title_race) else 0,
            'is_relegation_clash': 1 if (home_relegation_battle and away_relegation_battle) else 0,
            'is_cl_race_match': 1 if (home_in_cl_race and away_in_cl_race) else 0,
            'one_team_nothing_to_play': 1 if (home_importance < 0.2 or away_importance < 0.2) else 0,
        }

    def _default_importance_features(self, fixture_id: int) -> Dict:
        """Return default importance features for season start."""
        return {
            'fixture_id': fixture_id,
            'home_pts_to_leader': 0,
            'away_pts_to_leader': 0,
            'home_pts_to_cl': 0,
            'away_pts_to_cl': 0,
            'home_pts_to_safety': 0,
            'away_pts_to_safety': 0,
            'home_in_title_race': 0,
            'away_in_title_race': 0,
            'home_in_cl_race': 0,
            'away_in_cl_race': 0,
            'home_in_relegation_zone': 0,
            'away_in_relegation_zone': 0,
            'home_relegation_battle': 0,
            'away_relegation_battle': 0,
            'home_importance': 0.5,
            'away_importance': 0.5,
            'match_importance': 0.5,
            'importance_diff': 0,
            'is_title_decider': 0,
            'is_relegation_clash': 0,
            'is_cl_race_match': 0,
            'one_team_nothing_to_play': 0,
        }

    def _calculate_team_importance(
        self,
        in_title_race: bool,
        in_cl_race: bool,
        in_relegation_zone: bool,
        in_relegation_battle: bool,
    ) -> float:
        """Calculate importance score for a team (0-1 scale)."""
        importance = 0.3

        if in_title_race:
            importance += 0.4
        elif in_cl_race:
            importance += 0.25

        if in_relegation_zone:
            importance += 0.35
        elif in_relegation_battle:
            importance += 0.2

        return min(importance, 1.0)


