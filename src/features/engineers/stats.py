"""Feature engineering - Team statistics and goal features."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer


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


