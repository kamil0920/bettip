"""Feature engineering - Lineup and player-related features."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer


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


