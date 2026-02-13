"""Feature engineering - Lineup and player-related features."""
from collections import defaultdict
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
    Creates features based on coaches/managers using walk-forward tracking.

    Features (6):
    - home/away_coach_change_recent: 1 if different coach from previous match
    - home/away_coach_tenure: consecutive matches with current coach (normalized)
    - coach_stability_diff: home_tenure - away_tenure (stability advantage)
    - coach_change_either: 1 if either team has a new coach
    """

    def __init__(self, lookback_matches: int = 5):
        """
        Args:
            lookback_matches: Number of matches to look back for coach history
        """
        self.lookback_matches = lookback_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate coach features using walk-forward approach."""
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)
        lineups = data.get('lineups')

        if lineups is None or lineups.empty:
            print("No lineups data available, skipping coach features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Build coach lookup: {(fixture_id, team_id): coach_name}
        coach_lookup: Dict[tuple, Optional[str]] = {}
        has_coach_data = 'coach_name' in lineups.columns

        if has_coach_data:
            coach_rows = lineups[lineups['coach_name'].notna()].drop_duplicates(
                subset=['fixture_id', 'team_id'], keep='first'
            )
            for _, row in coach_rows.iterrows():
                key = (row['fixture_id'], row['team_id'])
                coach_lookup[key] = str(row['coach_name'])

        # Walk-forward: track coach history per team
        # {team_id: list of coach_name strings, most recent last}
        team_coach_history: Dict[int, List[str]] = defaultdict(list)
        default_tenure = self.lookback_matches / 2.0

        features_list = []

        for _, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Current match coaches
            home_coach = coach_lookup.get((fixture_id, home_id))
            away_coach = coach_lookup.get((fixture_id, away_id))

            # Calculate features from history BEFORE this match
            home_feats = self._coach_features(home_id, home_coach, team_coach_history)
            away_feats = self._coach_features(away_id, away_coach, team_coach_history)

            features = {
                'fixture_id': fixture_id,
                'home_coach_change_recent': home_feats['change'],
                'away_coach_change_recent': away_feats['change'],
                'home_coach_tenure': home_feats['tenure'],
                'away_coach_tenure': away_feats['tenure'],
                'coach_stability_diff': home_feats['tenure'] - away_feats['tenure'],
                'coach_change_either': max(home_feats['change'], away_feats['change']),
            }
            features_list.append(features)

            # Update history AFTER feature calculation
            if home_coach is not None:
                team_coach_history[home_id].append(home_coach)
                if len(team_coach_history[home_id]) > self.lookback_matches:
                    team_coach_history[home_id].pop(0)
            if away_coach is not None:
                team_coach_history[away_id].append(away_coach)
                if len(team_coach_history[away_id]) > self.lookback_matches:
                    team_coach_history[away_id].pop(0)

        n_with_data = sum(1 for f in features_list if f['home_coach_tenure'] != default_tenure)
        print(f"Created {len(features_list)} coach features ({n_with_data} with coach data)")
        return pd.DataFrame(features_list)

    def _coach_features(
        self,
        team_id: int,
        current_coach: Optional[str],
        team_coach_history: Dict[int, List[str]],
    ) -> Dict[str, float]:
        """Calculate coach features from historical data only."""
        history = team_coach_history.get(team_id, [])
        default_tenure = self.lookback_matches / 2.0

        # No history or no current coach data → neutral defaults
        if not history or current_coach is None:
            return {'change': 0.0, 'tenure': default_tenure}

        # Coach change: different from most recent match
        last_coach = history[-1]
        change = 1.0 if current_coach != last_coach else 0.0

        # Tenure: consecutive matches with current coach (count from end of history)
        tenure = 0.0
        for past_coach in reversed(history):
            if past_coach == current_coach:
                tenure += 1.0
            else:
                break

        return {'change': change, 'tenure': tenure}



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


class GoalkeeperChangeFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on goalkeeper changes.

    GK is the highest-impact single position change. A backup GK vs the
    regular starter represents a massive quality drop not captured by
    engineers that treat all positions equally.

    Features (6):
    - home/away_gk_is_regular: 1.0 if GK started majority of last N matches
    - home/away_gk_experience: GK's starts in last N team matches / N
    - home_gk_rating_avg: GK's rolling avg rating from player_stats
    - gk_change_advantage: away_gk_changed - home_gk_changed
    """

    def __init__(self, lookback_matches: int = 5, rating_lookback: int = 10):
        """
        Args:
            lookback_matches: Matches to determine "regular" GK
            rating_lookback: Matches for GK rating average
        """
        self.lookback_matches = lookback_matches
        self.rating_lookback = rating_lookback

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate goalkeeper change features using walk-forward approach."""
        matches = data['matches'].copy()
        assert matches['date'].is_monotonic_increasing or matches.equals(
            matches.sort_values('date')
        ), "Matches must be sorted by date for walk-forward features"
        matches = matches.sort_values('date').reset_index(drop=True)
        lineups = data.get('lineups')
        player_stats = data.get('player_stats')

        if lineups is None or lineups.empty:
            print("No lineups data available, skipping goalkeeper change features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Pre-filter: starting goalkeepers only
        starting_gks = lineups[
            (lineups['starting'] == True) & (lineups['pos'] == 'G')
        ][['fixture_id', 'team_id', 'player_id']].copy()

        # Build player rating lookup from player_stats
        gk_ratings: Dict[int, Dict[int, float]] = {}  # {fixture_id: {player_id: rating}}
        if player_stats is not None and not player_stats.empty:
            rated = player_stats[player_stats['rating'].notna() & (player_stats['rating'] > 0)]
            for _, row in rated.iterrows():
                fid = row['fixture_id']
                pid = row['player_id']
                if fid not in gk_ratings:
                    gk_ratings[fid] = {}
                gk_ratings[fid][pid] = float(row['rating'])

        # Walk-forward: track GK history per team
        # {team_id: list of (gk_player_id, fixture_id)} most recent last
        team_gk_history: Dict[int, List[tuple]] = defaultdict(list)
        # {player_id: list of ratings} for EMA
        gk_rating_history: Dict[int, List[float]] = defaultdict(list)

        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Current match GKs
            match_gks = starting_gks[starting_gks['fixture_id'] == fixture_id]
            home_gk_row = match_gks[match_gks['team_id'] == home_id]
            away_gk_row = match_gks[match_gks['team_id'] == away_id]
            home_gk = home_gk_row['player_id'].values[0] if len(home_gk_row) > 0 else None
            away_gk = away_gk_row['player_id'].values[0] if len(away_gk_row) > 0 else None

            # Calculate features from history (before this match)
            home_feats = self._gk_features(
                home_id, home_gk, team_gk_history, gk_rating_history
            )
            away_feats = self._gk_features(
                away_id, away_gk, team_gk_history, gk_rating_history
            )

            home_changed = 0.0 if home_feats['is_regular'] else 1.0
            away_changed = 0.0 if away_feats['is_regular'] else 1.0

            features = {
                'fixture_id': fixture_id,
                'home_gk_is_regular': home_feats['is_regular'],
                'away_gk_is_regular': away_feats['is_regular'],
                'home_gk_experience': home_feats['experience'],
                'away_gk_experience': away_feats['experience'],
                'home_gk_rating_avg': home_feats['rating_avg'],
                'gk_change_advantage': away_changed - home_changed,
            }
            features_list.append(features)

            # Update history AFTER feature calculation
            if home_gk is not None:
                team_gk_history[home_id].append(home_gk)
                if len(team_gk_history[home_id]) > self.rating_lookback:
                    team_gk_history[home_id].pop(0)
                # Update GK rating
                if fixture_id in gk_ratings and home_gk in gk_ratings[fixture_id]:
                    gk_rating_history[home_gk].append(gk_ratings[fixture_id][home_gk])

            if away_gk is not None:
                team_gk_history[away_id].append(away_gk)
                if len(team_gk_history[away_id]) > self.rating_lookback:
                    team_gk_history[away_id].pop(0)
                if fixture_id in gk_ratings and away_gk in gk_ratings[fixture_id]:
                    gk_rating_history[away_gk].append(gk_ratings[fixture_id][away_gk])

        print(f"Created {len(features_list)} goalkeeper change features")
        return pd.DataFrame(features_list)

    def _gk_features(
        self,
        team_id: int,
        current_gk: Optional[int],
        team_gk_history: Dict[int, List],
        gk_rating_history: Dict[int, List[float]],
    ) -> Dict[str, float]:
        """Calculate GK features from historical data only."""
        history = team_gk_history.get(team_id, [])

        if not history or current_gk is None:
            return {'is_regular': 0.5, 'experience': 0.5, 'rating_avg': 6.5}

        recent = history[-self.lookback_matches:]
        starts_in_recent = sum(1 for gk in recent if gk == current_gk)
        is_regular = 1.0 if starts_in_recent > len(recent) / 2 else 0.0
        experience = starts_in_recent / len(recent)

        ratings = gk_rating_history.get(current_gk, [])
        rating_avg = sum(ratings[-self.rating_lookback:]) / len(ratings[-self.rating_lookback:]) if ratings else 6.5

        return {'is_regular': is_regular, 'experience': experience, 'rating_avg': rating_avg}


class SquadQualityFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on squad quality using historical player ratings.

    Existing features count missing players but don't quantify quality impact.
    Missing a 7.5-rated player != missing a 6.2-rated player.

    Features (6):
    - home/away_xi_avg_rating: Mean historical rating of starting XI
    - home/away_missing_rating: Sum of avg ratings of expected starters not in XI
    - xi_rating_advantage: home - away starting XI avg rating
    - missing_rating_disadvantage: home_missing - away_missing
    """

    def __init__(self, lookback_matches: int = 10, ema_alpha: float = 0.3):
        """
        Args:
            lookback_matches: Matches to determine "expected starters"
            ema_alpha: EMA smoothing factor for player ratings
        """
        self.lookback_matches = lookback_matches
        self.ema_alpha = ema_alpha

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate squad quality features using walk-forward approach."""
        matches = data['matches'].copy()
        assert matches['date'].is_monotonic_increasing or matches.equals(
            matches.sort_values('date')
        ), "Matches must be sorted by date for walk-forward features"
        matches = matches.sort_values('date').reset_index(drop=True)
        lineups = data.get('lineups')
        player_stats = data.get('player_stats')

        if lineups is None or lineups.empty or player_stats is None or player_stats.empty:
            print("Missing lineups or player_stats, skipping squad quality features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Pre-filter starting XI
        starters = lineups[lineups['starting'] == True][
            ['fixture_id', 'team_id', 'player_id']
        ].copy()

        # Build rating lookup: {fixture_id: {player_id: rating}}
        rating_lookup: Dict[int, Dict[int, float]] = {}
        rated = player_stats[player_stats['rating'].notna() & (player_stats['rating'] > 0)]
        for _, row in rated.iterrows():
            fid = row['fixture_id']
            if fid not in rating_lookup:
                rating_lookup[fid] = {}
            rating_lookup[fid][row['player_id']] = float(row['rating'])

        # Walk-forward state
        # {team_id: {player_id: ema_rating}}
        player_ema: Dict[int, Dict[int, float]] = defaultdict(dict)
        # {team_id: {player_id: appearance_count}} over recent window
        player_appearances: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # {team_id: list of fixture_ids} for windowed appearance tracking
        team_fixture_history: Dict[int, List[int]] = defaultdict(list)
        # {(team_id, fixture_id): set of player_ids} for removing old appearances
        team_fixture_starters: Dict[tuple, set] = {}

        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Current starters
            match_starters = starters[starters['fixture_id'] == fixture_id]
            home_xi = set(match_starters[match_starters['team_id'] == home_id]['player_id'].tolist())
            away_xi = set(match_starters[match_starters['team_id'] == away_id]['player_id'].tolist())

            # Calculate features from historical data
            home_feats = self._squad_features(home_id, home_xi, player_ema, player_appearances)
            away_feats = self._squad_features(away_id, away_xi, player_ema, player_appearances)

            features = {
                'fixture_id': fixture_id,
                'home_xi_avg_rating': home_feats['xi_avg_rating'],
                'away_xi_avg_rating': away_feats['xi_avg_rating'],
                'home_missing_rating': home_feats['missing_rating'],
                'away_missing_rating': away_feats['missing_rating'],
                'xi_rating_advantage': home_feats['xi_avg_rating'] - away_feats['xi_avg_rating'],
                'missing_rating_disadvantage': home_feats['missing_rating'] - away_feats['missing_rating'],
            }
            features_list.append(features)

            # Update state AFTER feature calculation
            for team_id, xi in [(home_id, home_xi), (away_id, away_xi)]:
                # Update player EMA ratings
                if fixture_id in rating_lookup:
                    for pid in xi:
                        if pid in rating_lookup[fixture_id]:
                            new_rating = rating_lookup[fixture_id][pid]
                            if pid in player_ema[team_id]:
                                old = player_ema[team_id][pid]
                                player_ema[team_id][pid] = self.ema_alpha * new_rating + (1 - self.ema_alpha) * old
                            else:
                                player_ema[team_id][pid] = new_rating

                # Update appearance tracking with sliding window
                team_fixture_starters[(team_id, fixture_id)] = xi
                team_fixture_history[team_id].append(fixture_id)
                for pid in xi:
                    player_appearances[team_id][pid] += 1

                # Remove oldest fixture if beyond lookback
                if len(team_fixture_history[team_id]) > self.lookback_matches:
                    old_fid = team_fixture_history[team_id].pop(0)
                    old_key = (team_id, old_fid)
                    if old_key in team_fixture_starters:
                        for pid in team_fixture_starters[old_key]:
                            player_appearances[team_id][pid] -= 1
                            if player_appearances[team_id][pid] <= 0:
                                del player_appearances[team_id][pid]
                        del team_fixture_starters[old_key]

        print(f"Created {len(features_list)} squad quality features")
        return pd.DataFrame(features_list)

    def _squad_features(
        self,
        team_id: int,
        current_xi: set,
        player_ema: Dict[int, Dict[int, float]],
        player_appearances: Dict[int, Dict[int, int]],
    ) -> Dict[str, float]:
        """Calculate squad quality features from historical data only."""
        team_emas = player_ema.get(team_id, {})
        team_apps = player_appearances.get(team_id, {})

        if not team_emas or not current_xi:
            return {'xi_avg_rating': 6.5, 'missing_rating': 0.0}

        # XI average rating (only players with known ratings)
        xi_ratings = [team_emas[pid] for pid in current_xi if pid in team_emas]
        xi_avg = sum(xi_ratings) / len(xi_ratings) if xi_ratings else 6.5

        # Expected starters: top 11 by appearances
        sorted_players = sorted(team_apps.items(), key=lambda x: x[1], reverse=True)
        expected_xi = set(pid for pid, _ in sorted_players[:11])

        # Missing expected starters
        missing = expected_xi - current_xi
        missing_rating = sum(team_emas.get(pid, 0) for pid in missing)

        return {'xi_avg_rating': xi_avg, 'missing_rating': missing_rating}


