"""Data cleaning utilities for feature engineering."""
import ast
import numpy as np
import pandas as pd

from src.features.interfaces import IDataCleaner


class BasicDataCleaner(IDataCleaner):
    """Basic data cleaner."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic clean: duplicates, sorting.

        Args:
            df: DataFrame

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        if removed > 0:
            print(f"  Removed {removed} duplicates")

        return df_clean


class MatchDataCleaner(IDataCleaner):
    """Match data cleaner with column mapping for raw API format."""

    COLUMN_MAPPING = {
        'fixture.id': 'fixture_id',
        'fixture.date': 'date',
        'fixture.referee': 'referee',
        'fixture.venue.id': 'venue_id',
        'fixture.venue.name': 'venue_name',
        'fixture.status.short': 'status',
        'league.id': 'league_id',
        'league.round': 'round',
        'league.season': 'season',
        'teams.home.id': 'home_team_id',
        'teams.home.name': 'home_team_name',
        'teams.away.id': 'away_team_id',
        'teams.away.name': 'away_team_name',
        'goals.home': 'ft_home',
        'goals.away': 'ft_away',
        'score.halftime.home': 'ht_home',
        'score.halftime.away': 'ht_away',
    }

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean match data with automatic column mapping.

        Args:
            df: DataFrame (raw API format or already cleaned)

        Returns:
            Cleaned DataFrame with standardized column names
        """
        df_clean = df.copy()

        if 'fixture.id' in df_clean.columns:
            df_clean = self._apply_column_mapping(df_clean)

        if 'home_team_name' in df_clean.columns and 'home_team' not in df_clean.columns:
            df_clean['home_team'] = df_clean['home_team_name']
        if 'away_team_name' in df_clean.columns and 'away_team' not in df_clean.columns:
            df_clean['away_team'] = df_clean['away_team_name']

        score_cols = self._get_score_columns(df_clean)
        if score_cols:
            df_clean = df_clean.dropna(subset=score_cols)

        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'])
            df_clean = df_clean.sort_values('date').reset_index(drop=True)

        print(f"Matches: {len(df_clean)} (with full scores)")
        return df_clean

    def _apply_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column mapping from raw API format to clean format."""
        rename_map = {}
        for raw_col, clean_col in self.COLUMN_MAPPING.items():
            if raw_col in df.columns and clean_col not in df.columns:
                rename_map[raw_col] = clean_col

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _get_score_columns(self, df: pd.DataFrame) -> list:
        """Get available score columns for filtering."""
        if 'ft_home' in df.columns and 'ft_away' in df.columns:
            return ['ft_home', 'ft_away']
        elif 'goals.home' in df.columns and 'goals.away' in df.columns:
            return ['goals.home', 'goals.away']
        return []


class PlayerStatsDataCleaner(IDataCleaner):
    """Player stats data cleaner with column mapping for flat API format."""

    # Mapping from flat API columns to clean column names
    COLUMN_MAPPING = {
        'id': 'player_id',
        'team_name': 'team_id',  # Will use team_name as team_id for now
        'games.minutes': 'minutes',
        'games.rating': 'rating',
        'games.position': 'position',
        'goals.total': 'goals',
        'goals.assists': 'assists',
        'goals.saves': 'saves',
        'shots.total': 'shots_total',
        'shots.on': 'shots_on',
        'passes.total': 'passes_total',
        'passes.key': 'passes_key',
        'passes.accuracy': 'passes_accuracy',
        'tackles.total': 'tackles_total',
        'tackles.blocks': 'tackles_blocks',
        'tackles.interceptions': 'tackles_interceptions',
        'duels.total': 'duels_total',
        'duels.won': 'duels_won',
        'dribbles.attempts': 'dribbles_attempts',
        'dribbles.success': 'dribbles_success',
        'fouls.drawn': 'fouls_drawn',
        'fouls.committed': 'fouls_committed',
        'cards.yellow': 'yellow_cards',
        'cards.red': 'red_cards',
    }

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean players stats data.

        Args:
            df: DataFrame

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        if df_clean.empty:
            print(f"Player stats: 0 records (empty)")
            return df_clean

        if 'games.minutes' in df_clean.columns:
            df_clean = self._apply_column_mapping(df_clean)

        # Convert columns that may be strings to numeric
        numeric_cols_to_convert = [
            'rating', 'assists', 'passes_accuracy', 'shots_total', 'shots_on',
            'passes_total', 'passes_key', 'tackles_total', 'fouls_drawn',
            'fouls_committed', 'dribbles_attempts', 'dribbles_success',
            'duels_total', 'duels_won', 'yellow_cards', 'red_cards',
        ]
        for col in numeric_cols_to_convert:
            if col in df_clean.columns and df_clean[col].dtype == 'object':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(0)

        minutes_col = None
        for col_name in ['minutes', 'games.minutes']:
            if col_name in df_clean.columns:
                minutes_col = col_name
                break

        if minutes_col:
            df_clean = df_clean[df_clean[minutes_col] > 0]
            print(f"Player stats: {len(df_clean)} records (filtered by {minutes_col})")
        else:
            # No minutes column - data might be in nested format, skip filtering
            print(f"Player stats: {len(df_clean)} records (no minutes column found)")

        return df_clean

    def _apply_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column mapping from flat API format to clean format."""
        rename_map = {}
        for raw_col, clean_col in self.COLUMN_MAPPING.items():
            if raw_col in df.columns and clean_col not in df.columns:
                rename_map[raw_col] = clean_col

        if rename_map:
            df = df.rename(columns=rename_map)

        return df


class LineupsDataCleaner(IDataCleaner):
    """Lineups data cleaner that extracts formation and starting info."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean lineups data.

        - Extracts formation from nested lineups column if present
        - Adds starting column based on type field
        - Renames columns for consistency

        Args:
            df: DataFrame

        Returns:
            Cleaned DataFrame with formation and starting columns
        """
        df_clean = df.copy()

        if df_clean.empty:
            print("Lineups: 0 records (empty)")
            return df_clean

        # Extract formation and coach from nested lineups column if present
        if 'lineups' in df_clean.columns and df_clean['lineups'].notna().any():
            df_clean = self._extract_formation_and_coach(df_clean)

        # Add starting column based on type field
        if 'type' in df_clean.columns and 'starting' not in df_clean.columns:
            df_clean['starting'] = df_clean['type'].str.lower().str.contains('start', na=False)

        # Rename id to player_id if needed
        if 'id' in df_clean.columns and 'player_id' not in df_clean.columns:
            df_clean = df_clean.rename(columns={'id': 'player_id'})

        print(f"Lineups: {len(df_clean)} records")
        if 'formation' in df_clean.columns:
            formations_count = df_clean['formation'].notna().sum()
            print(f"  Formations available: {formations_count}")

        return df_clean

    def _extract_formation_and_coach(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract formation and coach data from nested lineups column."""
        formations = {}
        coaches = {}  # {(fixture_id, team_name): {'coach_name': str, 'coach_id': int}}

        for idx, row in df.iterrows():
            lineups_val = row.get('lineups')
            if pd.isna(lineups_val) or lineups_val == 'nan':
                continue

            try:
                if isinstance(lineups_val, str):
                    lineups_val = ast.literal_eval(lineups_val)

                # Get fixture_id from the row or from fixture_info
                fixture_id = row.get('fixture_id')
                if pd.isna(fixture_id) and 'fixture_info' in row.index:
                    fixture_info = row.get('fixture_info')
                    if isinstance(fixture_info, str) and fixture_info != 'nan':
                        try:
                            fi_parsed = ast.literal_eval(fixture_info)
                            if isinstance(fi_parsed, dict):
                                fixture_id = fi_parsed.get('id')
                        except (ValueError, SyntaxError):
                            pass

                if isinstance(lineups_val, list):
                    for team_data in lineups_val:
                        if isinstance(team_data, dict):
                            team_info = team_data.get('team', {})
                            team_name = team_info.get('name') if isinstance(team_info, dict) else None
                            formation = team_data.get('formation')

                            if team_name and fixture_id:
                                key = (fixture_id, team_name)
                                if formation:
                                    formations[key] = formation

                                coach_data = team_data.get('coach', {})
                                if isinstance(coach_data, dict) and coach_data.get('name'):
                                    coaches[key] = {
                                        'coach_name': coach_data.get('name'),
                                        'coach_id': coach_data.get('id'),
                                    }
            except (ValueError, SyntaxError):
                continue

        # Apply formations to matching rows
        if formations:
            def get_formation(row):
                fixture_id = row.get('fixture_id')
                team_name = row.get('team_name')
                if pd.isna(fixture_id) or team_name == 'nan' or pd.isna(team_name):
                    return None
                key = (fixture_id, team_name)
                return formations.get(key)

            df['formation'] = df.apply(get_formation, axis=1)

        # Apply coach data to matching rows
        if coaches:
            def get_coach_field(field):
                def getter(row):
                    fixture_id = row.get('fixture_id')
                    team_name = row.get('team_name')
                    if pd.isna(fixture_id) or team_name == 'nan' or pd.isna(team_name):
                        return None
                    key = (fixture_id, team_name)
                    coach_info = coaches.get(key)
                    return coach_info[field] if coach_info else None
                return getter

            df['coach_name'] = df.apply(get_coach_field('coach_name'), axis=1)
            df['coach_id'] = df.apply(get_coach_field('coach_id'), axis=1)

        return df
