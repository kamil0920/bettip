"""
Niche Markets Feature Engineering

Builds predictive features for niche betting markets:
- Fouls: Total fouls over/under
- Cards: Total cards (yellows + reds) over/under
- Shots: Total shots over/under

Data distributions (from 6500+ matches):
- Fouls: Mean 24.5, Std 6.0, Over 24.5: 48.2%
- Shots: Mean 24.8, Std 5.9, Over 24.5: 50.2%
- Cards: Mean 4.5, Std 2.2, Over 4.5: 45.7%
"""
import logging
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer

logger = logging.getLogger(__name__)


class FoulsFeatureEngineer(BaseFeatureEngineer):
    """
    Generates features for predicting match fouls.

    Key insight: Fouls correlate negatively with attacking play
    (more shots/corners = fewer fouls).
    """

    DEFAULTS = {
        'fouls_committed': 12.0,  # Average fouls per team
        'fouls_drawn': 12.0,
    }

    def __init__(
        self,
        window_sizes: List[int] = [5, 10],
        min_matches: int = 3,
        ema_span: int = 10
    ):
        self.window_sizes = window_sizes
        self.min_matches = min_matches
        self.ema_span = ema_span
        self.data_dir = Path("data/01-raw")

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create fouls features from match data."""
        matches = data.get('matches')
        if matches is None or matches.empty:
            return pd.DataFrame()

        match_stats = self._load_match_stats()
        if match_stats.empty:
            return pd.DataFrame()

        featured = self._build_features(match_stats)
        feature_cols = [c for c in featured.columns if c not in match_stats.columns or c == 'fixture_id']
        if 'fixture_id' not in feature_cols:
            feature_cols = ['fixture_id'] + feature_cols

        return featured[feature_cols]

    def _load_match_stats(self) -> pd.DataFrame:
        """Load match stats with fouls data."""
        all_stats = []
        for league in ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']:
            league_dir = self.data_dir / league
            if not league_dir.exists():
                continue
            for season_dir in league_dir.iterdir():
                if not season_dir.is_dir():
                    continue
                stats_path = season_dir / 'match_stats.parquet'
                if stats_path.exists():
                    try:
                        df = pd.read_parquet(stats_path)
                        if 'home_fouls' in df.columns:
                            df['league'] = league
                            all_stats.append(df)
                    except Exception as e:
                        logger.debug(f"Could not load {stats_path}: {e}")

        return pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build fouls prediction features."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)

        if 'total_fouls' not in df.columns:
            df['total_fouls'] = df['home_fouls'] + df['away_fouls']

        # Calculate team rolling fouls (match-level, distinct from player-stats EMA)
        df['home_fouls_match_ema'] = df.groupby('home_team')['home_fouls'].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )
        df['away_fouls_match_ema'] = df.groupby('away_team')['away_fouls'].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )

        # Fouls drawn (opponent's fouls)
        df['home_fouls_drawn_ema'] = df.groupby('home_team')['away_fouls'].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )
        df['away_fouls_drawn_ema'] = df.groupby('away_team')['home_fouls'].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )

        # Expected fouls
        df['expected_home_fouls'] = (
            df['home_fouls_match_ema'].fillna(self.DEFAULTS['fouls_committed']) +
            df['away_fouls_drawn_ema'].fillna(self.DEFAULTS['fouls_drawn'])
        ) / 2

        df['expected_away_fouls'] = (
            df['away_fouls_match_ema'].fillna(self.DEFAULTS['fouls_committed']) +
            df['home_fouls_drawn_ema'].fillna(self.DEFAULTS['fouls_drawn'])
        ) / 2

        df['expected_total_fouls'] = df['expected_home_fouls'] + df['expected_away_fouls']

        # Fouls intensity (fouls per possession proxy)
        df['fouls_diff'] = df['home_fouls_match_ema'] - df['away_fouls_match_ema']

        return df


class CardsFeatureEngineer(BaseFeatureEngineer):
    """
    Generates features for predicting match cards.

    Cards = yellow cards + red cards
    Key insight: Referee assignment is critical for cards prediction.
    """

    DEFAULTS = {
        'cards_received': 2.25,  # Average cards per team
        'cards_caused': 2.25,
    }

    def __init__(
        self,
        window_sizes: List[int] = [5, 10],
        min_matches: int = 3,
        ema_span: int = 10
    ):
        self.window_sizes = window_sizes
        self.min_matches = min_matches
        self.ema_span = ema_span
        self.data_dir = Path("data")

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create cards features from match and events data."""
        matches = data.get('matches')
        events = data.get('events')

        if matches is None or matches.empty:
            return pd.DataFrame()

        # Build cards per match from events
        cards_df = self._build_cards_data(events)
        if cards_df.empty:
            return pd.DataFrame()

        # Merge with matches
        match_cards = matches.merge(cards_df, on='fixture_id', how='left')
        match_cards['total_cards'] = match_cards['total_cards'].fillna(0)
        match_cards['home_cards'] = match_cards['home_cards'].fillna(0)
        match_cards['away_cards'] = match_cards['away_cards'].fillna(0)

        featured = self._build_features(match_cards)
        feature_cols = [c for c in featured.columns if 'card' in c.lower() or c == 'fixture_id']

        return featured[feature_cols]

    def _build_cards_data(self, events: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Build cards per match from events data."""
        if events is None or events.empty:
            # Try loading from files
            events = self._load_events()

        if events.empty:
            return pd.DataFrame()

        cards = events[events['type'] == 'Card'].copy()
        if cards.empty:
            return pd.DataFrame()

        # Count cards per match and team
        cards_per_match = cards.groupby(['fixture_id', 'team_id']).size().reset_index(name='cards')

        # Pivot to get home/away cards
        # For now, just get total cards per match
        total_cards = cards.groupby('fixture_id').size().reset_index(name='total_cards')

        # Try to separate home/away (need team mapping)
        # Simplified: assume first team is home
        home_cards = cards.groupby('fixture_id').apply(
            lambda x: len(x[x['team_id'] == x['team_id'].iloc[0]]) if len(x) > 0 else 0
        ).reset_index(name='home_cards')

        away_cards = cards.groupby('fixture_id').apply(
            lambda x: len(x[x['team_id'] != x['team_id'].iloc[0]]) if len(x) > 0 else 0
        ).reset_index(name='away_cards')

        result = total_cards.merge(home_cards, on='fixture_id', how='left')
        result = result.merge(away_cards, on='fixture_id', how='left')

        return result

    def _load_events(self) -> pd.DataFrame:
        """Load events from preprocessed data."""
        all_events = []
        base = self.data_dir / '02-preprocessed'

        for league in ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']:
            league_dir = base / league
            if not league_dir.exists():
                continue
            for season_dir in league_dir.iterdir():
                if not season_dir.is_dir():
                    continue
                events_path = season_dir / 'events.parquet'
                if events_path.exists():
                    try:
                        df = pd.read_parquet(events_path)
                        all_events.append(df)
                    except Exception as e:
                        logger.debug(f"Could not load {events_path}: {e}")

        return pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build cards prediction features."""
        df = df.copy()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.sort_values('date').reset_index(drop=True)

        # Team rolling cards
        if 'home_team_name' in df.columns:
            team_col = 'home_team_name'
            away_team_col = 'away_team_name'
        elif 'home_team' in df.columns:
            team_col = 'home_team'
            away_team_col = 'away_team'
        else:
            return df

        df['home_cards_ema'] = df.groupby(team_col)['home_cards'].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )
        df['away_cards_ema'] = df.groupby(away_team_col)['away_cards'].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )

        # Expected cards
        df['expected_home_cards'] = df['home_cards_ema'].fillna(self.DEFAULTS['cards_received'])
        df['expected_away_cards'] = df['away_cards_ema'].fillna(self.DEFAULTS['cards_received'])
        df['expected_total_cards'] = df['expected_home_cards'] + df['expected_away_cards']

        # Cards differential
        df['cards_diff'] = df['home_cards_ema'] - df['away_cards_ema']

        return df


class ShotsFeatureEngineer(BaseFeatureEngineer):
    """
    Generates features for predicting total match shots.

    Key insight: Corners correlate 0.271 with shots (useful predictor).
    """

    DEFAULTS = {
        'shots': 12.5,  # Average shots per team
        'shots_on_target': 4.5,
    }

    def __init__(
        self,
        window_sizes: List[int] = [5, 10],
        min_matches: int = 3,
        ema_span: int = 10
    ):
        self.window_sizes = window_sizes
        self.min_matches = min_matches
        self.ema_span = ema_span
        self.data_dir = Path("data/01-raw")

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create shots features from match data."""
        matches = data.get('matches')
        if matches is None or matches.empty:
            return pd.DataFrame()

        match_stats = self._load_match_stats()
        if match_stats.empty:
            return pd.DataFrame()

        featured = self._build_features(match_stats)
        feature_cols = [c for c in featured.columns if 'shot' in c.lower() or c == 'fixture_id']

        return featured[feature_cols]

    def _load_match_stats(self) -> pd.DataFrame:
        """Load match stats with shots data."""
        all_stats = []
        for league in ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']:
            league_dir = self.data_dir / league
            if not league_dir.exists():
                continue
            for season_dir in league_dir.iterdir():
                if not season_dir.is_dir():
                    continue
                stats_path = season_dir / 'match_stats.parquet'
                if stats_path.exists():
                    try:
                        df = pd.read_parquet(stats_path)
                        if 'home_shots' in df.columns:
                            df['league'] = league
                            all_stats.append(df)
                    except Exception as e:
                        logger.debug(f"Could not load {stats_path}: {e}")

        return pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build shots prediction features."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)

        if 'total_shots' not in df.columns:
            df['total_shots'] = df['home_shots'] + df['away_shots']

        # Team rolling shots (match-level, distinct from player-stats EMA)
        df['home_shots_match_ema'] = df.groupby('home_team')['home_shots'].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )
        df['away_shots_match_ema'] = df.groupby('away_team')['away_shots'].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )

        # Shots conceded
        df['home_shots_conceded_ema'] = df.groupby('home_team')['away_shots'].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )
        df['away_shots_conceded_ema'] = df.groupby('away_team')['home_shots'].transform(
            lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
        )

        # Shots on target
        if 'home_shots_on_target' in df.columns:
            df['home_shots_on_target_ema'] = df.groupby('home_team')['home_shots_on_target'].transform(
                lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
            )
            df['away_shots_on_target_ema'] = df.groupby('away_team')['away_shots_on_target'].transform(
                lambda x: x.shift(1).ewm(span=self.ema_span, min_periods=self.min_matches).mean()
            )

        # Expected shots
        df['expected_home_shots'] = (
            df['home_shots_match_ema'].fillna(self.DEFAULTS['shots']) +
            df['away_shots_conceded_ema'].fillna(self.DEFAULTS['shots'])
        ) / 2

        df['expected_away_shots'] = (
            df['away_shots_match_ema'].fillna(self.DEFAULTS['shots']) +
            df['home_shots_conceded_ema'].fillna(self.DEFAULTS['shots'])
        ) / 2

        df['expected_total_shots'] = df['expected_home_shots'] + df['expected_away_shots']

        # Shot quality ratio
        if 'home_shots_on_target_ema' in df.columns:
            df['home_shot_accuracy'] = df['home_shots_on_target_ema'] / df['home_shots_match_ema'].replace(0, 1)
            df['away_shot_accuracy'] = df['away_shots_on_target_ema'] / df['away_shots_match_ema'].replace(0, 1)

        # Shots differential
        df['shots_attack_diff'] = df['home_shots_match_ema'] - df['away_shots_match_ema']

        return df


def validate_niche_features(market: str = 'fouls') -> Dict:
    """
    Validate predictive power of niche market features.

    Args:
        market: 'fouls', 'cards', or 'shots'

    Returns:
        Dict with validation metrics
    """
    from pathlib import Path
    import numpy as np

    # Load data
    all_stats = []
    base = Path('data/01-raw')

    for league in ['premier_league', 'la_liga', 'serie_a']:
        for season_dir in (base / league).iterdir():
            if season_dir.is_dir():
                stats_path = season_dir / 'match_stats.parquet'
                if stats_path.exists():
                    df = pd.read_parquet(stats_path)
                    df['league'] = league
                    all_stats.append(df)

    stats = pd.concat(all_stats, ignore_index=True)

    if market == 'fouls':
        engineer = FoulsFeatureEngineer()
        target_col = 'total_fouls'
        expected_col = 'expected_total_fouls'
        lines = [22.5, 24.5, 26.5]
    elif market == 'shots':
        engineer = ShotsFeatureEngineer()
        target_col = 'total_shots'
        expected_col = 'expected_total_shots'
        lines = [22.5, 24.5, 26.5]
    else:
        raise ValueError(f"Unknown market: {market}")

    # Build features
    featured = engineer._build_features(stats)

    # Walk-forward validation
    featured = featured.sort_values('date').reset_index(drop=True)
    train_size = int(len(featured) * 0.7)
    test = featured.iloc[train_size:].copy()
    test = test.dropna(subset=[expected_col, target_col])

    # Calculate correlation
    corr = test[expected_col].corr(test[target_col])

    # Calculate accuracy for each line
    results = {
        'market': market,
        'test_size': len(test),
        'correlation': corr,
        'lines': {}
    }

    for line in lines:
        test[f'pred_over_{line}'] = test[expected_col] > line
        test[f'actual_over_{line}'] = test[target_col] > line

        base_rate = test[f'actual_over_{line}'].mean()

        # High confidence predictions
        high_conf_over = test[test[expected_col] > line + 2]
        high_conf_under = test[test[expected_col] < line - 2]

        over_acc = high_conf_over[f'actual_over_{line}'].mean() if len(high_conf_over) > 0 else None
        under_acc = (~high_conf_under[f'actual_over_{line}']).mean() if len(high_conf_under) > 0 else None

        results['lines'][line] = {
            'base_rate_over': base_rate,
            'high_conf_over_count': len(high_conf_over),
            'high_conf_over_accuracy': over_acc,
            'high_conf_under_count': len(high_conf_under),
            'high_conf_under_accuracy': under_acc,
            'over_edge': (over_acc - base_rate) if over_acc else None,
            'under_edge': (under_acc - (1 - base_rate)) if under_acc else None,
        }

    return results
