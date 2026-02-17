"""
Niche Stat Derived Feature Engineering

Cross-stat ratio features and rolling volatility for niche betting markets.
Adds two critical signal dimensions missing from level-based EMA features:

1. **Consistency signal (volatility)**: Rolling std captures distribution width.
   Two teams averaging 12 fouls each ≈ expected total 24, but std=1 vs std=5
   changes OVER/UNDER probability significantly.

2. **Quality/efficiency signal (ratios)**: Fouls-to-cards ratio captures referee
   strictness; shots-to-corners ratio captures attack style; shot conversion
   captures finishing efficiency.

Data: Loads match_stats.parquet (same source as FoulsFeatureEngineer).
"""
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_collection.match_stats_utils import normalize_match_stats_columns
from src.features.engineers.base import BaseFeatureEngineer
from src.leagues import EUROPEAN_LEAGUES

logger = logging.getLogger(__name__)


class NicheStatDerivedFeatureEngineer(BaseFeatureEngineer):
    """
    Generates cross-stat ratio and volatility features for niche markets.

    Ratio features capture quality/efficiency signals (e.g., fouls per card,
    shots per corner). Volatility features capture consistency signals via
    rolling standard deviation.

    All features use shift(1) to prevent data leakage.
    """

    # Stat pairs for ratio features: (numerator, denominator, output_name)
    RATIO_DEFS = [
        ('home_fouls', 'home_cards', 'fouls_per_card'),
        ('away_fouls', 'away_cards', 'fouls_per_card'),
        ('home_cards', 'home_fouls', 'cards_per_foul'),
        ('away_cards', 'away_fouls', 'cards_per_foul'),
        ('home_goals', 'home_shots', 'shot_conversion'),
        ('away_goals', 'away_shots', 'shot_conversion'),
        ('home_shots', 'home_corners', 'shots_per_corner'),
        ('away_shots', 'away_corners', 'shots_per_corner'),
        ('home_corners', 'home_shots', 'corners_per_shot'),
        ('away_corners', 'away_shots', 'corners_per_shot'),
    ]

    # Stats for volatility features
    VOLATILITY_STATS = ['fouls', 'shots', 'corners', 'cards']

    def __init__(
        self,
        volatility_window: int = 10,
        ratio_ema_span: int = 10,
        min_matches: int = 3,
    ):
        self.volatility_window = volatility_window
        self.ratio_ema_span = ratio_ema_span
        self.min_matches = min_matches
        self.data_dir = Path("data/01-raw")

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create ratio and volatility features from match stats."""
        matches = data.get('matches')
        if matches is None or matches.empty:
            return pd.DataFrame()

        match_stats = self._load_match_stats()
        if match_stats.empty:
            return pd.DataFrame()

        # Merge goals from matches for shot conversion ratio
        match_stats = self._merge_goals(match_stats, matches)

        # Derive cards from yellow + red if available
        match_stats = self._derive_cards(match_stats)

        featured = self._build_features(match_stats)

        # Return only new feature columns + fixture_id
        feature_cols = [
            c for c in featured.columns
            if c not in match_stats.columns or c == 'fixture_id'
        ]
        if 'fixture_id' not in feature_cols:
            feature_cols = ['fixture_id'] + feature_cols

        return featured[feature_cols]

    def _load_match_stats(self) -> pd.DataFrame:
        """Load match stats from all leagues."""
        all_stats = []
        for league in EUROPEAN_LEAGUES:
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
                        df = normalize_match_stats_columns(df)
                        df['league'] = league
                        all_stats.append(df)
                    except Exception as e:
                        logger.debug(f"Could not load {stats_path}: {e}")

        return pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

    def _merge_goals(self, stats: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
        """Merge goal columns from matches if not in stats."""
        if 'home_goals' in stats.columns and 'away_goals' in stats.columns:
            return stats

        if 'fixture_id' not in matches.columns:
            return stats

        goal_cols = []
        if 'home_goals' in matches.columns:
            goal_cols.append('home_goals')
        if 'away_goals' in matches.columns:
            goal_cols.append('away_goals')

        if not goal_cols:
            return stats

        goals_df = matches[['fixture_id'] + goal_cols].drop_duplicates('fixture_id')
        stats = stats.merge(goals_df, on='fixture_id', how='left')
        return stats

    def _derive_cards(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive home_cards/away_cards from yellow + red if not present."""
        if 'home_cards' not in df.columns:
            if 'home_yellow_cards' in df.columns:
                red = df.get('home_red_cards', 0)
                df['home_cards'] = df['home_yellow_cards'].fillna(0) + pd.Series(red).fillna(0)
            else:
                return df

        if 'away_cards' not in df.columns:
            if 'away_yellow_cards' in df.columns:
                red = df.get('away_red_cards', 0)
                df['away_cards'] = df['away_yellow_cards'].fillna(0) + pd.Series(red).fillna(0)
            else:
                return df

        return df

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all ratio and volatility features."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)

        # Ensure totals exist
        for stat in self.VOLATILITY_STATS:
            home_col = f'home_{stat}'
            away_col = f'away_{stat}'
            total_col = f'total_{stat}'
            if home_col in df.columns and away_col in df.columns and total_col not in df.columns:
                df[total_col] = df[home_col] + df[away_col]

        # --- Ratio features ---
        self._build_ratio_features(df)

        # --- Cross-stat expected ratios ---
        self._build_cross_stat_ratios(df)

        # --- Volatility features ---
        self._build_volatility_features(df)

        return df

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Safe division replacing 0 denominator with NaN, clipping result."""
        return (numerator / denominator.replace(0, np.nan)).clip(0, 20)

    def _build_ratio_features(self, df: pd.DataFrame) -> None:
        """Build per-team EMA ratio features with shift(1) leakage protection."""
        for num_col, den_col, ratio_name in self.RATIO_DEFS:
            if num_col not in df.columns or den_col not in df.columns:
                continue

            # Determine side (home/away) from column name prefix
            side = 'home' if num_col.startswith('home_') else 'away'
            team_col = f'{side}_team'
            if team_col not in df.columns:
                continue

            # Per-match ratio
            match_ratio = self._safe_divide(df[num_col], df[den_col])

            # EMA of ratio with shift(1) for leakage safety
            feat_name = f'{side}_{ratio_name}_ema'
            temp_col = f'_tmp_{side}_{ratio_name}'
            df[temp_col] = match_ratio
            df[feat_name] = df.groupby(team_col)[temp_col].transform(
                lambda x: x.shift(1).ewm(span=self.ratio_ema_span, min_periods=self.min_matches).mean()
            )
            df.drop(columns=[temp_col], inplace=True)

        # Opponent fouls drawn ratio (opponent_fouls / team_fouls)
        if 'home_fouls' in df.columns and 'away_fouls' in df.columns:
            if 'home_team' in df.columns:
                temp = self._safe_divide(df['away_fouls'], df['home_fouls'])
                df['_tmp_home_drawn'] = temp
                df['home_fouls_drawn_ratio_ema'] = df.groupby('home_team')['_tmp_home_drawn'].transform(
                    lambda x: x.shift(1).ewm(span=self.ratio_ema_span, min_periods=self.min_matches).mean()
                )
                df.drop(columns=['_tmp_home_drawn'], inplace=True)

            if 'away_team' in df.columns:
                temp = self._safe_divide(df['home_fouls'], df['away_fouls'])
                df['_tmp_away_drawn'] = temp
                df['away_fouls_drawn_ratio_ema'] = df.groupby('away_team')['_tmp_away_drawn'].transform(
                    lambda x: x.shift(1).ewm(span=self.ratio_ema_span, min_periods=self.min_matches).mean()
                )
                df.drop(columns=['_tmp_away_drawn'], inplace=True)

        # Diff features for key ratios
        ratio_pairs = [
            ('fouls_per_card', 'fouls_per_card_diff'),
            ('shot_conversion', 'shot_conversion_diff'),
            ('shots_per_corner', 'shots_per_corner_diff'),
            ('cards_per_foul', 'cards_per_foul_diff'),
        ]
        for ratio_name, diff_name in ratio_pairs:
            home_col = f'home_{ratio_name}_ema'
            away_col = f'away_{ratio_name}_ema'
            if home_col in df.columns and away_col in df.columns:
                df[diff_name] = df[home_col] - df[away_col]

        if 'home_fouls_drawn_ratio_ema' in df.columns and 'away_fouls_drawn_ratio_ema' in df.columns:
            df['fouls_drawn_ratio_diff'] = df['home_fouls_drawn_ratio_ema'] - df['away_fouls_drawn_ratio_ema']

    def _build_cross_stat_ratios(self, df: pd.DataFrame) -> None:
        """Build cross-stat expected ratios from existing niche market features or raw totals."""
        # These use the total columns directly with shift(1) EMA
        cross_pairs = [
            ('total_fouls', 'total_shots', 'expected_fouls_shots_ratio'),
            ('total_cards', 'total_fouls', 'expected_cards_fouls_ratio'),
            ('total_corners', 'total_shots', 'expected_corners_shots_ratio'),
        ]
        for num_col, den_col, feat_name in cross_pairs:
            if num_col not in df.columns or den_col not in df.columns:
                continue
            match_ratio = self._safe_divide(df[num_col], df[den_col])
            # Global rolling EMA (not per-team, since these are match totals)
            df[feat_name] = match_ratio.shift(1).ewm(
                span=self.ratio_ema_span, min_periods=self.min_matches
            ).mean()

        # Corner dominance: expected_home / (expected_home + expected_away)
        if 'home_corners' in df.columns and 'away_corners' in df.columns:
            total = df['home_corners'] + df['away_corners']
            dominance = self._safe_divide(df['home_corners'], total)
            df['_tmp_corner_dom'] = dominance
            if 'home_team' in df.columns:
                df['corner_dominance'] = df.groupby('home_team')['_tmp_corner_dom'].transform(
                    lambda x: x.shift(1).ewm(span=self.ratio_ema_span, min_periods=self.min_matches).mean()
                )
            else:
                df['corner_dominance'] = dominance.shift(1).ewm(
                    span=self.ratio_ema_span, min_periods=self.min_matches
                ).mean()
            df.drop(columns=['_tmp_corner_dom'], inplace=True)

    def _build_volatility_features(self, df: pd.DataFrame) -> None:
        """Build rolling std volatility features with shift(1) leakage protection."""
        window = self.volatility_window

        for stat in self.VOLATILITY_STATS:
            home_col = f'home_{stat}'
            away_col = f'away_{stat}'
            total_col = f'total_{stat}'

            # Per-team volatility (home side)
            if home_col in df.columns and 'home_team' in df.columns:
                df[f'home_{stat}_volatility'] = df.groupby('home_team')[home_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=self.min_matches).std()
                )

            # Per-team volatility (away side)
            if away_col in df.columns and 'away_team' in df.columns:
                df[f'away_{stat}_volatility'] = df.groupby('away_team')[away_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=self.min_matches).std()
                )

            # Volatility diff
            home_vol = f'home_{stat}_volatility'
            away_vol = f'away_{stat}_volatility'
            if home_vol in df.columns and away_vol in df.columns:
                df[f'{stat}_volatility_diff'] = df[home_vol] - df[away_vol]

            # Match total volatility (global rolling, not per-team)
            if total_col in df.columns:
                df[f'total_{stat}_volatility'] = df[total_col].shift(1).rolling(
                    window=window, min_periods=self.min_matches
                ).std()
