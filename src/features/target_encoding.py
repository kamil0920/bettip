"""
CV-aware Target Encoding for High-Cardinality Features

Encodes categorical features (team IDs, referee IDs, league) using
target statistics, with proper CV-based fitting to prevent data leakage.

Key protections:
- Time-based CV folds ensure no future information leaks
- Smoothing for rare categories (prevents overfitting to small samples)
- Global mean fallback for unseen categories at inference
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class TimeAwareTargetEncoder:
    """Target encoder that respects time ordering to prevent leakage.

    For each categorical feature, computes smoothed target mean per category
    using only past data (via time-series CV folds).
    """

    def __init__(
        self,
        columns: List[str],
        n_folds: int = 5,
        smoothing: float = 20.0,
        min_samples: int = 5,
    ):
        """
        Args:
            columns: Categorical columns to encode.
            n_folds: Number of time-series CV folds for OOF encoding.
            smoothing: Smoothing factor. Higher = more regularization toward global mean.
                      Effective weight: n_samples / (n_samples + smoothing).
            min_samples: Minimum samples per category; below this, use global mean.
        """
        self.columns = columns
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.encodings_: Dict[str, Dict] = {}  # col -> {category: encoded_value}
        self.global_means_: Dict[str, float] = {}  # col -> global target mean

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> pd.DataFrame:
        """Fit encoder and transform training data using OOF encoding.

        Uses TimeSeriesSplit to compute target encodings only from past data,
        preventing data leakage.

        Args:
            df: Training DataFrame (must be sorted by date).
            target_col: Name of the target column.

        Returns:
            DataFrame with new encoded columns (original columns unchanged).
        """
        df = df.copy()
        y = df[target_col].values
        cv = TimeSeriesSplit(n_splits=self.n_folds)

        for col in self.columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame, skipping")
                continue

            encoded_col = f"{col}_target_enc"
            df[encoded_col] = np.nan
            global_mean = y.mean()
            self.global_means_[col] = float(global_mean)

            # OOF encoding: for each fold, compute encodings from train, apply to val
            for train_idx, val_idx in cv.split(df):
                train_y = y[train_idx]
                train_cats = df[col].iloc[train_idx]

                # Compute smoothed target mean per category
                cat_stats = pd.DataFrame({
                    'category': train_cats,
                    'target': train_y,
                }).groupby('category')['target'].agg(['mean', 'count'])

                # Smoothed encoding: weighted average of category mean and global mean
                cat_stats['encoded'] = (
                    (cat_stats['count'] * cat_stats['mean'] + self.smoothing * global_mean) /
                    (cat_stats['count'] + self.smoothing)
                )

                # Apply minimum samples filter
                cat_stats.loc[cat_stats['count'] < self.min_samples, 'encoded'] = global_mean

                encoding_map = cat_stats['encoded'].to_dict()

                # Apply to validation fold
                df.iloc[val_idx, df.columns.get_loc(encoded_col)] = (
                    df[col].iloc[val_idx].map(encoding_map).fillna(global_mean).values
                )

            # Fill any remaining NaNs (first fold has no OOF predictions)
            df[encoded_col] = df[encoded_col].fillna(global_mean)

            # Store final encodings (computed on full training data) for inference
            full_stats = pd.DataFrame({
                'category': df[col],
                'target': y,
            }).groupby('category')['target'].agg(['mean', 'count'])

            full_stats['encoded'] = (
                (full_stats['count'] * full_stats['mean'] + self.smoothing * global_mean) /
                (full_stats['count'] + self.smoothing)
            )
            full_stats.loc[full_stats['count'] < self.min_samples, 'encoded'] = global_mean

            self.encodings_[col] = full_stats['encoded'].to_dict()

            logger.info(f"Target encoded {col}: {len(self.encodings_[col])} categories, "
                       f"global_mean={global_mean:.4f}")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encodings.

        Unseen categories get the global mean.

        Args:
            df: DataFrame to transform.

        Returns:
            DataFrame with encoded columns added.
        """
        if not self.encodings_:
            raise RuntimeError("Call fit_transform() first")

        df = df.copy()
        for col in self.columns:
            if col not in df.columns:
                continue

            encoded_col = f"{col}_target_enc"
            encoding_map = self.encodings_.get(col, {})
            global_mean = self.global_means_.get(col, 0.5)

            df[encoded_col] = df[col].map(encoding_map).fillna(global_mean)

        return df

    def get_encoded_column_names(self) -> List[str]:
        """Return names of the encoded columns."""
        return [f"{col}_target_enc" for col in self.columns if col in self.encodings_]
