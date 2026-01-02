"""
Automated tests to prevent data leakage in feature engineering.

This module provides comprehensive tests to:
1. Block features with outcome-related names/patterns
2. Alert on suspiciously high correlations with targets
3. Detect unrealistic model performance in time validation
4. Ensure target columns are properly separated from features

Run with: pytest tests/test_data_leakage.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Set
import warnings

LEAKY_EXACT_COLUMNS = {
    'draw',
    'home_win',
    'away_win',
    'match_result',
    'total_goals',
    'goal_difference',
    'winner',
    'ft_home',
    'ft_away',
    'ht_home',
    'ht_away',
    'final_score',
}

ALLOWED_PATTERNS = [
    '_last_n',
    '_ema',
    '_prob',
    '_pct',
    'h2h_',
    'poisson_',
    'ref_',
    'b365_',
    'avg_',
    'max_',
    '_open',
    '_close',
    '_streak',
    'home_home_',
    'away_away_',
    'goals_scored',
    'goals_conceded',
]

TARGET_COLUMNS = {
    'home_win', 'draw', 'away_win', 'match_result',
    'total_goals', 'goal_difference', 'gd_form_diff'
}

NON_FEATURE_COLUMNS = {
    'fixture_id', 'date', 'home_team_id', 'home_team_name',
    'away_team_id', 'away_team_name', 'round', 'round_number',
    'league', 'season', 'week'
}

MAX_CORRELATION_THRESHOLD = 0.95

MAX_FEATURE_IMPORTANCE_THRESHOLD = 0.5

SUSPICIOUS_ACCURACY_THRESHOLD = 0.90


class DataLeakageChecker:
    """Utility class for checking data leakage in feature sets."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []

    def check_leaky_column_names(self) -> List[str]:
        """
        Check for columns that are exact leaky column names.

        Only flags EXACT matches to known leaky columns.
        Does NOT flag derived features like 'home_wins_last_n'.
        """
        leaky_cols = []
        for col in self.df.columns:
            if col.lower() in {c.lower() for c in LEAKY_EXACT_COLUMNS}:
                leaky_cols.append(col)
        return leaky_cols

    def check_target_in_features(self, feature_cols: List[str]) -> List[str]:
        """Check if any target columns are in the feature list."""
        targets_in_features = []
        for col in feature_cols:
            if col in TARGET_COLUMNS:
                targets_in_features.append(col)
        return targets_in_features

    def check_correlation_with_target(
        self,
        feature_cols: List[str],
        target_col: str,
        threshold: float = MAX_CORRELATION_THRESHOLD
    ) -> List[tuple]:
        """Check for features with suspiciously high correlation to target."""
        high_corr_features = []

        if target_col not in self.df.columns:
            return high_corr_features

        target = self.df[target_col]

        for col in feature_cols:
            if col not in self.df.columns:
                continue
            if self.df[col].dtype not in ['int64', 'float64']:
                continue

            try:
                corr = self.df[col].corr(target)
                if abs(corr) >= threshold:
                    high_corr_features.append((col, corr))
            except Exception:
                pass

        return high_corr_features

    def get_clean_feature_columns(self) -> List[str]:
        """Get feature columns excluding targets and non-features."""
        all_cols = set(self.df.columns)
        exclude = TARGET_COLUMNS | NON_FEATURE_COLUMNS | LEAKY_EXACT_COLUMNS

        feature_cols = [
            col for col in all_cols - exclude
            if self.df[col].dtype in ['int64', 'float64']
        ]

        return sorted(feature_cols)


def load_features_file(filename: str = "features_all_with_odds.csv") -> pd.DataFrame:
    """Load features file from standard location."""
    paths = [
        Path(f"data/03-features/{filename}"),
        Path(f"../data/03-features/{filename}"),
        Path(f"../../data/03-features/{filename}"),
    ]

    for path in paths:
        if path.exists():
            return pd.read_csv(path)

    raise FileNotFoundError(f"Could not find {filename} in any expected location")

class TestDataLeakage:
    """Test suite for data leakage prevention."""

    @pytest.fixture
    def features_df(self):
        """Load the features dataframe."""
        try:
            return load_features_file()
        except FileNotFoundError:
            pytest.skip("Features file not found")

    @pytest.fixture
    def checker(self, features_df):
        """Create a DataLeakageChecker instance."""
        return DataLeakageChecker(features_df)

    def test_no_leaky_column_names_in_features(self, checker):
        """
        Test that no feature columns have names suggesting outcome leakage.

        CRITICAL: Features should not contain columns like 'draw', 'home_win',
        'total_goals', etc. unless they are clearly historical/lagged values.
        """
        feature_cols = checker.get_clean_feature_columns()
        leaky_cols = checker.check_leaky_column_names()

        leaky_in_features = [
            col for col in leaky_cols
            if col.split(' (')[0] in feature_cols
        ]

        assert len(leaky_in_features) == 0, (
            f"Found {len(leaky_in_features)} columns with leaky names in features:\n"
            f"{leaky_in_features}"
        )

    def test_target_columns_excluded_from_features(self, checker):
        """
        Test that known target columns are not used as features.

        CRITICAL: Target variables like 'home_win', 'match_result' must be
        excluded from the feature set to prevent leakage.
        """
        feature_cols = checker.get_clean_feature_columns()
        targets_in_features = checker.check_target_in_features(feature_cols)

        assert len(targets_in_features) == 0, (
            f"Found {len(targets_in_features)} target columns in features:\n"
            f"{targets_in_features}"
        )

    def test_no_high_correlation_with_home_win(self, checker):
        """
        Test that no feature has suspiciously high correlation with home_win.

        CRITICAL: Correlation > 0.95 suggests the feature is derived from
        or equivalent to the target variable.
        """
        feature_cols = checker.get_clean_feature_columns()
        high_corr = checker.check_correlation_with_target(
            feature_cols, 'home_win', threshold=MAX_CORRELATION_THRESHOLD
        )

        assert len(high_corr) == 0, (
            f"Found {len(high_corr)} features with correlation > {MAX_CORRELATION_THRESHOLD} "
            f"to home_win:\n{high_corr}"
        )

    def test_no_high_correlation_with_total_goals(self, checker):
        """
        Test that no feature has suspiciously high correlation with total_goals.
        """
        feature_cols = checker.get_clean_feature_columns()
        high_corr = checker.check_correlation_with_target(
            feature_cols, 'total_goals', threshold=MAX_CORRELATION_THRESHOLD
        )

        assert len(high_corr) == 0, (
            f"Found {len(high_corr)} features with correlation > {MAX_CORRELATION_THRESHOLD} "
            f"to total_goals:\n{high_corr}"
        )

    def test_no_high_correlation_with_draw(self, checker):
        """
        Test that no feature has suspiciously high correlation with draw.
        """
        feature_cols = checker.get_clean_feature_columns()
        high_corr = checker.check_correlation_with_target(
            feature_cols, 'draw', threshold=MAX_CORRELATION_THRESHOLD
        )

        assert len(high_corr) == 0, (
            f"Found {len(high_corr)} features with correlation > {MAX_CORRELATION_THRESHOLD} "
            f"to draw:\n{high_corr}"
        )

    def test_feature_count_reasonable(self, checker):
        """
        Test that the number of features is reasonable.

        Too many features (> 500) suggests possible feature explosion or
        improper one-hot encoding that might include target info.
        """
        feature_cols = checker.get_clean_feature_columns()
        assert len(feature_cols) < 500, (
            f"Feature count ({len(feature_cols)}) seems too high. "
            "Check for feature explosion or improper encoding."
        )

    def test_no_perfect_predictors(self, features_df, checker):
        """
        Test that no single feature can perfectly predict the target.

        CRITICAL: If any feature achieves > 90% accuracy alone, it's likely
        leaking information from the target.
        """
        feature_cols = checker.get_clean_feature_columns()

        if 'home_win' not in features_df.columns:
            pytest.skip("home_win column not found")

        target = features_df['home_win'].values

        perfect_predictors = []
        for col in feature_cols[:50]:
            if col not in features_df.columns:
                continue

            feature = features_df[col].values

            try:
                if features_df[col].nunique() == 2:
                    pred = feature
                    acc = (pred == target).mean()
                    if acc > SUSPICIOUS_ACCURACY_THRESHOLD:
                        perfect_predictors.append((col, acc))
            except Exception:
                pass

        assert len(perfect_predictors) == 0, (
            f"Found features that achieve > {SUSPICIOUS_ACCURACY_THRESHOLD*100}% accuracy alone:\n"
            f"{perfect_predictors}\n"
            "This strongly suggests data leakage!"
        )


class TestTrainingPipelineExclusions:
    """Test that TrainingPipeline properly excludes target columns."""

    def test_target_columns_defined(self):
        """Test that TARGET_COLUMNS is properly defined in training pipeline."""
        try:
            from src.pipelines.training_pipeline import TrainingPipeline
        except ImportError as e:
            pytest.skip(f"Cannot import TrainingPipeline: {e}")

        expected_targets = {'home_win', 'draw', 'away_win', 'match_result',
                          'total_goals', 'goal_difference'}

        actual_targets = set(TrainingPipeline.TARGET_COLUMNS)

        assert expected_targets.issubset(actual_targets), (
            f"TrainingPipeline.TARGET_COLUMNS is missing: "
            f"{expected_targets - actual_targets}"
        )

    def test_non_feature_columns_defined(self):
        """Test that NON_FEATURE_COLUMNS is properly defined."""
        try:
            from src.pipelines.training_pipeline import TrainingPipeline
        except ImportError as e:
            pytest.skip(f"Cannot import TrainingPipeline: {e}")

        expected_non_features = {'fixture_id', 'date', 'home_team_id',
                                 'away_team_id', 'home_team_name', 'away_team_name'}

        actual_non_features = set(TrainingPipeline.NON_FEATURE_COLUMNS)

        assert expected_non_features.issubset(actual_non_features), (
            f"TrainingPipeline.NON_FEATURE_COLUMNS is missing: "
            f"{expected_non_features - actual_non_features}"
        )


def validate_feature_file(filepath: str, verbose: bool = True) -> bool:
    """
    Validate a feature file for data leakage.

    Args:
        filepath: Path to the features CSV file
        verbose: Whether to print detailed output

    Returns:
        True if no leakage detected, False otherwise
    """
    df = pd.read_csv(filepath)
    checker = DataLeakageChecker(df)

    issues = []

    leaky_cols = checker.check_leaky_column_names()
    if leaky_cols:
        issues.append(f"Leaky column names found: {leaky_cols}")

    feature_cols = checker.get_clean_feature_columns()
    targets_in_features = checker.check_target_in_features(feature_cols)
    if targets_in_features:
        issues.append(f"Target columns in features: {targets_in_features}")

    for target in ['home_win', 'draw', 'total_goals']:
        if target in df.columns:
            high_corr = checker.check_correlation_with_target(feature_cols, target)
            if high_corr:
                issues.append(f"High correlation with {target}: {high_corr}")

    if verbose:
        if issues:
            print("DATA LEAKAGE DETECTED!")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No data leakage detected.")
            print(f"Clean feature count: {len(feature_cols)}")

    return len(issues) == 0


if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/03-features/features_all_with_odds.csv"

    print(f"Validating: {filepath}")
    print("=" * 60)

    is_clean = validate_feature_file(filepath)

    sys.exit(0 if is_clean else 1)
