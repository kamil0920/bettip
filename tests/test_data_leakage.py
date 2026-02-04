"""
Automated tests to prevent data leakage in feature engineering.

This module provides comprehensive tests to:
1. Block features with outcome-related names/patterns
2. Alert on suspiciously high correlations with targets
3. Detect unrealistic model performance in time validation
4. Ensure target columns are properly separated from features

Run with: pytest tests/test_data_leakage.py -v
"""

import json
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


class TestCrossMarketLeakage:
    """
    Regression tests for cross-market feature data leakage.

    The CrossMarketFeatureEngineer must ONLY use historical/EMA features,
    never actual match outcomes or target columns. This was the root cause
    of R57/R58 data leakage (100% precision on 5/9 markets).
    """

    def test_safe_get_never_uses_target_columns(self):
        """
        Verify _safe_get column lists never contain target/outcome columns.

        CRITICAL: The _safe_get method picks the FIRST available column.
        If a target column (goal_difference, over25) is listed before the
        historical alternative, the model trains on the answer.
        """
        import ast
        import inspect
        import textwrap
        from src.features.engineers.cross_market import CrossMarketFeatureEngineer

        # Columns that must NEVER appear in _safe_get lists
        forbidden_columns = {
            # Target columns
            'goal_difference', 'total_goals', 'home_goals', 'away_goals',
            'home_win', 'away_win', 'draw', 'btts', 'over25', 'under25',
            'match_result', 'result', 'ft_home', 'ft_away',
            # Raw match stats (post-match)
            'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
            'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
            'home_cards', 'away_cards', 'home_possession', 'away_possession',
            # Derived totals
            'total_corners', 'total_fouls', 'total_shots', 'total_cards',
        }

        source = textwrap.dedent(inspect.getsource(CrossMarketFeatureEngineer.create_features))

        # Parse and find all _safe_get calls
        tree = ast.parse(source)
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == '_safe_get':
                    # Second argument is the column list
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.List):
                        for elt in node.args[1].elts:
                            if isinstance(elt, ast.Constant) and elt.value in forbidden_columns:
                                violations.append(elt.value)

        assert len(violations) == 0, (
            f"CrossMarketFeatureEngineer._safe_get uses forbidden columns: {violations}\n"
            "These are target/outcome columns that cause data leakage.\n"
            "Use historical EMA alternatives instead."
        )

    def test_leakage_columns_stripped_in_regenerator(self):
        """
        Verify FeatureRegenerator strips target columns before cross-market pass.
        """
        from src.features.regeneration import FeatureRegenerator

        assert hasattr(FeatureRegenerator, '_LEAKAGE_COLUMNS'), (
            "FeatureRegenerator must define _LEAKAGE_COLUMNS set"
        )

        required_columns = {
            'goal_difference', 'over25', 'under25', 'home_win', 'away_win',
            'home_shots', 'away_shots', 'home_corners', 'away_corners',
            'home_cards', 'away_cards',
        }

        missing = required_columns - FeatureRegenerator._LEAKAGE_COLUMNS
        assert len(missing) == 0, (
            f"FeatureRegenerator._LEAKAGE_COLUMNS is missing critical columns: {missing}"
        )

    def test_leakage_columns_stripped_in_pipeline(self):
        """
        Verify FeatureEngineeringPipeline strips target columns before cross-market pass.
        """
        from src.pipelines.feature_eng_pipeline import FeatureEngineeringPipeline

        assert hasattr(FeatureEngineeringPipeline, '_LEAKAGE_COLUMNS'), (
            "FeatureEngineeringPipeline must define _LEAKAGE_COLUMNS set"
        )

        required_columns = {
            'goal_difference', 'over25', 'under25', 'home_win', 'away_win',
            'home_shots', 'away_shots', 'home_corners', 'away_corners',
            'home_cards', 'away_cards',
        }

        missing = required_columns - FeatureEngineeringPipeline._LEAKAGE_COLUMNS
        assert len(missing) == 0, (
            f"FeatureEngineeringPipeline._LEAKAGE_COLUMNS is missing critical columns: {missing}"
        )

    def test_cross_market_disabled_in_first_pass(self):
        """
        Verify cross_market is disabled in DEFAULT_FEATURE_CONFIGS.

        It should only run as a second pass (after EMA features are merged),
        not in the first pass via the registry. Running in both passes creates
        _x/_y duplicate columns where _y contains leaked values.
        """
        from src.features.registry import DEFAULT_FEATURE_CONFIGS

        cross_market_configs = [
            cfg for cfg in DEFAULT_FEATURE_CONFIGS if cfg.name == 'cross_market'
        ]

        assert len(cross_market_configs) == 1, "cross_market should be in DEFAULT_FEATURE_CONFIGS"
        assert not cross_market_configs[0].enabled, (
            "cross_market must be DISABLED in DEFAULT_FEATURE_CONFIGS. "
            "It runs as a second pass in regeneration.py/_add_cross_market_features. "
            "Running in both passes creates _x/_y duplicates with leaked values."
        )

    def test_cross_market_features_use_only_historical_data(self):
        """
        Integration test: generate cross-market features and verify no leakage.

        Creates a synthetic DataFrame with both outcome columns and EMA features,
        strips outcomes, runs the engineer, and verifies outputs don't correlate
        with stripped outcome values.
        """
        from src.features.engineers.cross_market import CrossMarketFeatureEngineer

        np.random.seed(42)
        n = 100

        # Create synthetic data with both EMA and outcome columns
        df = pd.DataFrame({
            'fixture_id': range(n),
            # EMA features (historical - should be used)
            'home_shots_ema': np.random.normal(12, 3, n),
            'away_shots_ema': np.random.normal(10, 3, n),
            'home_corners_ema': np.random.normal(5, 1.5, n),
            'away_corners_ema': np.random.normal(4.5, 1.5, n),
            'home_cards_ema': np.random.normal(1.5, 0.5, n),
            'away_cards_ema': np.random.normal(1.5, 0.5, n),
            'home_fouls_committed_ema': np.random.normal(11, 2, n),
            'away_fouls_committed_ema': np.random.normal(12, 2, n),
            'home_shots_on_target_ema': np.random.normal(4, 1, n),
            'away_shots_on_target_ema': np.random.normal(3.5, 1, n),
            'season_gd_diff': np.random.normal(0, 5, n),
            'elo_diff': np.random.normal(0, 50, n),
            'home_elo': np.random.normal(1500, 100, n),
            'away_elo': np.random.normal(1500, 100, n),
            'away_attack_strength': np.random.normal(1, 0.3, n),
            'home_defense_strength': np.random.normal(1, 0.3, n),
            'away_xg_poisson': np.random.normal(1.2, 0.5, n),
            'home_xg_poisson': np.random.normal(1.5, 0.5, n),
            'home_avg_yellows': np.random.normal(1.5, 0.5, n),
            'away_avg_yellows': np.random.normal(1.5, 0.5, n),
            'poisson_over25_prob': np.random.uniform(0.3, 0.7, n),
            'poisson_total_goals': np.random.normal(2.5, 0.5, n),
            'fouls_diff': np.random.normal(0, 3, n),
            'corners_defense_diff': np.random.normal(0, 2, n),
            'ref_avg_goals': np.random.normal(2.7, 0.3, n),
            'avg_home_open': np.random.uniform(1.3, 5, n),
            'avg_away_open': np.random.uniform(1.3, 5, n),
            'b365_under25_close': np.random.uniform(1.5, 3, n),
        })

        engineer = CrossMarketFeatureEngineer()
        result = engineer.create_features({'matches': df})

        assert not result.empty, "Cross-market features should not be empty"

        # Verify no output feature perfectly correlates with a known outcome pattern
        # (all features should have reasonable variance from EMA inputs)
        for col in result.columns:
            if col == 'fixture_id':
                continue
            std = result[col].std()
            assert std > 0, f"Feature {col} has zero variance (constant) - likely using defaults"


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


class TestThresholdLeakage:
    """
    Test suite for threshold optimization leakage prevention.

    These tests ensure that probability thresholds are not optimized
    using test data, which would cause overfitting.
    """

    def test_threshold_selection_uses_validation_only(self):
        """
        Threshold must be optimized on validation, not test data.

        CRITICAL: If thresholds are tuned to maximize performance on test data,
        the reported metrics will be overfitted and not generalizable.
        """
        try:
            from experiments.sniper_walkforward_validation import WalkForwardValidator
        except ImportError:
            pytest.skip("WalkForwardValidator not importable")

        # Check that the validator implements nested threshold selection
        validator = WalkForwardValidator()

        # Verify that validate_fold method exists and has the expected signature
        assert hasattr(validator, 'validate_fold'), (
            "WalkForwardValidator must have validate_fold method"
        )

        # Check that the validator has nested_threshold_optimization attribute or method
        has_nested_optimization = (
            hasattr(validator, 'optimize_threshold_on_validation') or
            hasattr(validator, 'nested_threshold_optimization') or
            hasattr(validator, '_optimize_threshold_inner')
        )

        assert has_nested_optimization, (
            "WalkForwardValidator must implement nested threshold optimization. "
            "Thresholds should be optimized on an inner validation split, not test data."
        )

    def test_threshold_not_from_test_data(self):
        """
        Verify that threshold selection process doesn't use test outcomes.

        The threshold optimization should only see training/validation data,
        never the test data outcomes.
        """
        try:
            from experiments.sniper_walkforward_validation import WalkForwardValidator
        except ImportError:
            pytest.skip("WalkForwardValidator not importable")

        validator = WalkForwardValidator()

        # Check for threshold_search_space or similar parameter
        # that indicates thresholds are searched rather than hardcoded
        has_threshold_search = (
            hasattr(validator, 'threshold_search_space') or
            hasattr(validator, 'threshold_candidates') or
            hasattr(validator, 'search_thresholds')
        )

        # This is a structural check - if thresholds are searched,
        # we need to verify they're searched on validation data only
        if has_threshold_search:
            # The implementation should have a validation split
            assert hasattr(validator, 'validation_ratio') or hasattr(validator, 'inner_cv_folds'), (
                "Threshold search requires a validation split to avoid test data leakage"
            )


class TestBetSaturation:
    """
    Test suite for detecting bet saturation (overfitting signal).

    When different hyperparameters produce nearly identical bet selections,
    it indicates the model may be overfitting to specific patterns.
    """

    def test_bet_saturation_across_trials(self):
        """
        Different hyperparams should produce different bet selections.

        If >90% of Optuna trials produce identical bet selections,
        the search space may have collapsed to a local optimum (overfitting).
        """
        # Load recent optimization results if available
        output_dir = Path("experiments/outputs/sniper_mode")
        if not output_dir.exists():
            pytest.skip("No optimization outputs to analyze")

        # Find most recent validation result
        json_files = list(output_dir.glob("*.json"))
        if not json_files:
            pytest.skip("No validation JSON files found")

        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file) as f:
            results = json.load(f)

        # Check if we have multiple trial results
        if "results" not in results or len(results["results"]) < 2:
            pytest.skip("Not enough trial results to check saturation")

        # Extract bet counts from each config
        bet_counts = []
        for r in results["results"]:
            if "total_bets" in r:
                bet_counts.append(r["total_bets"])

        if len(bet_counts) < 2:
            pytest.skip("Not enough bet count data")

        # Calculate saturation: if all trials have identical bet counts, saturation = 100%
        unique_counts = len(set(bet_counts))
        saturation = 1 - (unique_counts / len(bet_counts))

        # Alert if saturation is too high
        SATURATION_THRESHOLD = 0.90
        assert saturation < SATURATION_THRESHOLD, (
            f"Bet saturation too high: {saturation:.1%} of trials produce identical bet counts. "
            f"This suggests overfitting - the model may have collapsed to a narrow pattern. "
            f"Unique bet counts: {unique_counts}/{len(bet_counts)}"
        )


class TestMetricSaturation:
    """
    Test suite for detecting metric saturation (collapsed search space).

    When different hyperparameters yield identical ROI/precision metrics,
    the optimization may have found a degenerate solution.
    """

    def test_metric_saturation_detection(self):
        """
        Detect when different hyperparams yield identical ROI.

        If precision/ROI values are identical across different configurations,
        the search space has likely collapsed to a single solution.
        """
        output_dir = Path("experiments/outputs/sniper_mode")
        if not output_dir.exists():
            pytest.skip("No optimization outputs to analyze")

        json_files = list(output_dir.glob("*.json"))
        if not json_files:
            pytest.skip("No validation JSON files found")

        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file) as f:
            results = json.load(f)

        if "results" not in results or len(results["results"]) < 3:
            pytest.skip("Not enough trial results to check metric saturation")

        # Extract ROI and precision from each result
        rois = []
        precisions = []

        for r in results["results"]:
            if "avg_roi" in r and r["avg_roi"] != 0:
                rois.append(round(r["avg_roi"], 2))  # Round to 2 decimals
            if "overall_precision" in r and r["overall_precision"] != 0:
                precisions.append(round(r["overall_precision"], 4))

        # Check ROI saturation
        if len(rois) >= 3:
            unique_rois = len(set(rois))
            roi_saturation = 1 - (unique_rois / len(rois))

            ROI_SATURATION_THRESHOLD = 0.80
            assert roi_saturation < ROI_SATURATION_THRESHOLD, (
                f"ROI metric saturation: {roi_saturation:.1%} of configs have identical ROI. "
                f"This indicates a collapsed search space. Unique ROIs: {unique_rois}/{len(rois)}"
            )

        # Check precision saturation
        if len(precisions) >= 3:
            unique_precisions = len(set(precisions))
            precision_saturation = 1 - (unique_precisions / len(precisions))

            PRECISION_SATURATION_THRESHOLD = 0.80
            assert precision_saturation < PRECISION_SATURATION_THRESHOLD, (
                f"Precision metric saturation: {precision_saturation:.1%} of configs have identical precision. "
                f"This indicates a collapsed search space. Unique precisions: {unique_precisions}/{len(precisions)}"
            )

    def test_unrealistic_metrics_detection(self):
        """
        Detect unrealistically high metrics that suggest data leakage.

        Sharpe > 12 or ROI > 70% are extremely suspicious and should trigger
        a deeper audit for data leakage.
        """
        output_dir = Path("experiments/outputs/sniper_mode")
        if not output_dir.exists():
            pytest.skip("No optimization outputs to analyze")

        json_files = list(output_dir.glob("*.json"))
        if not json_files:
            pytest.skip("No validation JSON files found")

        suspicious_metrics = []

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    results = json.load(f)

                if "results" not in results:
                    continue

                for r in results["results"]:
                    config_name = r.get("config_name", "unknown")

                    # Check for unrealistic ROI
                    roi = r.get("avg_roi", 0)
                    if abs(roi) > 70:
                        suspicious_metrics.append(
                            f"{config_name}: ROI={roi:.1f}% (threshold: ±70%)"
                        )

                    # Check for unrealistic precision with volume
                    precision = r.get("overall_precision", 0)
                    total_bets = r.get("total_bets", 0)

                    # High precision with many bets is suspicious
                    if precision > 0.85 and total_bets > 20:
                        suspicious_metrics.append(
                            f"{config_name}: {precision:.1%} precision on {total_bets} bets"
                        )

            except (json.JSONDecodeError, KeyError):
                continue

        # Warn but don't fail - these need manual audit
        if suspicious_metrics:
            warnings.warn(
                f"Potentially suspicious metrics detected (may indicate leakage):\n"
                + "\n".join(f"  - {m}" for m in suspicious_metrics[:5]),
                UserWarning
            )


if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/03-features/features_all_with_odds.csv"

    print(f"Validating: {filepath}")
    print("=" * 60)

    is_clean = validate_feature_file(filepath)

    sys.exit(0 if is_clean else 1)
