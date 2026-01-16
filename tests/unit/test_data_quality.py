"""Tests for data quality validation and imputation."""

import pandas as pd
import numpy as np
import pytest

from src.features.data_quality import (
    analyze_nan_distribution,
    detect_league,
    impute_with_league_medians,
    add_data_quality_flags,
    validate_features_for_prediction,
    get_prediction_confidence,
    prepare_features_pipeline,
    CRITICAL_EMA_FEATURES,
    LEAGUE_MEDIANS,
)


class TestAnalyzeNanDistribution:
    """Tests for NaN analysis."""

    def test_complete_data(self):
        """Complete data should show 100% completeness."""
        df = pd.DataFrame({
            'home_fouls_committed_ema': [10, 11, 12],
            'away_fouls_committed_ema': [13, 14, 15],
            'other_col': [1, 2, 3],
        })
        report = analyze_nan_distribution(df)

        assert report.total_rows == 3
        assert report.complete_rows == 3
        assert report.completeness_ratio == 1.0
        assert len(report.nan_by_column) == 0

    def test_missing_critical_features(self):
        """Should identify missing critical EMA features."""
        df = pd.DataFrame({
            'home_fouls_committed_ema': [10, np.nan, np.nan],  # 66% missing
            'away_fouls_committed_ema': [13, 14, np.nan],  # 33% missing
            'other_col': [1, 2, 3],
        })
        report = analyze_nan_distribution(df)

        assert report.total_rows == 3
        assert report.complete_rows == 1  # Only first row complete
        assert report.completeness_ratio == pytest.approx(1/3)
        assert 'home_fouls_committed_ema' in report.critical_missing

    def test_no_critical_features(self):
        """Should handle datasets without critical features."""
        df = pd.DataFrame({
            'goals_scored': [1, 2, np.nan],
            'points': [3, 3, 1],
        })
        report = analyze_nan_distribution(df)

        assert report.total_rows == 3
        assert report.complete_rows == 3  # No critical features to check
        assert len(report.critical_missing) == 0


class TestDetectLeague:
    """Tests for league detection."""

    def test_detect_from_league_column(self):
        """Should detect league from league column."""
        df = pd.DataFrame({
            'league': ['serie_a', 'serie_a'],
        })
        assert detect_league(df) == 'serie_a'

    def test_detect_partial_match(self):
        """Should detect league from partial match."""
        df = pd.DataFrame({
            'league': ['premier_league_2024', 'premier_league_2024'],
        })
        assert detect_league(df) == 'premier_league'

    def test_default_for_unknown(self):
        """Should return default for unknown league."""
        df = pd.DataFrame({
            'league': ['mls', 'mls'],
        })
        assert detect_league(df) == 'default'

    def test_row_specific_detection(self):
        """Should detect league for specific row."""
        df = pd.DataFrame({
            'league': ['serie_a', 'bundesliga', 'premier_league'],
        })
        assert detect_league(df, row_idx=1) == 'bundesliga'


class TestImputeWithLeagueMedians:
    """Tests for league median imputation."""

    def test_imputes_missing_values(self):
        """Should impute missing values with league medians."""
        df = pd.DataFrame({
            'league': ['serie_a', 'serie_a', 'serie_a'],
            'home_fouls_committed_ema': [10, np.nan, np.nan],
        })
        result, imputed_cols = impute_with_league_medians(df)

        assert 'home_fouls_committed_ema' in imputed_cols
        # Should be imputed with Serie A median
        assert result.iloc[1]['home_fouls_committed_ema'] == LEAGUE_MEDIANS['serie_a']['fouls_committed_ema']
        assert result.iloc[2]['home_fouls_committed_ema'] == LEAGUE_MEDIANS['serie_a']['fouls_committed_ema']

    def test_preserves_existing_values(self):
        """Should not modify existing values."""
        df = pd.DataFrame({
            'league': ['serie_a', 'serie_a'],
            'home_fouls_committed_ema': [15.0, np.nan],
        })
        result, _ = impute_with_league_medians(df)

        assert result.iloc[0]['home_fouls_committed_ema'] == 15.0

    def test_inplace_option(self):
        """Should modify in place when inplace=True."""
        df = pd.DataFrame({
            'league': ['default', 'default'],
            'home_fouls_committed_ema': [np.nan, np.nan],
        })
        original_id = id(df)
        result, _ = impute_with_league_medians(df, inplace=True)

        assert id(result) == original_id


class TestAddDataQualityFlags:
    """Tests for data quality flag addition."""

    def test_complete_data_flags(self):
        """Should flag complete data correctly."""
        df = pd.DataFrame({
            'home_fouls_committed_ema': [10, 11],
            'away_fouls_committed_ema': [12, 13],
        })
        result = add_data_quality_flags(df)

        assert '_has_complete_ema' in result.columns
        assert '_ema_completeness' in result.columns
        assert result['_has_complete_ema'].all()
        assert (result['_ema_completeness'] == 1.0).all()

    def test_incomplete_data_flags(self):
        """Should flag incomplete data correctly."""
        df = pd.DataFrame({
            'home_fouls_committed_ema': [10, np.nan],
            'away_fouls_committed_ema': [12, 13],
        })
        result = add_data_quality_flags(df)

        assert result.iloc[0]['_has_complete_ema'] == True
        assert result.iloc[1]['_has_complete_ema'] == False
        assert result.iloc[0]['_ema_completeness'] == 1.0
        assert result.iloc[1]['_ema_completeness'] == 0.5


class TestGetPredictionConfidence:
    """Tests for confidence calculation."""

    def test_full_confidence_clean_data(self):
        """Should return full confidence for clean data."""
        row = pd.Series({
            '_is_imputed': False,
            '_ema_completeness': 1.0,
        })
        confidence = get_prediction_confidence(row)
        assert confidence == 1.0

    def test_reduced_confidence_imputed(self):
        """Should reduce confidence for imputed data."""
        row = pd.Series({
            '_is_imputed': True,
            '_ema_completeness': 1.0,
        })
        confidence = get_prediction_confidence(row)
        assert confidence < 1.0
        assert confidence == pytest.approx(0.7)

    def test_scaled_confidence_partial_data(self):
        """Should scale confidence by completeness."""
        row = pd.Series({
            '_is_imputed': False,
            '_ema_completeness': 0.5,
        })
        confidence = get_prediction_confidence(row)
        assert 0.5 < confidence < 1.0

    def test_minimum_confidence(self):
        """Should not go below minimum confidence."""
        row = pd.Series({
            '_is_imputed': True,
            '_ema_completeness': 0.0,
        })
        confidence = get_prediction_confidence(row)
        assert confidence > 0


class TestValidateFeaturesForPrediction:
    """Tests for feature validation pipeline."""

    def test_validation_with_imputation(self):
        """Should validate and impute features."""
        df = pd.DataFrame({
            'league': ['serie_a', 'serie_a'],
            'home_fouls_committed_ema': [10, np.nan],
            'other_feature': [1, 2],
        })
        result, report = validate_features_for_prediction(
            df,
            required_features=['home_fouls_committed_ema', 'other_feature'],
            allow_imputation=True
        )

        # Should have imputed the missing value
        assert not result['home_fouls_committed_ema'].isna().any()
        assert '_has_complete_ema' in result.columns

    def test_validation_without_imputation(self):
        """Should not impute when disabled."""
        df = pd.DataFrame({
            'league': ['serie_a'],
            'home_fouls_committed_ema': [np.nan],
        })
        result, report = validate_features_for_prediction(
            df,
            required_features=['home_fouls_committed_ema'],
            allow_imputation=False
        )

        # Should still have NaN
        assert result['home_fouls_committed_ema'].isna().any()


class TestPrepareFeaturesPipeline:
    """Tests for the full pipeline."""

    def test_splits_complete_and_imputed(self):
        """Should split data into complete and imputed subsets."""
        df = pd.DataFrame({
            'league': ['serie_a', 'serie_a', 'serie_a', 'serie_a'],
            'home_fouls_committed_ema': [10, 11, np.nan, np.nan],
            'away_fouls_committed_ema': [12, 13, 14, np.nan],
        })

        complete_df, imputed_df, report = prepare_features_pipeline(
            df,
            model_features=['home_fouls_committed_ema', 'away_fouls_committed_ema'],
            min_completeness=0.3
        )

        # First two rows should be complete
        assert len(complete_df) == 2
        # One row has partial data (50% completeness), meets 0.3 threshold
        # Last row has 0% original completeness, doesn't meet threshold after flag calc
        assert len(imputed_df) >= 1  # At least partial data row
        # Total should be >= 3 (2 complete + at least 1 imputed)
        assert len(complete_df) + len(imputed_df) >= 3


class TestLeagueMediansConfiguration:
    """Tests for league medians configuration."""

    def test_all_leagues_have_required_keys(self):
        """All leagues should have required median values."""
        required_keys = ['fouls_committed_ema', 'shots_total_ema', 'shots_on_ema', 'rating_ema']

        for league, medians in LEAGUE_MEDIANS.items():
            for key in required_keys:
                assert key in medians, f"{league} missing {key}"

    def test_medians_are_reasonable(self):
        """Median values should be in reasonable ranges."""
        for league, medians in LEAGUE_MEDIANS.items():
            # Fouls per team per match: typically 10-15
            assert 8 <= medians['fouls_committed_ema'] <= 18, f"{league} fouls out of range"
            # Shots per team per match: typically 10-15
            assert 8 <= medians['shots_total_ema'] <= 18, f"{league} shots out of range"
            # Shots on target: typically 3-6
            assert 2 <= medians['shots_on_ema'] <= 8, f"{league} shots_on out of range"
            # Rating: typically 6.5-7.2
            assert 6 <= medians['rating_ema'] <= 8, f"{league} rating out of range"
