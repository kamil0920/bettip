"""Unit tests for SniperConfig validation."""

import pytest

from experiments.sniper.config import SniperConfig


class TestSniperConfigValidation:
    def test_valid_config(self):
        """Test valid config passes validation."""
        cfg = SniperConfig(bet_type="home_win")
        errors = cfg.validate()
        assert errors == []

    def test_missing_bet_type(self):
        """Test empty bet_type is caught."""
        cfg = SniperConfig(bet_type="")
        errors = cfg.validate()
        assert any("bet_type" in e for e in errors)

    def test_min_gt_max_rfe_features(self):
        """Test min > max RFE features is caught."""
        cfg = SniperConfig(
            bet_type="home_win",
            min_rfe_features=80,
            max_rfe_features=20,
        )
        errors = cfg.validate()
        assert any("min_rfe_features" in e for e in errors)

    def test_conflicting_catboost_flags(self):
        """Test only_catboost + no_catboost conflict is caught."""
        cfg = SniperConfig(
            bet_type="home_win",
            only_catboost=True,
            no_catboost=True,
        )
        errors = cfg.validate()
        assert any("catboost" in e.lower() for e in errors)

    def test_holdout_folds_too_high(self):
        """Test n_holdout_folds >= n_folds - 1 is caught."""
        cfg = SniperConfig(
            bet_type="home_win",
            n_folds=5,
            n_holdout_folds=4,
        )
        errors = cfg.validate()
        assert any("holdout" in e.lower() for e in errors)

    def test_invalid_max_ece(self):
        """Test max_ece out of range is caught."""
        cfg = SniperConfig(bet_type="home_win", max_ece=1.5)
        errors = cfg.validate()
        assert any("max_ece" in e for e in errors)

    def test_invalid_n_folds(self):
        """Test n_folds < 3 is caught."""
        cfg = SniperConfig(bet_type="home_win", n_folds=2)
        errors = cfg.validate()
        assert any("n_folds" in e for e in errors)

    def test_invalid_threshold_alpha(self):
        """Test threshold_alpha out of range is caught."""
        cfg = SniperConfig(bet_type="home_win", threshold_alpha=-0.1)
        errors = cfg.validate()
        assert any("threshold_alpha" in e for e in errors)

    def test_valid_edge_case_holdout(self):
        """Test holdout_folds = n_folds - 2 is valid."""
        cfg = SniperConfig(
            bet_type="home_win",
            n_folds=5,
            n_holdout_folds=3,
        )
        errors = cfg.validate()
        assert errors == []

    def test_multiple_errors_reported(self):
        """Test multiple validation errors are all reported."""
        cfg = SniperConfig(
            bet_type="",
            only_catboost=True,
            no_catboost=True,
            min_rfe_features=100,
            max_rfe_features=10,
        )
        errors = cfg.validate()
        assert len(errors) >= 3
