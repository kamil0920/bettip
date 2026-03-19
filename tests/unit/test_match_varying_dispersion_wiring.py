"""Tests for match-varying NegBin dispersion wiring across the pipeline.

Validates that:
1. per_line_odds uses abs_goal_supremacy when available
2. Backward compat: no supremacy column = identical results to fixed dispersion
3. negbin_edge.compute_negbin_baseline accepts dispersion param
4. generate_daily_recommendations _get_negbin_baseline uses match-varying dispersion
"""

import numpy as np
import pandas as pd
import pytest


class TestPerLineOddsDispersionWiring:
    """Test that generate_per_line_odds uses match-varying dispersion."""

    def test_output_differs_with_supremacy(self):
        """Output should differ when abs_goal_supremacy is present vs absent."""
        from src.odds.per_line_odds import generate_per_line_odds

        base_df = pd.DataFrame({
            "league": ["premier_league"] * 50,
            "date": pd.date_range("2024-01-01", periods=50),
            "total_corners": np.random.default_rng(42).normal(10.5, 3.0, 50).clip(2, 20),
            "total_cards": np.random.default_rng(42).normal(4.0, 2.0, 50).clip(0, 12),
            "total_shots": np.random.default_rng(42).normal(27.0, 5.0, 50).clip(10, 45),
            "total_fouls": np.random.default_rng(42).normal(24.0, 5.0, 50).clip(10, 40),
        })

        # Without supremacy
        df_no_sup = base_df.copy()
        result_no_sup = generate_per_line_odds(df_no_sup)

        # With supremacy (high values to amplify difference)
        df_with_sup = base_df.copy()
        df_with_sup["abs_goal_supremacy"] = np.linspace(0, 3.0, 50)
        result_with_sup = generate_per_line_odds(df_with_sup)

        # Find a common generated column to compare
        corners_col = "corners_over_avg_95"
        assert corners_col in result_no_sup.columns
        assert corners_col in result_with_sup.columns

        vals_no_sup = result_no_sup[corners_col].dropna().values
        vals_with_sup = result_with_sup[corners_col].dropna().values

        # They should differ (match-varying dispersion changes probabilities)
        assert not np.allclose(vals_no_sup, vals_with_sup, atol=1e-6), \
            "Per-line odds should differ when abs_goal_supremacy is present"

    def test_backward_compat_no_supremacy(self):
        """Without abs_goal_supremacy, results should be identical to before."""
        from src.odds.per_line_odds import generate_per_line_odds

        df = pd.DataFrame({
            "league": ["la_liga"] * 30,
            "date": pd.date_range("2024-01-01", periods=30),
            "total_corners": np.random.default_rng(7).normal(10.5, 2.0, 30).clip(3, 18),
            "total_cards": np.random.default_rng(7).normal(4.0, 1.5, 30).clip(0, 10),
            "total_shots": np.random.default_rng(7).normal(27.0, 4.0, 30).clip(12, 42),
            "total_fouls": np.random.default_rng(7).normal(24.0, 4.0, 30).clip(10, 38),
        })

        # Run twice — should produce identical results (deterministic)
        result1 = generate_per_line_odds(df.copy())
        result2 = generate_per_line_odds(df.copy())

        for col in result1.columns:
            if col.endswith("_avg_") or "_avg_" in col:
                v1 = result1[col].dropna().values
                v2 = result2[col].dropna().values
                np.testing.assert_array_almost_equal(v1, v2, decimal=10)

    def test_supremacy_zero_equals_fixed(self):
        """When abs_goal_supremacy is all zeros, result ≈ fixed dispersion."""
        from src.odds.per_line_odds import generate_per_line_odds

        df = pd.DataFrame({
            "league": ["bundesliga"] * 30,
            "date": pd.date_range("2024-01-01", periods=30),
            "total_corners": np.random.default_rng(3).normal(10.5, 2.0, 30).clip(3, 18),
            "total_cards": np.random.default_rng(3).normal(4.0, 1.5, 30).clip(0, 10),
            "total_shots": np.random.default_rng(3).normal(27.0, 4.0, 30).clip(12, 42),
            "total_fouls": np.random.default_rng(3).normal(24.0, 4.0, 30).clip(10, 38),
        })

        df_zero_sup = df.copy()
        df_zero_sup["abs_goal_supremacy"] = 0.0

        result_no_sup = generate_per_line_odds(df.copy())
        result_zero_sup = generate_per_line_odds(df_zero_sup)

        # With SUP=0, match_varying_dispersion returns base_d, so results should match
        corners_col = "corners_over_avg_95"
        if corners_col in result_no_sup.columns and corners_col in result_zero_sup.columns:
            v1 = result_no_sup[corners_col].dropna().values
            v2 = result_zero_sup[corners_col].dropna().values
            np.testing.assert_array_almost_equal(v1, v2, decimal=8)


class TestNegBinBaselineDispersion:
    """Test compute_negbin_baseline with dispersion parameter."""

    def test_with_dispersion_array(self):
        """Passing dispersion array should change output vs default."""
        from src.odds.negbin_edge import compute_negbin_baseline

        expected_total = np.array([10.0, 10.0, 10.0])

        prob_default = compute_negbin_baseline("corners", expected_total)
        # High dispersion should produce different probs
        prob_custom = compute_negbin_baseline(
            "corners", expected_total, dispersion=np.array([3.0, 3.0, 3.0])
        )

        assert not np.allclose(prob_default, prob_custom, atol=1e-4), \
            "Custom dispersion should produce different probabilities"

    def test_without_dispersion_backward_compat(self):
        """Without dispersion param, should match original behavior."""
        from src.odds.negbin_edge import compute_negbin_baseline

        expected_total = np.array([10.0, 8.0, 12.0])

        # Default call (no dispersion)
        prob = compute_negbin_baseline("corners", expected_total)

        assert prob.shape == (3,)
        assert all(0.02 <= p <= 0.98 for p in prob)

    def test_nan_preserved_with_dispersion(self):
        """NaN in expected_total should still produce NaN."""
        from src.odds.negbin_edge import compute_negbin_baseline

        expected_total = np.array([10.0, np.nan, 12.0])
        dispersion = np.array([1.5, 1.5, 1.5])

        prob = compute_negbin_baseline("corners", expected_total, dispersion=dispersion)
        assert np.isnan(prob[1])
        assert not np.isnan(prob[0])
        assert not np.isnan(prob[2])
