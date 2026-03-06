"""Tests for smart fillna in feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest


class TestSmartFillStat:
    """Test smart fill logic: only fill NaN->0 when match has recorded results."""

    def _apply_smart_fill(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Replicate the smart fill logic from feature_eng_pipeline.py."""
        has_result = df["ft_home"].notna() & df["ft_away"].notna()
        s = df[col]
        return s.where(s.notna() | ~has_result, 0)

    def test_fill_with_match_result_present(self):
        df = pd.DataFrame(
            {
                "ft_home": [2],
                "ft_away": [1],
                "home_yellow_cards": [np.nan],
            }
        )
        result = self._apply_smart_fill(df, "home_yellow_cards")
        assert result.iloc[0] == 0.0

    def test_no_fill_when_no_match_result(self):
        df = pd.DataFrame(
            {
                "ft_home": [np.nan],
                "ft_away": [np.nan],
                "home_yellow_cards": [np.nan],
            }
        )
        result = self._apply_smart_fill(df, "home_yellow_cards")
        assert pd.isna(result.iloc[0])

    def test_preserve_actual_values(self):
        df = pd.DataFrame(
            {
                "ft_home": [1],
                "ft_away": [0],
                "home_yellow_cards": [3.0],
            }
        )
        result = self._apply_smart_fill(df, "home_yellow_cards")
        assert result.iloc[0] == 3.0

    def test_mixed_rows(self):
        df = pd.DataFrame(
            {
                "ft_home": [2, 1, np.nan, 0, 3],
                "ft_away": [1, 0, np.nan, 2, 1],
                "home_yellow_cards": [np.nan, 2.0, np.nan, np.nan, 4.0],
            }
        )
        result = self._apply_smart_fill(df, "home_yellow_cards")
        # Row 0: result exists, cards NaN -> fill to 0
        assert result.iloc[0] == 0.0
        # Row 1: result exists, cards=2 -> keep 2
        assert result.iloc[1] == 2.0
        # Row 2: no result, cards NaN -> keep NaN
        assert pd.isna(result.iloc[2])
        # Row 3: result exists, cards NaN -> fill to 0
        assert result.iloc[3] == 0.0
        # Row 4: result exists, cards=4 -> keep 4
        assert result.iloc[4] == 4.0

    def test_zero_cards_preserved(self):
        """Genuine zero cards should be preserved, not confused with NaN."""
        df = pd.DataFrame(
            {
                "ft_home": [1],
                "ft_away": [0],
                "home_yellow_cards": [0.0],
            }
        )
        result = self._apply_smart_fill(df, "home_yellow_cards")
        assert result.iloc[0] == 0.0
