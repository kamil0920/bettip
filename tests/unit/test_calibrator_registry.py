"""Unit tests for calibrator registry and factory."""

import pytest

from src.calibration.calibration import (
    CALIBRATOR_REGISTRY,
    BetaCalibrator,
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
    VennAbersCalibrator,
    get_calibrator,
)


class TestCalibratorRegistry:
    def test_registry_has_all_methods(self):
        """Test all expected methods are registered."""
        expected = {"sigmoid", "platt", "isotonic", "beta", "temperature",
                    "venn_abers", "venn-abers", "va"}
        assert expected == set(CALIBRATOR_REGISTRY.keys())

    def test_sigmoid_creates_platt(self):
        """Test 'sigmoid' creates PlattScaling."""
        cal = get_calibrator("sigmoid")
        assert isinstance(cal, PlattScaling)

    def test_platt_creates_platt(self):
        """Test 'platt' alias creates PlattScaling."""
        cal = get_calibrator("platt")
        assert isinstance(cal, PlattScaling)

    def test_isotonic_creates_isotonic(self):
        """Test 'isotonic' creates IsotonicCalibrator."""
        cal = get_calibrator("isotonic")
        assert isinstance(cal, IsotonicCalibrator)

    def test_beta_creates_beta(self):
        """Test 'beta' creates BetaCalibrator."""
        cal = get_calibrator("beta")
        assert isinstance(cal, BetaCalibrator)

    def test_temperature_creates_temperature(self):
        """Test 'temperature' creates TemperatureScaling."""
        cal = get_calibrator("temperature")
        assert isinstance(cal, TemperatureScaling)

    def test_venn_abers_creates_va(self):
        """Test 'venn_abers' creates VennAbersCalibrator."""
        cal = get_calibrator("venn_abers")
        assert isinstance(cal, VennAbersCalibrator)

    def test_va_alias(self):
        """Test 'va' alias creates VennAbersCalibrator."""
        cal = get_calibrator("va")
        assert isinstance(cal, VennAbersCalibrator)

    def test_case_insensitive(self):
        """Test method names are case-insensitive."""
        cal = get_calibrator("SIGMOID")
        assert isinstance(cal, PlattScaling)

    def test_unknown_method_raises(self):
        """Test unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown calibration method"):
            get_calibrator("nonexistent")

    def test_each_call_creates_new_instance(self):
        """Test factory creates new instances each time."""
        cal1 = get_calibrator("sigmoid")
        cal2 = get_calibrator("sigmoid")
        assert cal1 is not cal2
