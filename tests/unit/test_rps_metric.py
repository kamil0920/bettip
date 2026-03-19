"""Tests for Ranked Probability Score (RPS) metric."""

import numpy as np
import pytest

from src.ml.metrics import ranked_probability_score


class TestRankedProbabilityScore:
    """Test RPS metric for ordered multi-class predictions."""

    def test_perfect_prediction(self):
        """Perfect one-hot predictions should give RPS = 0."""
        y_true = np.array([0, 1, 2])
        y_prob = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert ranked_probability_score(y_true, y_prob) == 0.0

    def test_worst_prediction_3class(self):
        """Worst case: predicting opposite end of ordered scale."""
        y_true = np.array([0, 2])
        y_prob = np.array([[0, 0, 1], [1, 0, 0]])
        rps = ranked_probability_score(y_true, y_prob)
        assert rps > 0.0
        # For 3-class, worst possible is 1.0 (predicting exact opposite)
        assert rps == pytest.approx(1.0, abs=1e-10)

    def test_uniform_prediction(self):
        """Uniform predictions should give intermediate RPS."""
        y_true = np.array([0, 1, 2])
        y_prob = np.full((3, 3), 1 / 3)
        rps = ranked_probability_score(y_true, y_prob)
        assert 0.0 < rps < 1.0

    def test_near_miss_better_than_far_miss(self):
        """Predicting adjacent class should have lower RPS than distant class."""
        # True = class 0
        y_true = np.array([0])

        # Near miss: predict class 1
        y_prob_near = np.array([[0, 1, 0]])
        rps_near = ranked_probability_score(y_true, y_prob_near)

        # Far miss: predict class 2
        y_prob_far = np.array([[0, 0, 1]])
        rps_far = ranked_probability_score(y_true, y_prob_far)

        assert rps_near < rps_far, "Near miss should have lower RPS than far miss"

    def test_1d_input_raises(self):
        """1D y_prob should raise ValueError."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        with pytest.raises(ValueError, match="2D"):
            ranked_probability_score(y_true, y_prob)

    def test_binary_case(self):
        """Should work for K=2 (binary) case."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
        rps = ranked_probability_score(y_true, y_prob)
        assert 0.0 <= rps <= 1.0
