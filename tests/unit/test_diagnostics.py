"""Unit tests for model diagnostics (learning curves)."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.ml.diagnostics import (
    diagnose_all_models,
    generate_learning_curve_plot,
    generate_learning_curves,
)


@pytest.fixture
def binary_data():
    """Small binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=42
    )
    return X, y


class TestGenerateLearningCurves:
    def test_returns_correct_keys(self, binary_data):
        X, y = binary_data
        result = generate_learning_curves(
            LogisticRegression(max_iter=200), X, y,
            cv_folds=3, n_points=4,
        )
        assert 'train_sizes' in result
        assert 'train_scores_mean' in result
        assert 'val_scores_mean' in result
        assert 'diagnosis' in result
        assert 'gap' in result
        assert 'scoring' in result

    def test_diagnosis_is_valid_string(self, binary_data):
        X, y = binary_data
        result = generate_learning_curves(
            LogisticRegression(max_iter=200), X, y,
            cv_folds=3, n_points=4,
        )
        assert result['diagnosis'] in ('overfit', 'underfit', 'more_data', 'good')

    def test_train_sizes_match_n_points(self, binary_data):
        X, y = binary_data
        result = generate_learning_curves(
            LogisticRegression(max_iter=200), X, y,
            cv_folds=3, n_points=5,
        )
        assert len(result['train_sizes']) == 5

    def test_gap_is_non_negative(self, binary_data):
        X, y = binary_data
        result = generate_learning_curves(
            LogisticRegression(max_iter=200), X, y,
            cv_folds=3, n_points=4,
        )
        assert result['gap'] >= 0


class TestGenerateLearningCurvePlot:
    def test_returns_figure(self, binary_data):
        X, y = binary_data
        result = generate_learning_curves(
            LogisticRegression(max_iter=200), X, y,
            cv_folds=3, n_points=4,
        )
        fig = generate_learning_curve_plot(result, title='Test')
        assert fig is not None

    def test_saves_to_file(self, binary_data, tmp_path):
        X, y = binary_data
        result = generate_learning_curves(
            LogisticRegression(max_iter=200), X, y,
            cv_folds=3, n_points=4,
        )
        out = str(tmp_path / 'test_lc.png')
        generate_learning_curve_plot(result, output_path=out)
        assert (tmp_path / 'test_lc.png').exists()


class TestDiagnoseAllModels:
    def test_returns_per_model_results(self, binary_data):
        X, y = binary_data
        models = {
            'lr': LogisticRegression(max_iter=200),
        }
        results = diagnose_all_models(models, X, y)
        assert 'lr' in results
        assert 'diagnosis' in results['lr']

    def test_saves_plots_when_output_dir(self, binary_data, tmp_path):
        X, y = binary_data
        models = {'lr': LogisticRegression(max_iter=200)}
        diagnose_all_models(models, X, y, output_dir=str(tmp_path))
        assert (tmp_path / 'learning_curve_lr.png').exists()
