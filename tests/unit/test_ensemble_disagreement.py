"""Unit tests for DisagreementEnsemble and factory."""

import numpy as np
import pytest

from src.ml.ensemble_disagreement import (
    AdaptiveDisagreementEnsemble,
    DisagreementEnsemble,
    create_disagreement_ensemble,
)


class _FakeModel:
    """Minimal model stub returning fixed probabilities."""

    def __init__(self, probs):
        self._probs = np.asarray(probs)

    def predict_proba(self, X):
        # Return (n_samples, 2) like sklearn classifiers
        pos = np.broadcast_to(self._probs, (X.shape[0],))
        return np.column_stack([1 - pos, pos])


class _FakeModelPredict:
    """Model without predict_proba, only predict."""

    def __init__(self, probs):
        self._probs = np.asarray(probs)

    def predict(self, X):
        return np.broadcast_to(self._probs, (X.shape[0],))


def _make_X(n=5):
    return np.random.RandomState(42).randn(n, 3)


class TestDisagreementEnsemble:
    def test_predict_proba_averages_models(self):
        """Average of two models returning 0.6 and 0.8 should be 0.7."""
        m1 = _FakeModel(0.6)
        m2 = _FakeModel(0.8)
        ens = DisagreementEnsemble(models=[m1, m2])
        probs = ens.predict_proba(_make_X())
        np.testing.assert_allclose(probs, 0.7, atol=1e-10)

    def test_predict_proba_raises_when_no_models(self):
        ens = DisagreementEnsemble()
        with pytest.raises(ValueError, match="No models"):
            ens.predict_proba(_make_X())

    def test_add_model(self):
        ens = DisagreementEnsemble()
        assert not ens._is_fitted
        ens.add_model(_FakeModel(0.5), name="test_model")
        assert ens._is_fitted
        assert ens.model_names[-1] == "test_model"

    def test_predict_proba_handles_predict_only_model(self):
        """Models with only predict() should still work."""
        m = _FakeModelPredict(0.65)
        ens = DisagreementEnsemble(models=[m])
        probs = ens.predict_proba(_make_X())
        np.testing.assert_allclose(probs, 0.65, atol=1e-10)

    def test_disagreement_std_method(self):
        """Models returning same value should have 0 disagreement."""
        m1 = _FakeModel(0.7)
        m2 = _FakeModel(0.7)
        ens = DisagreementEnsemble(models=[m1, m2], agreement_method='std')
        X = _make_X()
        result = ens.predict_with_disagreement(X, np.full(5, 0.5))
        np.testing.assert_allclose(result['disagreement'], 0.0, atol=1e-10)

    def test_disagreement_range_method(self):
        m1 = _FakeModel(0.6)
        m2 = _FakeModel(0.8)
        ens = DisagreementEnsemble(models=[m1, m2], agreement_method='range')
        X = _make_X()
        result = ens.predict_with_disagreement(X, np.full(5, 0.5))
        np.testing.assert_allclose(result['disagreement'], 0.2, atol=1e-10)

    def test_disagreement_iqr_method(self):
        m1 = _FakeModel(0.5)
        m2 = _FakeModel(0.6)
        m3 = _FakeModel(0.7)
        m4 = _FakeModel(0.8)
        ens = DisagreementEnsemble(models=[m1, m2, m3, m4], agreement_method='iqr')
        X = _make_X()
        result = ens.predict_with_disagreement(X, np.full(5, 0.5))
        assert np.all(result['disagreement'] > 0)

    def test_bet_signal_all_conditions_met(self):
        """When models agree, beat market, and prob in range -> bet."""
        m1 = _FakeModel(0.65)
        m2 = _FakeModel(0.65)
        ens = DisagreementEnsemble(
            models=[m1, m2],
            agreement_threshold=0.10,
            min_edge_vs_market=0.03,
            min_probability=0.50,
            max_probability=0.85,
        )
        X = _make_X()
        market = np.full(5, 0.50)  # models at 0.65 > market 0.50 + 0.03 edge
        result = ens.predict_with_disagreement(X, market)
        assert np.all(result['bet_signal'])

    def test_bet_signal_no_edge(self):
        """When edge < threshold -> no bet."""
        m1 = _FakeModel(0.52)
        m2 = _FakeModel(0.52)
        ens = DisagreementEnsemble(
            models=[m1, m2],
            min_edge_vs_market=0.05,
            min_probability=0.50,
        )
        X = _make_X()
        market = np.full(5, 0.50)  # edge = 0.02 < 0.05
        result = ens.predict_with_disagreement(X, market)
        assert not np.any(result['bet_signal'])

    def test_bet_signal_disagreement_too_high(self):
        """When models disagree strongly -> no bet."""
        m1 = _FakeModel(0.50)
        m2 = _FakeModel(0.90)
        ens = DisagreementEnsemble(
            models=[m1, m2],
            agreement_threshold=0.05,
            min_edge_vs_market=0.03,
            min_probability=0.50,
        )
        X = _make_X()
        market = np.full(5, 0.50)
        result = ens.predict_with_disagreement(X, market)
        assert not np.any(result['bet_signal'])

    def test_bet_signal_prob_below_min(self):
        """When avg prob below minimum -> no bet."""
        m1 = _FakeModel(0.40)
        m2 = _FakeModel(0.40)
        ens = DisagreementEnsemble(
            models=[m1, m2],
            min_probability=0.50,
            min_edge_vs_market=0.03,
        )
        X = _make_X()
        market = np.full(5, 0.30)
        result = ens.predict_with_disagreement(X, market)
        assert not np.any(result['bet_signal'])

    def test_bet_signal_prob_above_max(self):
        """When avg prob above maximum -> no bet."""
        m1 = _FakeModel(0.95)
        m2 = _FakeModel(0.95)
        ens = DisagreementEnsemble(
            models=[m1, m2],
            max_probability=0.85,
            min_edge_vs_market=0.03,
            min_probability=0.50,
        )
        X = _make_X()
        market = np.full(5, 0.50)
        result = ens.predict_with_disagreement(X, market)
        assert not np.any(result['bet_signal'])

    def test_predict_returns_binary(self):
        m1 = _FakeModel(0.65)
        m2 = _FakeModel(0.65)
        ens = DisagreementEnsemble(models=[m1, m2], min_edge_vs_market=0.03)
        X = _make_X()
        market = np.full(5, 0.50)
        preds = ens.predict(X, market)
        assert preds.dtype in (np.int64, np.int32, int)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_get_agreement_analysis_returns_dataframe(self):
        m1 = _FakeModel(0.65)
        m2 = _FakeModel(0.60)
        ens = DisagreementEnsemble(
            models=[m1, m2], model_names=["xgb", "lgb"]
        )
        X = _make_X()
        market = np.full(5, 0.50)
        df = ens.get_agreement_analysis(X, market)
        assert 'avg_prob' in df.columns
        assert 'prob_xgb' in df.columns
        assert 'prob_lgb' in df.columns
        assert len(df) == 5

    def test_confidence_score_zero_for_no_bet(self):
        """Confidence should be 0 when bet_signal is False."""
        m1 = _FakeModel(0.40)
        m2 = _FakeModel(0.40)
        ens = DisagreementEnsemble(models=[m1, m2], min_probability=0.50)
        X = _make_X()
        market = np.full(5, 0.30)
        conf = ens.get_confidence_score(X, market)
        np.testing.assert_allclose(conf, 0.0, atol=1e-10)

    def test_confidence_score_positive_for_bet(self):
        m1 = _FakeModel(0.70)
        m2 = _FakeModel(0.70)
        ens = DisagreementEnsemble(
            models=[m1, m2],
            agreement_threshold=0.10,
            min_edge_vs_market=0.03,
            min_probability=0.50,
        )
        X = _make_X()
        market = np.full(5, 0.50)
        conf = ens.get_confidence_score(X, market)
        assert np.all(conf > 0)

    def test_individual_probs_in_result(self):
        m1 = _FakeModel(0.60)
        m2 = _FakeModel(0.70)
        ens = DisagreementEnsemble(
            models=[m1, m2], model_names=["a", "b"]
        )
        X = _make_X()
        result = ens.predict_with_disagreement(X, np.full(5, 0.5))
        assert "a" in result['individual_probs']
        assert "b" in result['individual_probs']
        np.testing.assert_allclose(result['individual_probs']['a'], 0.6, atol=1e-10)


class TestAdaptiveDisagreementEnsemble:
    def test_adapt_tightens_when_losing(self):
        m1 = _FakeModel(0.6)
        ens = AdaptiveDisagreementEnsemble(
            models=[m1],
            base_agreement_threshold=0.10,
            base_edge_threshold=0.03,
        )
        # Record 25 losing bets
        for _ in range(25):
            ens.record_result(bet=True, won=False)
        ens.adapt_thresholds(market_volatility=0.0, recent_win_rate=0.30)
        assert ens.agreement_threshold < 0.10
        assert ens.min_edge_vs_market > 0.03

    def test_adapt_loosens_when_winning(self):
        m1 = _FakeModel(0.6)
        ens = AdaptiveDisagreementEnsemble(
            models=[m1],
            base_agreement_threshold=0.10,
            base_edge_threshold=0.03,
        )
        for _ in range(25):
            ens.record_result(bet=True, won=True)
        ens.adapt_thresholds(market_volatility=0.0, recent_win_rate=0.65)
        assert ens.agreement_threshold > 0.10

    def test_volatility_widens_thresholds(self):
        m1 = _FakeModel(0.6)
        ens = AdaptiveDisagreementEnsemble(
            models=[m1],
            base_agreement_threshold=0.10,
        )
        ens.adapt_thresholds(market_volatility=0.5)
        assert ens.agreement_threshold > 0.10

    def test_get_recent_win_rate_default(self):
        ens = AdaptiveDisagreementEnsemble(models=[_FakeModel(0.5)])
        assert ens.get_recent_win_rate() == 0.5  # default when no history

    def test_record_result_tracks_wins(self):
        ens = AdaptiveDisagreementEnsemble(models=[_FakeModel(0.5)])
        ens.record_result(bet=True, won=True)
        ens.record_result(bet=True, won=False)
        ens.record_result(bet=False, won=True)  # not a bet, not tracked
        assert len(ens._win_history) == 2
        assert ens.get_recent_win_rate() == 0.5


class TestCreateDisagreementEnsemble:
    def test_conservative_strategy(self):
        models = [_FakeModel(0.6)]
        ens = create_disagreement_ensemble(models, strategy='conservative')
        assert ens.agreement_threshold == 0.08
        assert ens.min_edge_vs_market == 0.05
        assert ens.min_probability == 0.55

    def test_balanced_strategy(self):
        models = [_FakeModel(0.6)]
        ens = create_disagreement_ensemble(models, strategy='balanced')
        assert ens.agreement_threshold == 0.12
        assert ens.min_edge_vs_market == 0.03

    def test_aggressive_strategy(self):
        models = [_FakeModel(0.6)]
        ens = create_disagreement_ensemble(models, strategy='aggressive')
        assert ens.agreement_threshold == 0.15
        assert ens.min_edge_vs_market == 0.02
        assert ens.min_probability == 0.45

    def test_unknown_strategy_falls_back_to_balanced(self):
        models = [_FakeModel(0.6)]
        ens = create_disagreement_ensemble(models, strategy='unknown')
        assert ens.agreement_threshold == 0.12

    def test_model_names_passed_through(self):
        models = [_FakeModel(0.6), _FakeModel(0.7)]
        ens = create_disagreement_ensemble(
            models, model_names=["xgb", "lgb"], strategy='balanced'
        )
        assert ens.model_names == ["xgb", "lgb"]
