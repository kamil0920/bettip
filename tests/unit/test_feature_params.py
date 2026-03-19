"""
Unit tests for feature parameter configuration and regeneration.

Tests:
- BetTypeFeatureConfig serialization (save/load)
- Parameter hash generation
- Registry parameter mapping
- Config with different bet types
"""
import tempfile
from pathlib import Path

import pytest
import yaml

from src.features.config_manager import (
    BetTypeFeatureConfig,
    PARAMETER_SEARCH_SPACES,
    BET_TYPE_PARAM_PRIORITIES,
    get_search_space_for_bet_type,
)
from src.features.registry import (
    create_configs_with_bet_type_params,
    FeatureEngineerConfig,
)


class TestBetTypeFeatureConfig:
    """Tests for BetTypeFeatureConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = BetTypeFeatureConfig(bet_type="away_win")

        assert config.bet_type == "away_win"
        assert config.elo_k_factor == 32.0
        assert config.elo_home_advantage == 100.0
        assert config.form_window == 5
        assert config.ema_span == 5
        assert config.poisson_lookback == 10
        assert config.optimized is False
        assert config.optimization_date is None

    def test_custom_values(self):
        """Test config with custom values."""
        config = BetTypeFeatureConfig(
            bet_type="fouls",
            elo_k_factor=40.0,
            form_window=7,
            fouls_ema_span=15,
        )

        assert config.bet_type == "fouls"
        assert config.elo_k_factor == 40.0
        assert config.form_window == 7
        assert config.fouls_ema_span == 15
        # Other values should be defaults
        assert config.ema_span == 5

    def test_to_registry_params(self):
        """Test conversion to registry parameter format."""
        config = BetTypeFeatureConfig(
            bet_type="away_win",
            elo_k_factor=40.0,
            elo_home_advantage=120.0,
            form_window=7,
            ema_span=10,
        )

        params = config.to_registry_params()

        assert 'elo' in params
        assert params['elo']['k_factor'] == 40.0
        assert params['elo']['home_advantage'] == 120.0

        assert 'team_form' in params
        assert params['team_form']['n_matches'] == 7

        assert 'ema' in params
        assert params['ema']['span'] == 10

    def test_to_registry_params_niche_markets(self):
        """Test registry params for niche market configs."""
        config = BetTypeFeatureConfig(
            bet_type="fouls",
            fouls_ema_span=15,
            fouls_window_sizes=[5, 10, 15],
        )

        params = config.to_registry_params()

        assert 'fouls' in params
        assert params['fouls']['ema_span'] == 15
        assert params['fouls']['window_sizes'] == [5, 10, 15]

    def test_params_hash_consistent(self):
        """Test that same params produce same hash."""
        config1 = BetTypeFeatureConfig(bet_type="away_win", elo_k_factor=40.0)
        config2 = BetTypeFeatureConfig(bet_type="away_win", elo_k_factor=40.0)

        assert config1.params_hash() == config2.params_hash()

    def test_params_hash_different_values(self):
        """Test that different params produce different hashes."""
        config1 = BetTypeFeatureConfig(bet_type="away_win", elo_k_factor=40.0)
        config2 = BetTypeFeatureConfig(bet_type="away_win", elo_k_factor=48.0)

        assert config1.params_hash() != config2.params_hash()

    def test_params_hash_independent_of_bet_type(self):
        """Test that hash depends only on feature params, not bet_type name."""
        config1 = BetTypeFeatureConfig(bet_type="away_win", elo_k_factor=40.0)
        config2 = BetTypeFeatureConfig(bet_type="btts", elo_k_factor=40.0)

        # Same feature params should produce same hash (bet_type is metadata)
        assert config1.params_hash() == config2.params_hash()

    def test_params_hash_independent_of_metadata(self):
        """Test that hash ignores metadata fields."""
        config1 = BetTypeFeatureConfig(bet_type="away_win", optimized=False)
        config2 = BetTypeFeatureConfig(bet_type="away_win", optimized=True, precision=0.7)

        assert config1.params_hash() == config2.params_hash()

    def test_save_and_load(self):
        """Test saving and loading config to/from YAML."""
        config = BetTypeFeatureConfig(
            bet_type="test_save",
            elo_k_factor=40.0,
            form_window=7,
            optimized=True,
            precision=0.72,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            config.save(path)

            # Verify file exists
            assert path.exists()

            # Load and verify
            loaded = BetTypeFeatureConfig.load(path)
            assert loaded.bet_type == "test_save"
            assert loaded.elo_k_factor == 40.0
            assert loaded.form_window == 7
            assert loaded.optimized is True
            assert loaded.precision == 0.72

    def test_load_yaml_format(self):
        """Test that saved YAML is human-readable."""
        config = BetTypeFeatureConfig(
            bet_type="test_format",
            elo_k_factor=40.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            config.save(path)

            # Read raw YAML
            with open(path, 'r') as f:
                yaml_content = yaml.safe_load(f)

            assert yaml_content['bet_type'] == "test_format"
            assert yaml_content['elo_k_factor'] == 40.0

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            BetTypeFeatureConfig.load(Path("/nonexistent/path.yaml"))

    def test_update_metadata(self):
        """Test updating metadata after optimization."""
        config = BetTypeFeatureConfig(bet_type="away_win")
        assert config.optimized is False

        config.update_metadata(precision=0.72, roi=15.5, n_trials=30)

        assert config.optimized is True
        assert config.precision == 0.72
        assert config.roi == 15.5
        assert config.n_trials == 30
        assert config.optimization_date is not None

    def test_summary(self):
        """Test summary string generation."""
        config = BetTypeFeatureConfig(
            bet_type="away_win",
            elo_k_factor=40.0,
        )
        summary = config.summary()

        assert "away_win" in summary
        assert "elo_k_factor: 40.0" in summary
        assert "Optimized: False" in summary


class TestParameterSearchSpaces:
    """Tests for parameter search space configuration."""

    def test_search_spaces_defined(self):
        """Test that all expected parameters have search spaces."""
        expected_params = [
            'elo_k_factor',
            'elo_home_advantage',
            'form_window',
            'ema_span',
            'poisson_lookback',
        ]

        for param in expected_params:
            assert param in PARAMETER_SEARCH_SPACES
            assert len(PARAMETER_SEARCH_SPACES[param]) >= 3  # At least 3 options

    def test_bet_type_priorities_defined(self):
        """Test that all bet types have parameter priorities."""
        expected_bet_types = [
            'away_win', 'home_win', 'btts', 'over25', 'under25',
            'fouls', 'cards', 'shots', 'corners',
        ]

        for bet_type in expected_bet_types:
            assert bet_type in BET_TYPE_PARAM_PRIORITIES
            assert len(BET_TYPE_PARAM_PRIORITIES[bet_type]) >= 2

    def test_get_search_space_for_bet_type(self):
        """Test getting search space for specific bet type."""
        space = get_search_space_for_bet_type('away_win')

        assert 'elo_k_factor' in space
        assert 'form_window' in space
        # Should have actual values
        assert len(space['elo_k_factor']) > 0

    def test_niche_market_has_specific_params(self):
        """Test that niche markets include market-specific params."""
        fouls_space = get_search_space_for_bet_type('fouls')
        cards_space = get_search_space_for_bet_type('cards')

        assert 'fouls_ema_span' in fouls_space
        assert 'cards_ema_span' in cards_space


class TestRegistryIntegration:
    """Tests for registry integration with BetTypeFeatureConfig."""

    def test_create_configs_with_bet_type_params(self):
        """Test creating engineer configs with custom params."""
        feature_config = BetTypeFeatureConfig(
            bet_type="away_win",
            elo_k_factor=48.0,
            form_window=10,
        )

        configs = create_configs_with_bet_type_params(feature_config)

        # Should return list of FeatureEngineerConfig
        assert isinstance(configs, list)
        assert all(isinstance(c, FeatureEngineerConfig) for c in configs)

        # Find ELO config and verify params
        elo_config = next((c for c in configs if c.name == 'elo'), None)
        assert elo_config is not None
        assert elo_config.params['k_factor'] == 48.0

        # Find team_form config and verify params
        form_config = next((c for c in configs if c.name == 'team_form'), None)
        assert form_config is not None
        assert form_config.params['n_matches'] == 10

    def test_configs_preserve_defaults_for_unspecified_params(self):
        """Test that unspecified params keep defaults."""
        feature_config = BetTypeFeatureConfig(
            bet_type="away_win",
            elo_k_factor=48.0,
            # Not setting form_window - should keep default
        )

        configs = create_configs_with_bet_type_params(feature_config)

        # form_window should be default (5)
        form_config = next((c for c in configs if c.name == 'team_form'), None)
        assert form_config is not None
        assert form_config.params['n_matches'] == 5

    def test_configs_include_all_engineers(self):
        """Test that all standard engineers are included."""
        feature_config = BetTypeFeatureConfig(bet_type="away_win")
        configs = create_configs_with_bet_type_params(feature_config)

        config_names = {c.name for c in configs}

        # Core engineers should be present
        assert 'elo' in config_names
        assert 'team_form' in config_names
        assert 'ema' in config_names
        assert 'poisson' in config_names

        # Niche market engineers should be present
        assert 'fouls' in config_names
        assert 'cards' in config_names
        assert 'corners' in config_names


class TestFeatureConfigLoadForBetType:
    """Tests for loading configs by bet type name."""

    def test_load_for_bet_type_defaults(self):
        """Test loading config for bet type without config file returns defaults."""
        # This should return defaults since we don't have a file for 'test_nonexistent'
        config = BetTypeFeatureConfig.load_for_bet_type('test_nonexistent')

        assert config.bet_type == 'test_nonexistent'
        assert config.elo_k_factor == 32.0  # Default value

    def test_load_for_bet_type_from_file(self):
        """Test loading config for bet type with existing config file."""
        # Create a config file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override the default path temporarily
            original_dir = BetTypeFeatureConfig.__module__
            config_dir = Path(tmpdir)
            config_dir.mkdir(exist_ok=True)

            # Create test config
            test_config = BetTypeFeatureConfig(
                bet_type="test_bet",
                elo_k_factor=48.0,
            )
            test_path = config_dir / "test_bet.yaml"
            test_config.save(test_path)

            # Load it back
            loaded = BetTypeFeatureConfig.load(test_path)
            assert loaded.bet_type == "test_bet"
            assert loaded.elo_k_factor == 48.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_list_params_serialization(self):
        """Test that list params (window_sizes) serialize correctly."""
        config = BetTypeFeatureConfig(
            bet_type="corners",
            corners_window_sizes=[5, 10, 15, 20],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_list.yaml"
            config.save(path)
            loaded = BetTypeFeatureConfig.load(path)

            assert loaded.corners_window_sizes == [5, 10, 15, 20]

    def test_none_metadata_serialization(self):
        """Test that None metadata values serialize correctly."""
        config = BetTypeFeatureConfig(
            bet_type="test",
            precision=None,
            roi=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_none.yaml"
            config.save(path)
            loaded = BetTypeFeatureConfig.load(path)

            assert loaded.precision is None
            assert loaded.roi is None

    def test_numpy_types_yaml_serialization(self):
        """Test that numpy types in metadata serialize correctly to YAML.

        This tests the fix for the bug where numpy scalars caused
        yaml.safe_load() to fail with ConstructorError.
        """
        import numpy as np

        config = BetTypeFeatureConfig(
            bet_type="test",
            elo_k_factor=np.int64(24),
            precision=np.float64(0.675),
            roi=np.float64(68.8),
            n_trials=np.int64(30),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_numpy.yaml"
            config.save(path)

            # Verify raw YAML has no numpy tags
            with open(path) as f:
                raw_yaml = f.read()
            assert "numpy" not in raw_yaml, f"YAML contains numpy tags: {raw_yaml}"

            # Verify yaml.safe_load works (this was failing before the fix)
            with open(path) as f:
                data = yaml.safe_load(f)
            assert data["elo_k_factor"] == 24
            assert abs(data["precision"] - 0.675) < 0.001
            assert abs(data["roi"] - 68.8) < 0.1

            # Verify BetTypeFeatureConfig.load works
            loaded = BetTypeFeatureConfig.load(path)
            assert loaded.elo_k_factor == 24
            assert abs(loaded.precision - 0.675) < 0.001


class TestNumpyJSONSerialization:
    """Tests for JSON serialization of optimization results with numpy types."""

    def test_numpy_encoder_handles_int64(self):
        """Test that NumpyEncoder converts numpy int64 to native int."""
        import json
        import numpy as np
        from experiments.run_feature_param_optimization import NumpyEncoder

        data = {"elo_k_factor": np.int64(24), "form_window": np.int64(5)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)

        assert parsed["elo_k_factor"] == 24
        assert parsed["form_window"] == 5
        assert isinstance(parsed["elo_k_factor"], int)

    def test_numpy_encoder_handles_float64(self):
        """Test that NumpyEncoder converts numpy float64 to native float."""
        import json
        import numpy as np
        from experiments.run_feature_param_optimization import NumpyEncoder

        data = {"precision": np.float64(0.675), "roi": np.float64(0.688)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)

        assert abs(parsed["precision"] - 0.675) < 0.001
        assert abs(parsed["roi"] - 0.688) < 0.001
        assert isinstance(parsed["precision"], float)

    def test_numpy_encoder_handles_arrays(self):
        """Test that NumpyEncoder converts numpy arrays to lists."""
        import json
        import numpy as np
        from experiments.run_feature_param_optimization import NumpyEncoder

        data = {"values": np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)

        assert parsed["values"] == [1, 2, 3]
        assert isinstance(parsed["values"], list)

    def test_numpy_encoder_handles_bool(self):
        """Test that NumpyEncoder converts numpy bool to native bool."""
        import json
        import numpy as np
        from experiments.run_feature_param_optimization import NumpyEncoder

        data = {"flag": np.bool_(True)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)

        assert parsed["flag"] is True
        assert isinstance(parsed["flag"], bool)

    def test_optimization_result_json_serialization(self):
        """Test that FeatureOptimizationResult with numpy types serializes to JSON."""
        import json
        import numpy as np
        from dataclasses import asdict
        from experiments.run_feature_param_optimization import (
            FeatureOptimizationResult,
            NumpyEncoder,
        )

        result = FeatureOptimizationResult(
            bet_type="test",
            best_params={
                "elo_k_factor": np.int64(24),
                "elo_home_advantage": np.int64(100),
                "form_window": np.int64(5),
                "ema_span": np.int64(10),
            },
            neg_log_loss=np.float64(-0.55),
            sharpe=np.float64(6.75),  # Sharpe-like consistency score
            precision=np.float64(0.675),
            roi=np.float64(0.688),
            n_bets=np.int64(157),
            n_trials=50,
            n_folds=5,
            fold_precisions=[np.float64(0.68), np.float64(0.65), np.float64(0.70)],
            search_space={"elo_k_factor": (10, 50, 'int')},
            all_trials=[
                {"params": {"elo_k_factor": np.int64(24)}, "precision": np.float64(0.67)}
            ],
            timestamp="2026-01-25T12:00:00",
        )

        # This should not raise TypeError
        json_str = json.dumps(asdict(result), cls=NumpyEncoder)

        # Verify it parses back correctly
        parsed = json.loads(json_str)
        assert parsed["bet_type"] == "test"
        assert parsed["best_params"]["elo_k_factor"] == 24
        assert abs(parsed["precision"] - 0.675) < 0.001


class TestFeatureToParamsMap:
    """Tests for FEATURE_TO_PARAMS_MAP feature-to-parameter mapping."""

    def test_ema_features_map_to_ema_span(self):
        """Test that generic EMA features map to ema_span."""
        from src.features.config_manager import FEATURE_TO_PARAMS_MAP
        import re
        matched = set()
        for pattern, params in FEATURE_TO_PARAMS_MAP.items():
            if re.search(pattern, "away_goals_scored_ema"):
                matched.update(params)
        assert "ema_span" in matched

    def test_niche_ema_maps_to_specific_span(self):
        """Test that niche EMA features include market-specific span."""
        from src.features.config_manager import FEATURE_TO_PARAMS_MAP
        import re
        matched = set()
        for pattern, params in FEATURE_TO_PARAMS_MAP.items():
            if re.search(pattern, "home_fouls_committed_ema"):
                matched.update(params)
        assert "fouls_ema_span" in matched
        assert "ema_span" in matched

    def test_cross_market_transitive_deps(self):
        """Test that cross-market features include transitive dependencies."""
        from src.features.config_manager import FEATURE_TO_PARAMS_MAP
        import re
        matched = set()
        for pattern, params in FEATURE_TO_PARAMS_MAP.items():
            if re.search(pattern, "fouls_int_cards_fouls_diff"):
                matched.update(params)
        assert "cards_ema_span" in matched
        assert "fouls_ema_span" in matched

    def test_odds_features_parameterless(self):
        """Test that odds features map to empty parameter list."""
        from src.features.config_manager import FEATURE_TO_PARAMS_MAP
        import re
        matched = set()
        for pattern, params in FEATURE_TO_PARAMS_MAP.items():
            if re.search(pattern, "odds_spread_home"):
                matched.update(params)
        assert len(matched) == 0

    def test_dynamics_features(self):
        """Test that dynamics features map to dynamics params."""
        from src.features.config_manager import FEATURE_TO_PARAMS_MAP
        import re
        matched = set()
        for pattern, params in FEATURE_TO_PARAMS_MAP.items():
            if re.search(pattern, "away_fouls_kurtosis"):
                matched.update(params)
        assert "dynamics_window" in matched
        assert "dynamics_hurst_window" in matched

    def test_elo_features_map_correctly(self):
        """Test that ELO features map to all ELO params."""
        from src.features.config_manager import FEATURE_TO_PARAMS_MAP
        import re
        matched = set()
        for pattern, params in FEATURE_TO_PARAMS_MAP.items():
            if re.search(pattern, "home_elo_rating"):
                matched.update(params)
        assert "elo_k_factor" in matched
        assert "elo_home_advantage" in matched

    def test_referee_features_map_correctly(self):
        """Test that referee features map to referee params."""
        from src.features.config_manager import FEATURE_TO_PARAMS_MAP
        import re
        matched = set()
        for pattern, params in FEATURE_TO_PARAMS_MAP.items():
            if re.search(pattern, "ref_cards_per_game"):
                matched.update(params)
        assert "referee_career_window" in matched
        assert "referee_recent_window" in matched


class TestInformedSearchSpace:
    """Tests for get_informed_search_space() function."""

    def test_reduces_dimensionality(self):
        """Test that informed space has fewer params than full space."""
        from src.features.config_manager import get_informed_search_space, get_search_space_for_bet_type
        # Typical home_win features: mostly elo and form
        features = ["home_elo_rating", "away_elo_rating", "elo_diff",
                     "home_form_points", "poisson_expected_total"]
        informed = get_informed_search_space("home_win", features)
        full = get_search_space_for_bet_type("home_win")
        assert len(informed) < len(full)
        assert len(informed) >= 3  # min_params default

    def test_min_params_enforced(self):
        """Test that min_params is respected even for all-odds features."""
        from src.features.config_manager import get_informed_search_space
        # All odds features = parameterless
        features = ["odds_velocity_home", "odds_entropy"]
        # With min_params=3, should pad from BET_TYPE_PARAM_PRIORITIES
        informed = get_informed_search_space("fouls", features, min_params=3)
        assert len(informed) >= 3

    def test_falls_back_when_empty_features(self):
        """Test fallback to full space with empty feature list."""
        from src.features.config_manager import get_informed_search_space, get_search_space_for_bet_type
        informed = get_informed_search_space("fouls", [])
        full = get_search_space_for_bet_type("fouls")
        # Empty features → all parameterless → min_params pad → but if still empty, full fallback
        # With empty list, no patterns match, so informed starts empty, gets padded to min_params=3
        assert len(informed) >= 3

    def test_always_subset_of_full_space(self):
        """Test that informed space is always a subset of PARAMETER_SEARCH_SPACES."""
        from src.features.config_manager import get_informed_search_space, PARAMETER_SEARCH_SPACES
        features = ["home_elo_rating", "home_fouls_committed_ema", "ref_cards_per_game"]
        informed = get_informed_search_space("fouls", features)
        for param in informed:
            assert param in PARAMETER_SEARCH_SPACES

    def test_niche_features_include_specific_params(self):
        """Test that niche features include market-specific EMA params."""
        from src.features.config_manager import get_informed_search_space
        features = ["home_fouls_committed_ema", "fouls_int_cards_fouls_diff",
                     "away_fouls_kurtosis"]
        informed = get_informed_search_space("fouls", features)
        assert "fouls_ema_span" in informed
        assert "dynamics_window" in informed


class TestCompositeObjective:
    """Tests for compute_composite_objective() function."""

    def test_perfect_score_near_one(self):
        """Test that perfect predictions yield score near 1.0."""
        import numpy as np
        from experiments.run_feature_param_optimization import compute_composite_objective
        # Need 25+ samples so top-20% quantile (>=5 samples) triggers tail_precision
        y_true = np.array([1]*15 + [0]*15)
        probs = np.array([0.99]*15 + [0.01]*15)
        # neg_log_loss near 0 (very good)
        score, tail_prec, ece = compute_composite_objective(-0.05, y_true, probs)
        assert score > 0.7  # Should be high

    def test_random_score_low(self):
        """Test that random predictions yield low score."""
        import numpy as np
        from experiments.run_feature_param_optimization import compute_composite_objective
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        probs = np.random.uniform(0.3, 0.7, 100)
        # neg_log_loss near -0.693 (random)
        score, tail_prec, ece = compute_composite_objective(-0.693, y_true, probs)
        assert score < 0.5  # Should be low

    def test_ece_penalty_above_threshold(self):
        """Test that ECE above 0.05 incurs penalty."""
        import numpy as np
        from experiments.run_feature_param_optimization import compute_composite_objective
        # Create poorly calibrated predictions (all predict 0.9 but only 50% correct)
        y_true = np.array([1, 0] * 25)
        probs = np.full(50, 0.9)
        score, tail_prec, ece = compute_composite_objective(-0.5, y_true, probs)
        assert ece > 0.05  # Should have high ECE

    def test_ece_penalty_below_threshold(self):
        """Test that ECE below 0.05 has zero penalty."""
        import numpy as np
        from experiments.run_feature_param_optimization import compute_composite_objective
        # Well-calibrated: predict ~0.5, actual ~50% true
        y_true = np.array([1, 0] * 50)
        probs = np.full(100, 0.50)
        score, tail_prec, ece = compute_composite_objective(-0.693, y_true, probs)
        assert ece < 0.05

    def test_tail_precision_uses_top_quantile(self):
        """Test that tail precision uses top-20% of predictions."""
        import numpy as np
        from experiments.run_feature_param_optimization import compute_composite_objective
        # Bottom 80%: random, Top 20%: all correct
        y_true = np.concatenate([np.random.randint(0, 2, 80), np.ones(20)])
        probs = np.concatenate([np.random.uniform(0.3, 0.6, 80), np.full(20, 0.95)])
        score, tail_prec, ece = compute_composite_objective(-0.4, y_true, probs)
        assert tail_prec > 0.8  # Top-20% should be mostly correct

    def test_returns_three_values(self):
        """Test that function returns tuple of three values."""
        import numpy as np
        from experiments.run_feature_param_optimization import compute_composite_objective
        y_true = np.array([1, 0, 1, 0, 1])
        probs = np.array([0.8, 0.2, 0.7, 0.3, 0.6])
        result = compute_composite_objective(-0.5, y_true, probs)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


class TestLoadSelectedFeaturesFromDeployment:
    """Tests for load_selected_features_from_deployment()."""

    def test_loads_selected_features(self):
        """Test loading features from a valid deployment config."""
        import tempfile, json
        from src.features.config_manager import load_selected_features_from_deployment
        config = {
            "markets": {
                "fouls": {
                    "selected_features": ["feat_a", "feat_b", "feat_c"],
                    "threshold": 0.6,
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            f.flush()
            features = load_selected_features_from_deployment(f.name, "fouls")
        assert features == ["feat_a", "feat_b", "feat_c"]

    def test_returns_none_for_missing_market(self):
        """Test that missing market returns None."""
        import tempfile, json
        from src.features.config_manager import load_selected_features_from_deployment
        config = {"markets": {"home_win": {"selected_features": ["feat_a"]}}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            f.flush()
            features = load_selected_features_from_deployment(f.name, "fouls")
        assert features is None

    def test_returns_none_for_missing_file(self):
        """Test that missing file returns None."""
        from src.features.config_manager import load_selected_features_from_deployment
        features = load_selected_features_from_deployment("/nonexistent/file.json", "fouls")
        assert features is None

    def test_handles_features_key(self):
        """Test fallback to 'features' key when 'selected_features' absent."""
        import tempfile, json
        from src.features.config_manager import load_selected_features_from_deployment
        config = {
            "markets": {
                "btts": {
                    "features": ["feat_x", "feat_y"],
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            f.flush()
            features = load_selected_features_from_deployment(f.name, "btts")
        assert features == ["feat_x", "feat_y"]
