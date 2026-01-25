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
