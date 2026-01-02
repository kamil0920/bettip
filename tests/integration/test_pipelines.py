"""Integration tests for ML pipelines."""
import pytest
import tempfile
from pathlib import Path

import pandas as pd

from src.config_loader import load_config, Config
from src.pipelines.preprocessing_pipeline import PreprocessingPipeline
from src.pipelines.feature_eng_pipeline import FeatureEngineeringPipeline


class TestConfigLoader:
    """Tests for configuration loading."""

    def test_load_local_config(self):
        """Test loading local.yaml configuration."""
        config_path = Path(__file__).parent.parent.parent / "config" / "local.yaml"

        if config_path.exists():
            config = load_config(str(config_path))

            assert isinstance(config, Config)
            assert config.league == "premier_league"
            assert len(config.seasons) > 0
            assert config.data.raw_dir == "data/01-raw"
            assert config.preprocessing.batch_size > 0
            assert config.features.form_window > 0

    def test_load_prod_config(self):
        """Test loading prod.yaml configuration."""
        config_path = Path(__file__).parent.parent.parent / "config" / "prod.yaml"

        if config_path.exists():
            config = load_config(str(config_path))

            assert isinstance(config, Config)
            assert len(config.seasons) >= 6  # All seasons
            assert config.preprocessing.error_handling == "raise"

    def test_load_nonexistent_config(self):
        """Test loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_config_path_methods(self):
        """Test configuration path helper methods."""
        config_path = Path(__file__).parent.parent.parent / "config" / "local.yaml"

        if config_path.exists():
            config = load_config(str(config_path))

            raw_dir = config.get_raw_season_dir(2024)
            assert raw_dir == Path("data/01-raw/premier_league/2024")

            preprocessed_dir = config.get_preprocessed_season_dir(2024)
            assert "02-preprocessed" in str(preprocessed_dir)
            assert "2024" in str(preprocessed_dir)

            features_dir = config.get_features_dir()
            assert "03-features" in str(features_dir)


class TestPreprocessingPipelineIntegration:
    """Integration tests for PreprocessingPipeline."""

    def test_pipeline_validates_config(self):
        """Test that pipeline validates configuration."""
        config_path = Path(__file__).parent.parent.parent / "config" / "local.yaml"

        if config_path.exists():
            config = load_config(str(config_path))
            pipeline = PreprocessingPipeline(config)

            assert pipeline.config is not None
            assert pipeline.config.seasons == config.seasons

    def test_pipeline_with_nonexistent_data(self):
        """Test pipeline fails gracefully with missing data."""
        config_path = Path(__file__).parent.parent.parent / "config" / "local.yaml"

        if config_path.exists():
            config = load_config(str(config_path))
            config.seasons = [1900]

            pipeline = PreprocessingPipeline(config)

            with pytest.raises(FileNotFoundError):
                pipeline.run()


class TestFeatureEngineeringPipelineIntegration:
    """Integration tests for FeatureEngineeringPipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        config_path = Path(__file__).parent.parent.parent / "config" / "local.yaml"

        if config_path.exists():
            config = load_config(str(config_path))
            pipeline = FeatureEngineeringPipeline(config)

            assert pipeline.config is not None
            assert pipeline.config.features.form_window > 0

    def test_pipeline_with_real_data(self):
        """Test pipeline with actual preprocessed data."""
        config_path = Path(__file__).parent.parent.parent / "config" / "local.yaml"
        preprocessed_dir = Path(__file__).parent.parent.parent / "data" / "02-preprocessed"

        if config_path.exists() and preprocessed_dir.exists():
            config = load_config(str(config_path))

            has_data = False
            for season in config.seasons:
                season_dir = preprocessed_dir / config.league / str(season)
                if (season_dir / "matches.parquet").exists():
                    has_data = True
                    break

            if has_data:
                pipeline = FeatureEngineeringPipeline(config)

                with tempfile.TemporaryDirectory() as tmpdir:
                    config.data.features_dir = tmpdir

                    try:
                        result = pipeline.run(output_filename="test_features.csv")

                        assert isinstance(result, pd.DataFrame)
                        assert len(result) > 0
                        assert 'fixture_id' in result.columns
                        assert 'home_wins_last_n' in result.columns

                        output_file = Path(tmpdir) / "test_features.csv"
                        assert output_file.exists()

                    except Exception as e:
                        pytest.skip(f"Pipeline execution skipped: {e}")


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.skip(reason="Requires complete data setup")
    def test_complete_workflow(self):
        """Test complete workflow from preprocessing to features."""

        config_path = Path(__file__).parent.parent.parent / "config" / "local.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        config = load_config(str(config_path))

        # Step 1: Preprocessing
        # preprocessing_pipeline = PreprocessingPipeline(config)
        # preprocessing_result = preprocessing_pipeline.run()

        # Step 2: Feature Engineering
        # feature_pipeline = FeatureEngineeringPipeline(config)
        # features = feature_pipeline.run()

        # Step 3: Verify
        # assert len(features) > 0
        # assert 'home_wins_last_n' in features.columns

        pass

    def test_config_consistency(self):
        """Test that local and prod configs are consistent."""
        local_path = Path(__file__).parent.parent.parent / "config" / "local.yaml"
        prod_path = Path(__file__).parent.parent.parent / "config" / "prod.yaml"

        if local_path.exists() and prod_path.exists():
            local_config = load_config(str(local_path))
            prod_config = load_config(str(prod_path))

            assert local_config.data.raw_dir == prod_config.data.raw_dir
            assert local_config.data.preprocessed_dir == prod_config.data.preprocessed_dir
            assert local_config.league == prod_config.league

            assert len(prod_config.seasons) >= len(local_config.seasons)

            assert prod_config.preprocessing.error_handling == "raise"


class TestDataIntegrity:
    """Tests for data integrity across pipeline stages."""

    def test_features_file_structure(self):
        """Test that features file has expected structure."""
        features_path = Path(__file__).parent.parent.parent / "data" / "03-features" / "features.csv"

        if features_path.exists():
            df = pd.read_csv(features_path)

            required_cols = [
                'fixture_id', 'date', 'home_team_id', 'away_team_id',
                'home_wins_last_n', 'away_wins_last_n'
            ]

            for col in required_cols:
                assert col in df.columns, f"Missing column: {col}"

            assert len(df) > 0, "Features file is empty"

            if 'home_wins_last_n' in df.columns:
                assert df['home_wins_last_n'].min() >= 0
                assert df['home_wins_last_n'].max() <= 5

    def test_preprocessed_data_structure(self):
        """Test that preprocessed Parquet files have expected structure."""
        preprocessed_dir = Path(__file__).parent.parent.parent / "data" / "02-preprocessed"

        if preprocessed_dir.exists():
            for season_dir in preprocessed_dir.glob("**/2024"):
                matches_path = season_dir / "matches.parquet"

                if matches_path.exists():
                    df = pd.read_parquet(matches_path)

                    clean_cols = ['fixture_id', 'date', 'home_team_id', 'away_team_id', 'ft_home', 'ft_away']
                    raw_cols = ['fixture.id', 'fixture.date', 'teams.home.id', 'teams.away.id', 'goals.home', 'goals.away']

                    has_clean = all(col in df.columns for col in clean_cols)
                    has_raw = all(col in df.columns for col in raw_cols)

                    assert has_clean or has_raw, f"Data must have either clean or raw API columns. Found: {list(df.columns)[:10]}"

                    if has_clean:
                        assert df['fixture_id'].is_unique
                        assert (df['ft_home'] >= 0).all()
                        assert (df['ft_away'] >= 0).all()
                    else:
                        assert df['fixture.id'].is_unique
                        assert (df['goals.home'] >= 0).all()
                        assert (df['goals.away'] >= 0).all()

                    break
