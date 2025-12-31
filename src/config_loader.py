"""
Configuration loader for ML pipeline.

Loads YAML configuration files and provides typed access to settings.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any, Dict, Union
import inspect

import yaml


def detect_seasons(data_dir: str, league: str) -> List[int]:
    """
    Auto-detect available seasons from data directory.

    Args:
        data_dir: Path to data directory (raw or preprocessed)
        league: League name (e.g., 'premier_league')

    Returns:
        Sorted list of season years found in the directory
    """
    league_path = Path(data_dir) / league

    if not league_path.exists():
        return []

    seasons = []
    for item in league_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            # Check if the season folder has data (matches.parquet)
            if (item / "matches.parquet").exists():
                seasons.append(int(item.name))

    return sorted(seasons)


@dataclass
class DataConfig:
    """Data directories configuration."""
    raw_dir: str = "data/01-raw"
    preprocessed_dir: str = "data/02-preprocessed"
    features_dir: str = "data/03-features"
    predictions_dir: str = "data/04-predictions"
    models_dir: str = "data/05-models"


@dataclass
class PreprocessingConfig:
    """Preprocessing pipeline configuration."""
    batch_size: int = 100
    error_handling: str = "log"
    include_player_features: bool = True
    include_events: bool = True
    output_format: str = "parquet"


@dataclass
class FeaturesConfig:
    """Feature engineering configuration."""
    form_window: int = 5
    ema_span: int = 10
    include_h2h: bool = True
    include_team_stats: bool = True

    lineup_lookback: int = 3
    star_top_n: int = 3
    star_min_matches: int = 5
    rating_lookback: int = 5
    key_player_top_n: int = 5
    discipline_lookback: int = 5
    goal_timing_lookback: int = 10


@dataclass
class ModelConfig:
    """Model training configuration."""
    type: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceConfig:
    """Inference pipeline configuration."""
    batch_size: int = 32
    output_format: str = "csv"
    min_confidence: float = 0.0
    stake: float = 1.0

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    league: str = "premier_league"
    seasons: Union[List[int], str] = field(default_factory=lambda: [2024, 2025])
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def resolve_seasons(self, data_dir: str = None) -> List[int]:
        """
        Resolve seasons, auto-detecting if configured as 'auto'.

        Args:
            data_dir: Directory to scan for seasons. If None, uses raw_dir.

        Returns:
            List of season years to process
        """
        if self.seasons == "auto":
            if data_dir is None:
                data_dir = self.data.raw_dir
            detected = detect_seasons(data_dir, self.league)
            if not detected:
                raise ValueError(
                    f"No seasons found in {data_dir}/{self.league}. "
                    "Ensure data exists or specify seasons explicitly."
                )
            return detected
        return self.seasons

    def get_raw_season_dir(self, season: int) -> Path:
        """Get path to raw data for a specific season."""
        return Path(self.data.raw_dir) / self.league / str(season)

    def get_preprocessed_season_dir(self, season: int) -> Path:
        """Get path to preprocessed data for a specific season."""
        path = Path(self.data.preprocessed_dir) / self.league / str(season)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_features_dir(self) -> Path:
        """Get path to features directory."""
        path = Path(self.data.features_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_predictions_dir(self) -> Path:
        """Get path to predictions directory."""
        path = Path(self.data.predictions_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with all settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        yaml_data = yaml.safe_load(f)

    return _parse_config(yaml_data)


def _parse_config(data: Dict[str, Any]) -> Config:
    """Parse YAML data into Config object."""
    config = Config()

    if 'data' in data:
        config.data = DataConfig(**data['data'])

    if 'league' in data:
        config.league = data['league']
    if 'seasons' in data:
        config.seasons = data['seasons']

    if 'preprocessing' in data:
        config.preprocessing = PreprocessingConfig(**data['preprocessing'])
    if 'features' in data:
        config.features = FeaturesConfig(**data['features'])

    if 'model' in data:
        model_data = data['model'].copy()
        selected_type = model_data.get('type', 'random_forest')

        base_config = {k: v for k, v in model_data.items() if k != 'params'}

        known_fields = inspect.signature(ModelConfig).parameters.keys()
        init_args = {k: v for k, v in base_config.items() if k in known_fields}

        config.model = ModelConfig(**init_args)

        all_params = model_data.get('params', {})

        if selected_type in all_params:
            config.model.params = all_params[selected_type]
        else:
            extra_params = {k: v for k, v in base_config.items() if k not in known_fields}
            config.model.params = extra_params

    if 'inference' in data:
        config.inference = InferenceConfig(**data['inference'])
    if 'logging' in data:
        config.logging = LoggingConfig(**data['logging'])

    return config