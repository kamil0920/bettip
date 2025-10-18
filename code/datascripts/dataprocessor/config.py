from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ProcessingConfig:
    """Data processing configuration."""

    base_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    seasons: List[int] = None
    league: str = "premier_league"
    batch_size: int = 100
    error_handling: str = "log"
    include_player_features: bool = True
    include_events: bool = True

    def __post_init__(self):
        """Auto-detect paths if not provided."""

        current = Path(__file__).resolve().parent
        project_root = current.parents[1]

        if self.base_dir is None:
            self.base_dir = project_root / "apicalls" / "football_data"
        elif isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)

        if self.output_dir is None:
            self.output_dir = project_root / "apicalls" / "processed_data"
        elif isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_dir}")

        if not self.seasons:
            raise ValueError("seasons parameter is required")

    def get_season_dir(self, season: int) -> Path:
        """Returns the path to the RAW season data."""
        return self.base_dir / self.league / str(season)

    def get_output_dir(self, season: int) -> Path:
        """Returns the path to the output directory."""
        output_season_dir = self.output_dir / self.league / str(season)
        output_season_dir.mkdir(parents=True, exist_ok=True)
        return output_season_dir
