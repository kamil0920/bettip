"""Factory for creating data processors."""
import logging

from src.config_loader import Config
from src.preprocessing.processors import SeasonDataProcessor
from src.preprocessing.loaders import (
    ParquetDataLoader,
    EventsLoader,
    LineupsLoader,
    PlayerStatsLoader,
    FixturesLoader
)
from src.preprocessing.validators import (
    FixtureValidator,
    PlayerStatsValidator,
    EventValidator,
    LineupValidator
)
from src.preprocessing.extractors import (
    FixtureExtractor,
    EventExtractor,
    PlayerStatsExtractor,
    LineupExtractor
)
from src.preprocessing.writers import ParquetDataWriter

logger = logging.getLogger(__name__)


class DataProcessorFactory:
    """Factory for creating data processors."""

    @staticmethod
    def create_data_processor(config: Config) -> SeasonDataProcessor:
        """
        Create data processor with all dependencies.

        Args:
            config: Configuration object from YAML

        Returns:
            Fully configured SeasonDataProcessor
        """
        logger.info("Creating data processor...")

        paraquet_loader = ParquetDataLoader()
        fixtures_loader = FixturesLoader(paraquet_loader)
        events_loader = EventsLoader(paraquet_loader)
        lineups_loader = LineupsLoader(paraquet_loader)
        player_stats_loader = PlayerStatsLoader(paraquet_loader)

        logger.debug("Loaders created")

        fixture_validator = FixtureValidator()
        player_validator = PlayerStatsValidator()
        event_validator = EventValidator()
        lineup_validator = LineupValidator()

        logger.debug("Validators created")

        fixture_extractor = FixtureExtractor(fixture_validator)
        event_extractor = EventExtractor(event_validator)
        player_extractor = PlayerStatsExtractor(player_validator)
        lineup_extractor = LineupExtractor(lineup_validator)

        logger.debug("Extractors created")

        writer = ParquetDataWriter()

        logger.debug("Writer created")

        processor = SeasonDataProcessor(
            config=config,
            fixtures_loader=fixtures_loader,
            events_loader=events_loader,
            lineups_loader=lineups_loader,
            player_stats_loader=player_stats_loader,
            fixture_extractor=fixture_extractor,
            event_extractor=event_extractor,
            player_extractor=player_extractor,
            lineup_extractor=lineup_extractor,
            writer=writer
        )

        logger.info("Season processor created successfully")

        return processor

    @staticmethod
    def validate_config(config: Config) -> bool:
        """
        Validate configuration before creating processor.

        Args:
            config: Configuration object

        Returns:
            True if configuration is valid, False otherwise
        """
        logger.info("Validating configuration...")

        raw_dir = config.get_raw_season_dir(config.seasons[0]).parent.parent
        if not raw_dir.exists():
            logger.error(f"Raw data directory does not exist: {raw_dir}")
            return False

        for season in config.seasons:
            season_dir = config.get_raw_season_dir(season)

            if not season_dir.exists():
                logger.error(f"Season directory does not exist: {season_dir}")
                return False

            fixtures_file = season_dir / "matches.parquet"
            if not fixtures_file.exists():
                logger.warning(f"Fixtures file not found: {fixtures_file}")

        logger.info("Configuration valid")
        return True
