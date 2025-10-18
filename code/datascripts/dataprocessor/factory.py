import logging

from config import ProcessingConfig
from processors import SeasonDataProcessor
from loaders import (
    JSONDataLoader,
    EventsLoader,
    LineupsLoader,
    PlayerStatsLoader,
    FixturesLoader
)
from validators import (
    FixtureValidator,
    PlayerStatsValidator,
    EventValidator,
    LineupValidator
)
from extractors import (
    FixtureExtractor,
    EventExtractor,
    PlayerStatsExtractor,
    LineupExtractor
)
from writers import ParquetDataWriter

logger = logging.getLogger(__name__)


class DataProcessorFactory:
    """Factory for creating data processors."""

    @staticmethod
    def create_season_processor(config: ProcessingConfig) -> SeasonDataProcessor:
        """
        Create season processor with all dependencies.
        """
        logger.info("🔧 Creating season processor...")

        # 1. Loaders
        json_loader = JSONDataLoader()
        fixtures_loader = FixturesLoader(json_loader)
        events_loader = EventsLoader(json_loader)
        lineups_loader = LineupsLoader(json_loader)
        player_stats_loader = PlayerStatsLoader(json_loader)

        logger.debug("✅ Loaders created")

        # 2. Validators
        fixture_validator = FixtureValidator()
        player_validator = PlayerStatsValidator()
        event_validator = EventValidator()
        lineup_validator = LineupValidator()

        logger.debug("✅ Validators created")

        # 3. Extractors
        fixture_extractor = FixtureExtractor(fixture_validator)
        event_extractor = EventExtractor(event_validator)
        player_extractor = PlayerStatsExtractor(player_validator)
        lineup_extractor = LineupExtractor(lineup_validator)

        logger.debug("✅ Extractors created")

        # 4. Writer
        writer = ParquetDataWriter()

        logger.debug("✅ Writer created")

        # 5. Processor
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

        logger.info("✅ Season processor created successfully")

        return processor

    @staticmethod
    def validate_config(config: ProcessingConfig) -> bool:
        """
        Validate configuration before creating processor.
        """
        logger.info("🔍 Validating configuration...")

        if not config.base_dir.exists():
            logger.error(f"❌ Base directory does not exist: {config.base_dir}")
            return False

        for season in config.seasons:
            season_dir = config.get_season_dir(season)

            if not season_dir.exists():
                logger.error(f"❌ Season directory does not exist: {season_dir}")
                return False

            fixtures_file = season_dir / "fixtures.json"
            if not fixtures_file.exists():
                logger.warning(f"⚠️  Fixtures file not found: {fixtures_file}")

        logger.info("✅ Configuration valid")
        return True
