#!/usr/bin/env python3
"""
Fetch odds data from football-data.co.uk and merge with existing features.

Usage:
    # Fetch odds for Premier League 2024/25 and merge with features
    uv run python entrypoints/fetch_odds.py --league premier_league --seasons 2024

    # Fetch multiple seasons
    uv run python entrypoints/fetch_odds.py --league premier_league --seasons 2023 2024

    # All configured leagues
    uv run python entrypoints/fetch_odds.py --all-leagues --seasons 2024
"""
import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.odds.football_data_loader import FootballDataLoader, LEAGUE_CODES
from src.odds.odds_features import OddsFeatureEngineer
from src.odds.odds_merger import OddsMerger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


CONFIGURED_LEAGUES = [
    "premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1",
    "eredivisie", "belgian_pro_league", "portuguese_liga",
    "turkish_super_lig", "scottish_premiership",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch and merge odds data")
    parser.add_argument("--league", type=str, help="League to fetch (e.g., premier_league)")
    parser.add_argument("--all-leagues", action="store_true", help="Fetch all configured leagues")
    parser.add_argument("--seasons", type=int, nargs="+", required=True,
                       help="Season(s) to fetch (e.g., 2024 for 2024/25)")
    parser.add_argument("--features-dir", type=Path, default=Path("data/03-features"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/odds-cache"))
    parser.add_argument("--output-suffix", type=str, default="_with_odds",
                       help="Suffix for output files")
    parser.add_argument("--no-merge", action="store_true",
                       help="Only download odds, don't merge with features")
    return parser.parse_args()


def fetch_odds_for_league(
    league: str,
    seasons: list,
    features_dir: Path,
    cache_dir: Path,
    output_suffix: str,
    merge: bool = True
) -> bool:
    """
    Fetch odds for a single league and optionally merge with features.

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FETCHING ODDS: {league.upper()}")
    logger.info(f"{'='*60}")

    if league not in LEAGUE_CODES:
        logger.error(f"League not supported by football-data.co.uk: {league}")
        logger.info(f"Supported leagues: {list(LEAGUE_CODES.keys())}")
        return False

    loader = FootballDataLoader(cache_dir=cache_dir)

    try:
        odds_df = loader.load_multiple_seasons(league, seasons)
    except Exception as e:
        logger.error(f"Failed to fetch odds for {league}: {e}")
        return False

    if odds_df.empty:
        logger.warning(f"No odds data found for {league}")
        return False

    logger.info(f"Fetched {len(odds_df)} matches with odds")

    engineer = OddsFeatureEngineer(use_closing_odds=True)
    odds_df = engineer.create_features(odds_df)

    odds_output = cache_dir / f"odds_{league}.csv"
    odds_df.to_csv(odds_output, index=False)
    logger.info(f"Saved odds data to: {odds_output}")

    if not merge:
        return True

    features_file = features_dir / f"features_{league}.csv"
    if not features_file.exists():
        logger.warning(f"Features file not found: {features_file}")
        logger.info("Skipping merge, odds data saved separately")
        return True

    logger.info(f"Merging with features: {features_file}")

    import pandas as pd
    features_df = pd.read_csv(features_file)
    logger.info(f"Loaded {len(features_df)} feature rows")

    merger = OddsMerger()
    merged_df = merger.merge_with_features(features_df, odds_df)

    odds_cols = [c for c in merged_df.columns if c.startswith('odds_')]
    if odds_cols:
        n_with_odds = merged_df[odds_cols[0]].notna().sum()
        logger.info(f"Successfully merged odds for {n_with_odds}/{len(merged_df)} matches")

    output_file = features_dir / f"features_{league}{output_suffix}.csv"
    merged_df.to_csv(output_file, index=False)
    logger.info(f"Saved merged features to: {output_file}")

    return True


def main():
    args = parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.all_leagues:
        leagues = CONFIGURED_LEAGUES
    elif args.league:
        leagues = [args.league]
    else:
        logger.error("Must specify --league or --all-leagues")
        return 1

    logger.info(f"Leagues to process: {leagues}")
    logger.info(f"Seasons: {args.seasons}")

    success_count = 0
    for league in leagues:
        success = fetch_odds_for_league(
            league=league,
            seasons=args.seasons,
            features_dir=args.features_dir,
            cache_dir=args.cache_dir,
            output_suffix=args.output_suffix,
            merge=not args.no_merge
        )
        if success:
            success_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETED: {success_count}/{len(leagues)} leagues processed successfully")
    logger.info(f"{'='*60}")

    return 0 if success_count == len(leagues) else 1


if __name__ == "__main__":
    sys.exit(main())
