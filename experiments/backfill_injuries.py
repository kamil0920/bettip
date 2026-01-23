#!/usr/bin/env python3
"""
Backfill historical injury data for training.

This script:
1. Collects injury data from API-Football for each league/season
2. Saves to parquet files for feature engineering
3. Merges injury features into existing feature dataset

Usage:
    # Backfill all leagues, 2022-2024 seasons
    uv run python experiments/backfill_injuries.py --start-season 2022 --end-season 2024

    # Single league
    uv run python experiments/backfill_injuries.py --league premier_league --season 2024

    # Merge into features file
    uv run python experiments/backfill_injuries.py --merge-only
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.prematch_collector import PreMatchCollector, LEAGUE_IDS
from src.features.engineers.injuries import HistoricalInjuryFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Output directories
INJURY_DATA_DIR = Path('data/07-injuries')
FEATURES_DIR = Path('data/03-features')


def collect_injuries_for_league(
    league: str,
    season: int,
    output_dir: Path
) -> Optional[pd.DataFrame]:
    """
    Collect injury data for a single league/season.

    Args:
        league: League key
        season: Season year
        output_dir: Output directory

    Returns:
        DataFrame with injuries or None if failed
    """
    league_id = LEAGUE_IDS.get(league)
    if not league_id:
        logger.error(f"Unknown league: {league}")
        return None

    collector = PreMatchCollector()

    logger.info(f"Collecting injuries for {league} ({league_id}) season {season}")

    try:
        injuries = collector.get_injuries_by_league(league_id, season)

        if injuries.empty:
            logger.warning(f"No injuries found for {league} {season}")
            return None

        logger.info(f"Found {len(injuries)} injury records")

        # Save to parquet
        output_file = output_dir / f"injuries_{league}_{season}.parquet"
        injuries.to_parquet(output_file, index=False)
        logger.info(f"Saved to {output_file}")

        return injuries

    except Exception as e:
        logger.error(f"Failed to collect {league} {season}: {e}")
        return None


def collect_all_injuries(
    leagues: List[str],
    seasons: List[int],
    output_dir: Path
) -> pd.DataFrame:
    """
    Collect injuries for multiple leagues and seasons.

    Args:
        leagues: List of league keys
        seasons: List of season years
        output_dir: Output directory

    Returns:
        Combined DataFrame with all injuries
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_injuries = []

    for league in leagues:
        for season in seasons:
            injuries = collect_injuries_for_league(league, season, output_dir)
            if injuries is not None and not injuries.empty:
                injuries['league'] = league
                injuries['season'] = season
                all_injuries.append(injuries)

    if not all_injuries:
        logger.warning("No injuries collected")
        return pd.DataFrame()

    combined = pd.concat(all_injuries, ignore_index=True)

    # Save combined file
    combined_file = output_dir / "injuries_all.parquet"
    combined.to_parquet(combined_file, index=False)
    logger.info(f"Saved combined injuries ({len(combined)} records) to {combined_file}")

    return combined


def create_injury_features(
    injuries: pd.DataFrame,
    features_file: Path
) -> pd.DataFrame:
    """
    Create injury features and merge with existing features.

    Args:
        injuries: Injury DataFrame
        features_file: Existing features file

    Returns:
        Features DataFrame with injury columns added
    """
    logger.info(f"Loading features from {features_file}")

    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        return pd.DataFrame()

    features = pd.read_csv(features_file, low_memory=False)
    logger.info(f"Loaded {len(features)} feature rows")

    # Check if already has injury features
    existing_inj_cols = [c for c in features.columns if c.startswith('inj_')]
    if existing_inj_cols:
        logger.info(f"Found existing injury columns: {existing_inj_cols}")

    # Create injury feature engineer
    engineer = HistoricalInjuryFeatureEngineer()

    # Prepare data for engineer
    matches = features[['fixture_id', 'home_team_id', 'away_team_id']].copy()

    injury_features = engineer.create_features({
        'matches': matches,
        'injuries': injuries
    })

    logger.info(f"Created injury features for {len(injury_features)} matches")

    # Merge with existing features
    # Drop old injury columns if present
    features = features.drop(columns=existing_inj_cols, errors='ignore')

    merged = features.merge(injury_features, on='fixture_id', how='left')

    # Fill NaN injury features with 0 (no injury data = assume no injuries)
    inj_cols = [c for c in merged.columns if c.startswith('inj_')]
    for col in inj_cols:
        merged[col] = merged[col].fillna(0)

    logger.info(f"Merged features shape: {merged.shape}")

    return merged


def analyze_injury_impact(
    features: pd.DataFrame,
    target: str = 'home_win'
) -> Dict:
    """
    Analyze correlation between injury features and outcomes.

    Args:
        features: Features DataFrame with injury columns
        target: Target variable

    Returns:
        Dict with analysis results
    """
    inj_cols = [c for c in features.columns if c.startswith('inj_')]

    if not inj_cols:
        logger.warning("No injury columns found")
        return {}

    if target not in features.columns:
        logger.warning(f"Target {target} not in features")
        return {}

    results = {}

    for col in inj_cols:
        corr = features[col].corr(features[target])
        results[col] = {
            'correlation': corr,
            'non_zero_count': (features[col] != 0).sum(),
            'mean': features[col].mean(),
            'std': features[col].std(),
        }

    # Sort by absolute correlation
    results = dict(sorted(
        results.items(),
        key=lambda x: abs(x[1]['correlation']),
        reverse=True
    ))

    return results


def main():
    parser = argparse.ArgumentParser(description='Backfill historical injury data')
    parser.add_argument('--league', type=str, help='Single league to collect')
    parser.add_argument('--season', type=int, help='Single season to collect')
    parser.add_argument('--start-season', type=int, default=2022)
    parser.add_argument('--end-season', type=int, default=2024)
    parser.add_argument('--leagues', type=str, nargs='+',
                       default=['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1'])
    parser.add_argument('--output-dir', type=str, default=str(INJURY_DATA_DIR))
    parser.add_argument('--features-file', type=str,
                       default=str(FEATURES_DIR / 'features_with_sportmonks_odds.csv'))
    parser.add_argument('--merge-only', action='store_true',
                       help='Only merge existing injury data into features')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze injury impact after merging')
    parser.add_argument('--save-merged', type=str,
                       help='Save merged features to this file')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.merge_only:
        # Load existing injury data
        injury_file = output_dir / "injuries_all.parquet"
        if not injury_file.exists():
            logger.error(f"Injury data not found: {injury_file}")
            logger.error("Run without --merge-only first to collect data")
            return 1

        injuries = pd.read_parquet(injury_file)
        logger.info(f"Loaded {len(injuries)} injury records")

    else:
        # Determine what to collect
        if args.league and args.season:
            leagues = [args.league]
            seasons = [args.season]
        else:
            leagues = args.leagues
            seasons = list(range(args.start_season, args.end_season + 1))

        logger.info(f"Collecting injuries for: {leagues}")
        logger.info(f"Seasons: {seasons}")

        injuries = collect_all_injuries(leagues, seasons, output_dir)

        if injuries.empty:
            logger.error("No injury data collected")
            return 1

    # Create and merge injury features
    features = create_injury_features(injuries, Path(args.features_file))

    if features.empty:
        return 1

    # Analyze impact
    if args.analyze:
        print("\n" + "=" * 60)
        print("INJURY FEATURE ANALYSIS")
        print("=" * 60)

        for target in ['home_win', 'away_win', 'draw', 'btts']:
            if target in features.columns:
                print(f"\n{target.upper()}:")
                analysis = analyze_injury_impact(features, target)
                for col, stats in list(analysis.items())[:5]:
                    print(f"  {col}: r={stats['correlation']:.3f} "
                          f"(non-zero: {stats['non_zero_count']})")

    # Save merged features
    if args.save_merged:
        output_file = Path(args.save_merged)
        features.to_csv(output_file, index=False)
        logger.info(f"Saved merged features to {output_file}")

        # Also save summary
        inj_cols = [c for c in features.columns if c.startswith('inj_')]
        logger.info(f"Injury columns added: {inj_cols}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
