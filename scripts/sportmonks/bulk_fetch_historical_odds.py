#!/usr/bin/env python
"""
Bulk fetch historical odds from Sportmonks API.

Fetches corners, cards, and shots odds for all historical matches
in our feature dataset. Designed to run within the 9-day trial period.

Usage:
    python scripts/bulk_fetch_historical_odds.py
    python scripts/bulk_fetch_historical_odds.py --start-date 2022-08-01 --end-date 2025-01-21
    python scripts/bulk_fetch_historical_odds.py --resume  # Resume from checkpoint
    python scripts/bulk_fetch_historical_odds.py --dry-run  # Estimate API calls only

API Limits:
    - Free trial: Check your plan limits at https://my.sportmonks.com/
    - Script saves progress every batch, can be resumed if interrupted
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
import requests
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.odds.sportmonks_loader import (
    SportMonksLoader,
    SPORTMONKS_LEAGUES,
    NICHE_MARKET_IDS,
    normalize_sportmonks_team
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Market IDs we want to fetch
CORNERS_MARKETS = [67, 69]  # Over/Under, Alternative
CARDS_MARKETS = [255, 272]  # Over/Under, Asian Total
SHOTS_MARKETS = [291, 292, 284, 285]  # Match shots, team shots
BTTS_MARKETS = [14, 13]  # Both Teams To Score, Result/BTTS

# Output directories
OUTPUT_DIR = Path("data/sportmonks_odds")
CHECKPOINT_FILE = OUTPUT_DIR / "fetch_checkpoint.json"
RAW_ODDS_DIR = OUTPUT_DIR / "raw"
PROCESSED_DIR = OUTPUT_DIR / "processed"


@dataclass
class FetchProgress:
    """Track fetching progress for resumption."""
    last_date: str
    last_league: str
    total_fixtures: int
    total_odds_entries: int
    started_at: str
    updated_at: str
    completed: bool = False


def load_checkpoint() -> Optional[FetchProgress]:
    """Load checkpoint from previous run."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
            return FetchProgress(**data)
    return None


def save_checkpoint(progress: FetchProgress):
    """Save checkpoint for resumption."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(asdict(progress), f, indent=2)


def get_date_ranges(
    start_date: datetime,
    end_date: datetime,
    chunk_days: int = 14
) -> List[Tuple[datetime, datetime]]:
    """Split date range into chunks to avoid API timeouts."""
    ranges = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=chunk_days), end_date)
        ranges.append((current, chunk_end))
        current = chunk_end

    return ranges


def extract_odds_for_fixture(
    fixture: Dict,
    market_ids: List[int]
) -> List[Dict]:
    """Extract odds entries for specific markets from a fixture."""
    entries = []

    fix_id = fixture.get('id')
    fix_name = fixture.get('name', '')
    start_time = fixture.get('starting_at', '')
    league_id = fixture.get('league_id')

    # Parse teams from fixture name
    if ' vs ' in fix_name:
        home_team, away_team = fix_name.split(' vs ', 1)
    else:
        home_team = away_team = ''

    odds_list = fixture.get('odds', [])

    for odd in odds_list:
        market_id = odd.get('market_id')
        if market_id not in market_ids:
            continue

        line = odd.get('total') or odd.get('handicap')
        value = odd.get('value')

        if value is None:
            continue

        try:
            value = float(value)
            if line is not None:
                line = float(line)
        except (ValueError, TypeError):
            continue

        entries.append({
            'fixture_id': fix_id,
            'fixture_name': fix_name,
            'home_team': home_team.strip(),
            'away_team': away_team.strip(),
            'home_team_normalized': normalize_sportmonks_team(home_team.strip()),
            'away_team_normalized': normalize_sportmonks_team(away_team.strip()),
            'start_time': start_time,
            'league_id': league_id,
            'market_id': market_id,
            'market_description': odd.get('market_description', ''),
            'label': odd.get('label', ''),
            'line': line,
            'odds': value,
            'bookmaker_id': odd.get('bookmaker_id'),
            'probability': odd.get('probability'),
        })

    return entries


def aggregate_odds_by_line(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate odds entries into over/under by line."""
    if df.empty:
        return pd.DataFrame()

    # Group by fixture and line
    grouped = []

    for (fix_id, line), group in df.groupby(['fixture_id', 'line']):
        if pd.isna(line):
            continue

        first = group.iloc[0]

        over_odds = group[group['label'].str.lower().str.contains('over', na=False)]['odds']
        under_odds = group[group['label'].str.lower().str.contains('under', na=False)]['odds']

        row = {
            'fixture_id': fix_id,
            'fixture_name': first['fixture_name'],
            'home_team': first['home_team'],
            'away_team': first['away_team'],
            'home_team_normalized': first['home_team_normalized'],
            'away_team_normalized': first['away_team_normalized'],
            'start_time': first['start_time'],
            'league_id': first['league_id'],
            'market_id': first['market_id'],
            'market': first['market_description'],
            'line': line,
            'over_avg': over_odds.mean() if len(over_odds) > 0 else None,
            'over_best': over_odds.max() if len(over_odds) > 0 else None,
            'over_count': len(over_odds),
            'under_avg': under_odds.mean() if len(under_odds) > 0 else None,
            'under_best': under_odds.max() if len(under_odds) > 0 else None,
            'under_count': len(under_odds),
        }
        grouped.append(row)

    return pd.DataFrame(grouped)


def aggregate_btts_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate BTTS odds into yes/no by fixture."""
    if df.empty:
        return pd.DataFrame()

    # Group by fixture
    grouped = []

    for fix_id, group in df.groupby('fixture_id'):
        first = group.iloc[0]

        # BTTS Yes/No - look for yes/no in label
        yes_odds = group[group['label'].str.lower().str.contains('yes', na=False)]['odds']
        no_odds = group[group['label'].str.lower().str.contains('no', na=False)]['odds']

        row = {
            'fixture_id': fix_id,
            'fixture_name': first['fixture_name'],
            'home_team': first['home_team'],
            'away_team': first['away_team'],
            'home_team_normalized': first['home_team_normalized'],
            'away_team_normalized': first['away_team_normalized'],
            'start_time': first['start_time'],
            'league_id': first['league_id'],
            'market_id': first['market_id'],
            'market': first['market_description'],
            'yes_avg': yes_odds.mean() if len(yes_odds) > 0 else None,
            'yes_best': yes_odds.max() if len(yes_odds) > 0 else None,
            'yes_count': len(yes_odds),
            'no_avg': no_odds.mean() if len(no_odds) > 0 else None,
            'no_best': no_odds.max() if len(no_odds) > 0 else None,
            'no_count': len(no_odds),
        }
        grouped.append(row)

    return pd.DataFrame(grouped)


def fetch_historical_odds(
    loader: SportMonksLoader,
    start_date: datetime,
    end_date: datetime,
    resume_from: Optional[FetchProgress] = None,
    dry_run: bool = False,
    delay_between_requests: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch all historical odds for corners, cards, and shots.

    Returns:
        Tuple of (corners_df, cards_df, shots_df)
    """
    # Create output directories
    RAW_ODDS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Get date ranges
    date_ranges = get_date_ranges(start_date, end_date, chunk_days=14)

    logger.info(f"Fetching odds from {start_date.date()} to {end_date.date()}")
    logger.info(f"Split into {len(date_ranges)} date chunks")
    logger.info(f"Leagues: {list(SPORTMONKS_LEAGUES.keys())}")

    if dry_run:
        # Estimate API calls
        estimated_calls = len(date_ranges) * len(SPORTMONKS_LEAGUES)
        logger.info(f"\n=== DRY RUN ===")
        logger.info(f"Estimated API calls: ~{estimated_calls} (date ranges × leagues)")
        logger.info(f"Each call fetches up to 50 fixtures with odds")
        logger.info(f"Actual calls may be higher due to pagination")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Initialize or resume progress
    if resume_from and not resume_from.completed:
        logger.info(f"Resuming from checkpoint: {resume_from.last_date}, {resume_from.last_league}")
        progress = resume_from
        skip_until = (resume_from.last_date, resume_from.last_league)
    else:
        progress = FetchProgress(
            last_date="",
            last_league="",
            total_fixtures=0,
            total_odds_entries=0,
            started_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            completed=False
        )
        skip_until = None

    all_corners = []
    all_cards = []
    all_shots = []
    all_btts = []

    skipping = skip_until is not None

    for chunk_start, chunk_end in date_ranges:
        date_str = chunk_start.strftime('%Y-%m-%d')

        for league_name, league_id in SPORTMONKS_LEAGUES.items():
            # Handle resumption
            if skipping:
                if date_str == skip_until[0] and league_name == skip_until[1]:
                    skipping = False
                    logger.info(f"Resuming from {date_str} - {league_name}")
                else:
                    continue

            logger.info(f"Fetching {league_name} from {chunk_start.date()} to {chunk_end.date()}")

            # Retry with exponential backoff
            max_retries = 3
            fixtures = None

            for retry in range(max_retries):
                try:
                    fixtures = loader.get_fixtures_between(
                        start_date=chunk_start,
                        end_date=chunk_end,
                        league_ids=[league_id],
                        include_odds=True,
                        per_page=50
                    )

                    time.sleep(delay_between_requests)
                    break  # Success, exit retry loop

                except requests.exceptions.RequestException as e:
                    wait_time = (2 ** retry) * 5  # 5s, 10s, 20s
                    if retry < max_retries - 1:
                        logger.warning(f"API error (retry {retry+1}/{max_retries}): {e}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"API error for {league_name} {date_str} after {max_retries} retries: {e}")
                        # Save checkpoint and continue to next league/date
                        progress.last_date = date_str
                        progress.last_league = league_name
                        progress.updated_at = datetime.now().isoformat()
                        save_checkpoint(progress)

            if fixtures is None:
                continue

            if not fixtures:
                continue

            progress.total_fixtures += len(fixtures)

            # Extract odds for each market type
            for fixture in fixtures:
                corners = extract_odds_for_fixture(fixture, CORNERS_MARKETS)
                cards = extract_odds_for_fixture(fixture, CARDS_MARKETS)
                shots = extract_odds_for_fixture(fixture, SHOTS_MARKETS)
                btts = extract_odds_for_fixture(fixture, BTTS_MARKETS)

                all_corners.extend(corners)
                all_cards.extend(cards)
                all_shots.extend(shots)
                all_btts.extend(btts)

                progress.total_odds_entries += len(corners) + len(cards) + len(shots) + len(btts)

            logger.info(f"  Found {len(fixtures)} fixtures, "
                       f"corners: {len([c for c in all_corners if c['fixture_id'] in [f['id'] for f in fixtures]])}, "
                       f"cards: {len([c for c in all_cards if c['fixture_id'] in [f['id'] for f in fixtures]])}")

            # Update checkpoint
            progress.last_date = date_str
            progress.last_league = league_name
            progress.updated_at = datetime.now().isoformat()
            save_checkpoint(progress)

    # Convert to DataFrames
    corners_df = pd.DataFrame(all_corners)
    cards_df = pd.DataFrame(all_cards)
    shots_df = pd.DataFrame(all_shots)
    btts_df = pd.DataFrame(all_btts)

    # Validate data quality
    logger.info("\n=== DATA VALIDATION ===")
    validation_results = []

    for df, name in [(corners_df, 'corners'), (cards_df, 'cards'), (shots_df, 'shots'), (btts_df, 'btts')]:
        result = validate_odds_data(df, name)
        validation_results.append(result)

        if result['warnings']:
            for w in result['warnings']:
                logger.warning(f"[{name}] {w}")
        if result['errors']:
            for e in result['errors']:
                logger.error(f"[{name}] {e}")

        logger.info(f"[{name}] {result['total_rows']} rows, {result.get('unique_fixtures', 0)} fixtures, "
                   f"NaN ratio: {result.get('nan_ratio', 0)*100:.1f}%")

    # Save validation report
    validation_path = OUTPUT_DIR / "validation_report.json"
    with open(validation_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    logger.info(f"Saved validation report to {validation_path}")

    # Check for critical errors
    has_errors = any(r.get('errors') for r in validation_results)
    if has_errors:
        logger.error("CRITICAL: Validation errors found! Data may be incomplete or corrupted.")

    # Save raw data
    if not corners_df.empty:
        corners_df.to_csv(RAW_ODDS_DIR / "corners_raw.csv", index=False)
        logger.info(f"Saved {len(corners_df)} raw corners odds entries")

    if not cards_df.empty:
        cards_df.to_csv(RAW_ODDS_DIR / "cards_raw.csv", index=False)
        logger.info(f"Saved {len(cards_df)} raw cards odds entries")

    if not shots_df.empty:
        shots_df.to_csv(RAW_ODDS_DIR / "shots_raw.csv", index=False)
        logger.info(f"Saved {len(shots_df)} raw shots odds entries")

    if not btts_df.empty:
        btts_df.to_csv(RAW_ODDS_DIR / "btts_raw.csv", index=False)
        logger.info(f"Saved {len(btts_df)} raw BTTS odds entries")

    # Aggregate by line (for over/under markets)
    corners_agg = aggregate_odds_by_line(corners_df)
    cards_agg = aggregate_odds_by_line(cards_df)
    shots_agg = aggregate_odds_by_line(shots_df)

    # Aggregate BTTS (yes/no market)
    btts_agg = aggregate_btts_odds(btts_df)

    # Save processed data
    if not corners_agg.empty:
        corners_agg.to_csv(PROCESSED_DIR / "corners_odds.csv", index=False)
        logger.info(f"Saved {len(corners_agg)} aggregated corners odds")

    if not cards_agg.empty:
        cards_agg.to_csv(PROCESSED_DIR / "cards_odds.csv", index=False)
        logger.info(f"Saved {len(cards_agg)} aggregated cards odds")

    if not shots_agg.empty:
        shots_agg.to_csv(PROCESSED_DIR / "shots_odds.csv", index=False)
        logger.info(f"Saved {len(shots_agg)} aggregated shots odds")

    if not btts_agg.empty:
        btts_agg.to_csv(PROCESSED_DIR / "btts_odds.csv", index=False)
        logger.info(f"Saved {len(btts_agg)} aggregated BTTS odds")

    # Mark complete
    progress.completed = True
    progress.updated_at = datetime.now().isoformat()
    save_checkpoint(progress)

    return corners_agg, cards_agg, shots_agg


def validate_odds_data(df: pd.DataFrame, market_name: str) -> Dict:
    """
    Validate odds data quality and return validation results.

    Checks:
    - Odds values are in reasonable range (1.01 to 100.0)
    - No excessive NaN values
    - Dates are valid
    - Required columns exist
    """
    results = {
        'market': market_name,
        'total_rows': len(df),
        'valid': True,
        'warnings': [],
        'errors': []
    }

    if df.empty:
        results['warnings'].append(f"No data for {market_name}")
        return results

    # Check required columns
    required_cols = ['fixture_id', 'odds', 'start_time']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        results['errors'].append(f"Missing columns: {missing_cols}")
        results['valid'] = False
        return results

    # Check odds values are reasonable (between 1.01 and 100)
    if 'odds' in df.columns:
        invalid_odds = df[(df['odds'] < 1.01) | (df['odds'] > 100)]
        if len(invalid_odds) > 0:
            pct = len(invalid_odds) / len(df) * 100
            if pct > 5:
                results['warnings'].append(f"{pct:.1f}% odds outside range 1.01-100")
            results['invalid_odds_count'] = len(invalid_odds)

    # Check NaN ratio
    nan_ratio = df['odds'].isna().sum() / len(df)
    if nan_ratio > 0.1:
        results['warnings'].append(f"{nan_ratio*100:.1f}% of odds are NaN")
    results['nan_ratio'] = nan_ratio

    # Check fixture coverage
    results['unique_fixtures'] = df['fixture_id'].nunique()
    results['avg_odds_per_fixture'] = len(df) / df['fixture_id'].nunique() if df['fixture_id'].nunique() > 0 else 0

    # Date range
    if 'start_time' in df.columns:
        df_dates = pd.to_datetime(df['start_time'], errors='coerce')
        results['date_min'] = str(df_dates.min()) if not df_dates.isna().all() else None
        results['date_max'] = str(df_dates.max()) if not df_dates.isna().all() else None

    return results


def create_odds_summary(
    corners_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    shots_df: pd.DataFrame
) -> Dict:
    """Create summary statistics for fetched odds."""
    summary = {
        'fetch_completed_at': datetime.now().isoformat(),
        'corners': {
            'total_entries': len(corners_df),
            'unique_fixtures': corners_df['fixture_id'].nunique() if not corners_df.empty else 0,
            'date_range': {
                'min': corners_df['start_time'].min() if not corners_df.empty else None,
                'max': corners_df['start_time'].max() if not corners_df.empty else None,
            },
            'lines_available': sorted(corners_df['line'].dropna().unique().tolist()) if not corners_df.empty else [],
        },
        'cards': {
            'total_entries': len(cards_df),
            'unique_fixtures': cards_df['fixture_id'].nunique() if not cards_df.empty else 0,
            'date_range': {
                'min': cards_df['start_time'].min() if not cards_df.empty else None,
                'max': cards_df['start_time'].max() if not cards_df.empty else None,
            },
            'lines_available': sorted(cards_df['line'].dropna().unique().tolist()) if not cards_df.empty else [],
        },
        'shots': {
            'total_entries': len(shots_df),
            'unique_fixtures': shots_df['fixture_id'].nunique() if not shots_df.empty else 0,
            'date_range': {
                'min': shots_df['start_time'].min() if not shots_df.empty else None,
                'max': shots_df['start_time'].max() if not shots_df.empty else None,
            },
            'lines_available': sorted(shots_df['line'].dropna().unique().tolist()) if not shots_df.empty else [],
        },
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Bulk fetch historical odds from Sportmonks')
    parser.add_argument(
        '--start-date',
        type=str,
        default='2022-08-01',
        help='Start date (YYYY-MM-DD). Default: 2022-08-01 (start of 2022-23 season)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Estimate API calls without fetching'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API requests in seconds (default: 0.5)'
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()

    # Check for resume
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            logger.info(f"Found checkpoint from {checkpoint.updated_at}")
            if checkpoint.completed:
                logger.info("Previous run completed. Starting fresh.")
                checkpoint = None
        else:
            logger.info("No checkpoint found. Starting fresh.")

    # Initialize loader
    try:
        loader = SportMonksLoader()
    except ValueError as e:
        logger.error(f"Failed to initialize loader: {e}")
        logger.error("Set SPORTSMONK_KEY environment variable")
        sys.exit(1)

    # Check subscription
    logger.info("Checking API subscription...")
    sub_info = loader.get_subscription_info()
    logger.info(f"Subscription: {sub_info}")

    # Fetch odds
    corners_df, cards_df, shots_df = fetch_historical_odds(
        loader=loader,
        start_date=start_date,
        end_date=end_date,
        resume_from=checkpoint,
        dry_run=args.dry_run,
        delay_between_requests=args.delay
    )

    if args.dry_run:
        return

    # Create and save summary
    summary = create_odds_summary(corners_df, cards_df, shots_df)

    summary_path = OUTPUT_DIR / "fetch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Print results
    print("\n" + "=" * 60)
    print("BULK FETCH COMPLETE")
    print("=" * 60)
    print(f"\nCorners odds:")
    print(f"  Entries: {summary['corners']['total_entries']}")
    print(f"  Fixtures: {summary['corners']['unique_fixtures']}")
    print(f"  Lines: {summary['corners']['lines_available'][:10]}...")

    print(f"\nCards odds:")
    print(f"  Entries: {summary['cards']['total_entries']}")
    print(f"  Fixtures: {summary['cards']['unique_fixtures']}")
    print(f"  Lines: {summary['cards']['lines_available'][:10]}...")

    print(f"\nShots odds:")
    print(f"  Entries: {summary['shots']['total_entries']}")
    print(f"  Fixtures: {summary['shots']['unique_fixtures']}")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
