#!/usr/bin/env python
"""
Merge Sportmonks historical odds with our feature dataset.

Matches fixtures by team names and dates, then adds real odds
columns to replace synthetic/estimated odds.

Usage:
    python scripts/merge_sportmonks_odds.py
    python scripts/merge_sportmonks_odds.py --features path/to/features.csv
    python scripts/merge_sportmonks_odds.py --corners-line 9.5 --cards-line 4.5
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.odds.sportmonks_loader import normalize_sportmonks_team, SPORTMONKS_TEAM_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SPORTMONKS_ODDS_DIR = Path("data/sportmonks_odds/processed")
DEFAULT_FEATURES_PATH = Path("data/03-features/features_all_5leagues_with_odds.csv")
OUTPUT_DIR = Path("data/sportmonks_odds/merged")


def build_team_name_index(odds_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build index of team name variations for fuzzy matching."""
    teams = set()
    for col in ['home_team', 'away_team', 'home_team_normalized', 'away_team_normalized']:
        if col in odds_df.columns:
            teams.update(odds_df[col].dropna().unique())
    return {t: t for t in teams}


def normalize_team_name(name: str, team_index: Dict[str, str]) -> str:
    """Normalize team name using fuzzy matching."""
    if name in team_index:
        return team_index[name]

    # Try predefined mapping
    normalized = normalize_sportmonks_team(name)
    if normalized in team_index:
        return normalized

    # Fuzzy match
    match = process.extractOne(name, list(team_index.keys()), scorer=fuzz.ratio)
    if match and match[1] >= 85:
        return match[0]

    return name


def match_fixtures(
    features_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    date_tolerance_days: int = 1
) -> pd.DataFrame:
    """
    Match features rows to odds rows by team names and date.

    Returns DataFrame with matched fixture_ids.
    """
    if odds_df.empty:
        return pd.DataFrame()

    # Build team index
    team_index = build_team_name_index(odds_df)

    # Prepare odds lookup
    odds_df = odds_df.copy()
    odds_df['start_date'] = pd.to_datetime(odds_df['start_time']).dt.date

    # Normalize features team names
    features_df = features_df.copy()

    # Determine team column names in features
    if 'home_team_name' in features_df.columns:
        home_col = 'home_team_name'
        away_col = 'away_team_name'
    elif 'home_team' in features_df.columns:
        home_col = 'home_team'
        away_col = 'away_team'
    else:
        home_col = 'HomeTeam'
        away_col = 'AwayTeam'
    date_col = 'date' if 'date' in features_df.columns else 'Date'

    if date_col not in features_df.columns:
        logger.error(f"Date column not found. Available: {features_df.columns.tolist()}")
        return pd.DataFrame()

    features_df['match_date'] = pd.to_datetime(features_df[date_col]).dt.date
    features_df['home_normalized'] = features_df[home_col].apply(
        lambda x: normalize_team_name(str(x), team_index)
    )
    features_df['away_normalized'] = features_df[away_col].apply(
        lambda x: normalize_team_name(str(x), team_index)
    )

    # Match fixtures
    matches = []

    for idx, row in features_df.iterrows():
        match_date = row['match_date']
        home = row['home_normalized']
        away = row['away_normalized']

        # Look for matching fixture in odds
        date_range = [
            match_date - timedelta(days=date_tolerance_days),
            match_date + timedelta(days=date_tolerance_days)
        ]

        candidates = odds_df[
            (odds_df['start_date'] >= date_range[0]) &
            (odds_df['start_date'] <= date_range[1])
        ]

        # Try exact match first
        exact_match = candidates[
            ((candidates['home_team_normalized'] == home) | (candidates['home_team'] == home)) &
            ((candidates['away_team_normalized'] == away) | (candidates['away_team'] == away))
        ]

        if not exact_match.empty:
            matches.append({
                'features_idx': idx,
                'fixture_id': exact_match.iloc[0]['fixture_id'],
                'match_quality': 'exact'
            })
            continue

        # Try fuzzy match
        for _, cand in candidates.iterrows():
            home_score = max(
                fuzz.ratio(home, str(cand.get('home_team', ''))),
                fuzz.ratio(home, str(cand.get('home_team_normalized', '')))
            )
            away_score = max(
                fuzz.ratio(away, str(cand.get('away_team', ''))),
                fuzz.ratio(away, str(cand.get('away_team_normalized', '')))
            )

            if home_score >= 80 and away_score >= 80:
                matches.append({
                    'features_idx': idx,
                    'fixture_id': cand['fixture_id'],
                    'match_quality': 'fuzzy',
                    'home_score': home_score,
                    'away_score': away_score
                })
                break

    return pd.DataFrame(matches)


def get_odds_for_line(
    odds_df: pd.DataFrame,
    fixture_id: int,
    target_line: float,
    line_tolerance: Optional[float] = 0.5
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Get over/under odds for a specific line.

    Returns (over_odds, under_odds, actual_line_used).
    If line_tolerance is None, accepts any available line (closest to target).
    """
    fixture_odds = odds_df[odds_df['fixture_id'] == fixture_id]

    if fixture_odds.empty:
        return None, None, None

    # Find closest line
    lines = fixture_odds['line'].dropna().unique()
    if len(lines) == 0:
        return None, None, None

    closest_line = min(lines, key=lambda x: abs(x - target_line))

    # If tolerance is None, accept any line; otherwise check tolerance
    if line_tolerance is not None and abs(closest_line - target_line) > line_tolerance:
        return None, None, None

    line_odds = fixture_odds[fixture_odds['line'] == closest_line].iloc[0]

    return line_odds.get('over_avg'), line_odds.get('under_avg'), closest_line


def merge_odds_with_features(
    features_path: Path,
    corners_line: float = 9.5,
    cards_line: float = 4.5,
    shots_line: float = 10.5
) -> pd.DataFrame:
    """
    Merge Sportmonks odds with feature dataset.

    Returns features DataFrame with real odds columns added.
    """
    # Load features
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(features_df)} feature rows")

    # Load odds
    corners_path = SPORTMONKS_ODDS_DIR / "corners_odds.csv"
    cards_path = SPORTMONKS_ODDS_DIR / "cards_odds.csv"
    shots_path = SPORTMONKS_ODDS_DIR / "shots_odds.csv"
    btts_path = SPORTMONKS_ODDS_DIR / "btts_odds.csv"

    corners_df = pd.read_csv(corners_path) if corners_path.exists() else pd.DataFrame()
    cards_df = pd.read_csv(cards_path) if cards_path.exists() else pd.DataFrame()
    shots_df = pd.read_csv(shots_path) if shots_path.exists() else pd.DataFrame()
    btts_df = pd.read_csv(btts_path) if btts_path.exists() else pd.DataFrame()

    logger.info(f"Loaded odds: corners={len(corners_df)}, cards={len(cards_df)}, shots={len(shots_df)}, btts={len(btts_df)}")

    # Match fixtures for each market
    results = features_df.copy()

    # Initialize new columns
    results['sm_corners_over_odds'] = np.nan
    results['sm_corners_under_odds'] = np.nan
    results['sm_corners_line'] = np.nan
    results['sm_cards_over_odds'] = np.nan
    results['sm_cards_under_odds'] = np.nan
    results['sm_cards_line'] = np.nan
    results['sm_shots_over_odds'] = np.nan
    results['sm_shots_under_odds'] = np.nan
    results['sm_shots_line'] = np.nan
    results['sm_btts_yes_odds'] = np.nan
    results['sm_btts_no_odds'] = np.nan
    results['sm_fixture_id'] = np.nan

    # Match corners - use flexible tolerance to get any available line
    if not corners_df.empty:
        logger.info("Matching corners odds...")
        corners_matches = match_fixtures(features_df, corners_df)
        logger.info(f"Matched {len(corners_matches)} fixtures for corners")

        for _, match in corners_matches.iterrows():
            idx = match['features_idx']
            fix_id = match['fixture_id']

            # Use None tolerance to accept any line (closest to target)
            over_odds, under_odds, actual_line = get_odds_for_line(corners_df, fix_id, corners_line, line_tolerance=None)

            if over_odds is not None:
                results.loc[idx, 'sm_corners_over_odds'] = over_odds
                results.loc[idx, 'sm_corners_under_odds'] = under_odds
                results.loc[idx, 'sm_corners_line'] = actual_line
                results.loc[idx, 'sm_fixture_id'] = fix_id

    # Match cards - use flexible tolerance
    if not cards_df.empty:
        logger.info("Matching cards odds...")
        cards_matches = match_fixtures(features_df, cards_df)
        logger.info(f"Matched {len(cards_matches)} fixtures for cards")

        for _, match in cards_matches.iterrows():
            idx = match['features_idx']
            fix_id = match['fixture_id']

            over_odds, under_odds, actual_line = get_odds_for_line(cards_df, fix_id, cards_line, line_tolerance=None)

            if over_odds is not None:
                results.loc[idx, 'sm_cards_over_odds'] = over_odds
                results.loc[idx, 'sm_cards_under_odds'] = under_odds
                results.loc[idx, 'sm_cards_line'] = actual_line

    # Match shots - use flexible tolerance (lines vary widely: 10.5-33.5)
    if not shots_df.empty:
        logger.info("Matching shots odds...")
        shots_matches = match_fixtures(features_df, shots_df)
        logger.info(f"Matched {len(shots_matches)} fixtures for shots")

        for _, match in shots_matches.iterrows():
            idx = match['features_idx']
            fix_id = match['fixture_id']

            over_odds, under_odds, actual_line = get_odds_for_line(shots_df, fix_id, shots_line, line_tolerance=None)

            if over_odds is not None:
                results.loc[idx, 'sm_shots_over_odds'] = over_odds
                results.loc[idx, 'sm_shots_under_odds'] = under_odds
                results.loc[idx, 'sm_shots_line'] = actual_line

    # Match BTTS
    if not btts_df.empty:
        logger.info("Matching BTTS odds...")
        btts_matches = match_fixtures(features_df, btts_df)
        logger.info(f"Matched {len(btts_matches)} fixtures for BTTS")

        for _, match in btts_matches.iterrows():
            idx = match['features_idx']
            fix_id = match['fixture_id']

            # Get BTTS odds directly (no line needed)
            fixture_btts = btts_df[btts_df['fixture_id'] == fix_id]
            if not fixture_btts.empty:
                row = fixture_btts.iloc[0]
                if pd.notna(row.get('yes_avg')):
                    results.loc[idx, 'sm_btts_yes_odds'] = row['yes_avg']
                    results.loc[idx, 'sm_btts_no_odds'] = row.get('no_avg')

    # Report coverage
    corners_coverage = results['sm_corners_over_odds'].notna().sum()
    cards_coverage = results['sm_cards_over_odds'].notna().sum()
    shots_coverage = results['sm_shots_over_odds'].notna().sum()
    btts_coverage = results['sm_btts_yes_odds'].notna().sum()

    logger.info(f"\nOdds coverage:")
    logger.info(f"  Corners: {corners_coverage}/{len(results)} ({corners_coverage/len(results)*100:.1f}%)")
    logger.info(f"  Cards: {cards_coverage}/{len(results)} ({cards_coverage/len(results)*100:.1f}%)")
    logger.info(f"  Shots: {shots_coverage}/{len(results)} ({shots_coverage/len(results)*100:.1f}%)")
    logger.info(f"  BTTS: {btts_coverage}/{len(results)} ({btts_coverage/len(results)*100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Merge Sportmonks odds with features')
    parser.add_argument(
        '--features',
        type=str,
        default=str(DEFAULT_FEATURES_PATH),
        help='Path to features CSV'
    )
    parser.add_argument(
        '--corners-line',
        type=float,
        default=9.5,
        help='Target corners line (default: 9.5)'
    )
    parser.add_argument(
        '--cards-line',
        type=float,
        default=4.5,
        help='Target cards line (default: 4.5)'
    )
    parser.add_argument(
        '--shots-line',
        type=float,
        default=10.5,
        help='Target shots line (default: 10.5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for merged features'
    )

    args = parser.parse_args()

    features_path = Path(args.features)
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        sys.exit(1)

    # Merge odds
    merged_df = merge_odds_with_features(
        features_path=features_path,
        corners_line=args.corners_line,
        cards_line=args.cards_line,
        shots_line=args.shots_line
    )

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / "features_with_sportmonks_odds.csv"

    merged_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved merged features to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"\nTotal rows: {len(merged_df)}")
    print(f"Corners odds: {merged_df['sm_corners_over_odds'].notna().sum()} matches")
    print(f"Cards odds: {merged_df['sm_cards_over_odds'].notna().sum()} matches")
    print(f"Shots odds: {merged_df['sm_shots_over_odds'].notna().sum()} matches")
    print(f"\nOutput: {output_path}")


if __name__ == '__main__':
    main()
