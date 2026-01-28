#!/usr/bin/env python3
"""
Generate predictions with pre-match intelligence.

This script:
1. Fetches upcoming fixtures
2. Collects pre-match data (injuries, lineups, predictions, H2H)
3. Combines with historical features
4. Generates betting recommendations

Usage:
    uv run python experiments/generate_prematch_predictions.py
    uv run python experiments/generate_prematch_predictions.py --league premier_league --days 3
    uv run python experiments/generate_prematch_predictions.py --fixture 1234567
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.prematch_collector import PreMatchCollector, LEAGUE_IDS
from src.features.engineers.prematch import (
    PreMatchFeatureEngineer,
    create_prematch_features_for_fixture,
)
from src.ml.confidence_adjuster import LineupConfidenceAdjuster, ConfidenceAdjustment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path('data/06-prematch')
FEATURES_FILE = Path('data/03-features/features_with_sportmonks_odds.parquet')

# Betting thresholds (from previous optimization)
BETTING_THRESHOLDS = {
    'away_win': {'prob_threshold': 0.60, 'min_odds': 1.5, 'max_odds': 8.0},
    'shots_over_22_5': {'prob_threshold': 0.75, 'min_odds': 1.3, 'max_odds': 3.0},
    'shots_over_24_5': {'prob_threshold': 0.75, 'min_odds': 1.5, 'max_odds': 3.5},
    'corners_over_9_5': {'prob_threshold': 0.75, 'min_odds': 1.5, 'max_odds': 3.0},
}


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_prematch_data(
    league: str,
    season: int,
    next_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Collect pre-match data for upcoming fixtures.

    Args:
        league: League key (e.g., 'premier_league')
        season: Season year
        next_n: Number of fixtures to collect

    Returns:
        List of pre-match data dicts
    """
    collector = PreMatchCollector()
    league_id = LEAGUE_IDS.get(league)

    if not league_id:
        raise ValueError(f"Unknown league: {league}")

    logger.info(f"Collecting pre-match data for {league} ({league_id})")

    # Get upcoming fixtures
    fixtures = collector.get_upcoming_fixtures(league_id, season, next_n)

    if fixtures.empty:
        logger.warning("No upcoming fixtures found")
        return []

    logger.info(f"Found {len(fixtures)} upcoming fixtures")

    # Collect data for each fixture
    results = []
    for _, fixture in fixtures.iterrows():
        fixture_id = fixture['fixture_id']
        logger.info(f"Collecting data for {fixture['home_team_name']} vs {fixture['away_team_name']}")

        try:
            data = collector.collect_prematch_data(fixture_id)
            data['fixture_info'] = fixture.to_dict()
            results.append(data)
        except Exception as e:
            logger.error(f"Failed to collect data for fixture {fixture_id}: {e}")

    return results


def collect_single_fixture(fixture_id: int) -> Dict[str, Any]:
    """Collect pre-match data for a single fixture."""
    collector = PreMatchCollector()
    return collector.collect_prematch_data(fixture_id)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(prematch_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create features from pre-match data.

    Args:
        prematch_data: List of pre-match data dicts

    Returns:
        DataFrame with pre-match features
    """
    engineer = PreMatchFeatureEngineer()

    # Build matches DataFrame
    matches_records = []
    prematch_map = {}

    for data in prematch_data:
        fixture_info = data.get('fixture_info', {})
        fixture_id = data['fixture_id']

        matches_records.append({
            'fixture_id': fixture_id,
            'date': fixture_info.get('date'),
            'home_team_id': fixture_info.get('home_team_id'),
            'home_team_name': fixture_info.get('home_team_name'),
            'away_team_id': fixture_info.get('away_team_id'),
            'away_team_name': fixture_info.get('away_team_name'),
            'league_id': fixture_info.get('league_id'),
        })

        prematch_map[fixture_id] = data

    matches = pd.DataFrame(matches_records)

    # Create features
    features = engineer.create_features({
        'matches': matches,
        'prematch': prematch_map,
    })

    # Merge with match info
    result = matches.merge(features, on='fixture_id', how='left')

    return result


# =============================================================================
# ANALYSIS & RECOMMENDATIONS
# =============================================================================

def analyze_injuries(prematch_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze injuries across all fixtures.

    Returns:
        DataFrame with injury summary
    """
    records = []

    for data in prematch_data:
        fixture_info = data.get('fixture_info', {})
        injuries = data.get('injuries', pd.DataFrame())

        record = {
            'fixture_id': data['fixture_id'],
            'match': f"{fixture_info.get('home_team_name')} vs {fixture_info.get('away_team_name')}",
            'date': fixture_info.get('date'),
            'total_injuries': len(injuries) if not injuries.empty else 0,
            'home_injuries': 0,
            'away_injuries': 0,
            'key_players_out': [],
        }

        if not injuries.empty:
            home_id = fixture_info.get('home_team_id')
            away_id = fixture_info.get('away_team_id')

            record['home_injuries'] = len(injuries[injuries['team_id'] == home_id])
            record['away_injuries'] = len(injuries[injuries['team_id'] == away_id])

            # List injured players
            for _, inj in injuries.iterrows():
                record['key_players_out'].append(
                    f"{inj['player_name']} ({inj['team_name']}) - {inj['injury_reason']}"
                )

        records.append(record)

    return pd.DataFrame(records)


def analyze_predictions(prematch_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze API predictions across all fixtures.

    Returns:
        DataFrame with prediction summary
    """
    records = []

    for data in prematch_data:
        fixture_info = data.get('fixture_info', {})
        predictions = data.get('predictions', {})

        record = {
            'fixture_id': data['fixture_id'],
            'match': f"{fixture_info.get('home_team_name')} vs {fixture_info.get('away_team_name')}",
            'date': fixture_info.get('date'),
        }

        if predictions:
            percent = predictions.get('percent', {})
            record['home_pct'] = percent.get('home', 'N/A')
            record['draw_pct'] = percent.get('draw', 'N/A')
            record['away_pct'] = percent.get('away', 'N/A')
            record['advice'] = predictions.get('advice', 'N/A')

            # Comparison metrics
            comparison = predictions.get('comparison', {})
            record['form_home'] = comparison.get('form', {}).get('home', 'N/A')
            record['attack_home'] = comparison.get('att', {}).get('home', 'N/A')
            record['defense_home'] = comparison.get('def', {}).get('home', 'N/A')
        else:
            record['home_pct'] = 'N/A'
            record['draw_pct'] = 'N/A'
            record['away_pct'] = 'N/A'
            record['advice'] = 'N/A'

        records.append(record)

    return pd.DataFrame(records)


def apply_lineup_adjustments(
    features: pd.DataFrame,
    prematch_data: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Apply lineup-based confidence adjustments to predictions.

    When lineups are confirmed (~1hr before match), adjust our
    confidence based on key player availability.

    Args:
        features: Pre-match features DataFrame
        prematch_data: Raw pre-match data with lineup info

    Returns:
        DataFrame with adjusted confidence columns
    """
    adjuster = LineupConfidenceAdjuster()
    adjusted_features = features.copy()

    # Initialize adjustment columns
    adjusted_features['lineup_adj_factor'] = 1.0
    adjusted_features['lineup_adj_reason'] = ''
    adjusted_features['home_strength'] = 1.0
    adjusted_features['away_strength'] = 1.0

    for data in prematch_data:
        fixture_id = data['fixture_id']
        lineups = data.get('lineups', {})
        fixture_info = data.get('fixture_info', {})

        # Only adjust if lineups are available
        if not lineups.get('available', False):
            continue

        home_lineup = lineups.get('home', {})
        away_lineup = lineups.get('away', {})
        home_team_id = fixture_info.get('home_team_id')
        away_team_id = fixture_info.get('away_team_id')
        home_team_name = fixture_info.get('home_team_name', '')
        away_team_name = fixture_info.get('away_team_name', '')

        # Analyze lineups
        home_analysis = adjuster.analyze_lineup(
            home_team_id, home_team_name, home_lineup)
        away_analysis = adjuster.analyze_lineup(
            away_team_id, away_team_name, away_lineup)

        # Update features for this fixture
        mask = adjusted_features['fixture_id'] == fixture_id
        if not mask.any():
            continue

        adjusted_features.loc[mask, 'home_strength'] = home_analysis.strength_score
        adjusted_features.loc[mask, 'away_strength'] = away_analysis.strength_score

        # Calculate adjustment for primary market (using away_win as reference)
        away_pct = adjusted_features.loc[mask, 'pm_pred_away_pct'].values[0] / 100 if 'pm_pred_away_pct' in adjusted_features.columns else 0.3

        adjustment = adjuster.calculate_adjustment(
            fixture_id=fixture_id,
            market='away_win',
            original_prob=away_pct,
            home_analysis=home_analysis,
            away_analysis=away_analysis,
        )

        adjusted_features.loc[mask, 'lineup_adj_factor'] = adjustment.adjustment_factor
        adjusted_features.loc[mask, 'lineup_adj_reason'] = '; '.join(adjustment.reasons)

        # Log significant adjustments
        if abs(adjustment.adjustment_factor - 1.0) > 0.05:
            logger.info(
                f"Lineup adjustment for {home_team_name} vs {away_team_name}: "
                f"factor={adjustment.adjustment_factor:.2f}, "
                f"reasons={adjustment.reasons}"
            )

    return adjusted_features


def generate_betting_signals(
    features: pd.DataFrame,
    predictions_data: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Generate betting signals based on pre-match intelligence.

    Args:
        features: Pre-match features DataFrame
        predictions_data: Raw predictions data

    Returns:
        DataFrame with betting signals
    """
    signals = []

    for _, row in features.iterrows():
        fixture_id = row['fixture_id']

        # Find corresponding predictions
        pred_data = None
        for data in predictions_data:
            if data['fixture_id'] == fixture_id:
                pred_data = data.get('predictions', {})
                break

        if not pred_data:
            continue

        signal = {
            'fixture_id': fixture_id,
            'match': f"{row['home_team_name']} vs {row['away_team_name']}",
            'date': row['date'],
            'signals': [],
        }

        # Away win signal
        away_pct = row.get('pm_pred_away_pct', 0)
        if away_pct >= BETTING_THRESHOLDS['away_win']['prob_threshold']:
            signal['signals'].append({
                'market': 'Away Win',
                'confidence': away_pct,
                'reason': f"API predicts {away_pct*100:.0f}% away win probability",
            })

        # Form-based signals
        form_diff = row.get('pm_form_points_diff', 0)
        if form_diff < -5:  # Away team much better form
            signal['signals'].append({
                'market': 'Away Win/Draw',
                'confidence': 0.6,
                'reason': f"Away team has {abs(form_diff)} more form points in last 5",
            })
        elif form_diff > 5:  # Home team much better form
            signal['signals'].append({
                'market': 'Home Win',
                'confidence': 0.6,
                'reason': f"Home team has {form_diff} more form points in last 5",
            })

        # Key player injury signals
        if row.get('pm_home_key_player_out', 0) and not row.get('pm_away_key_player_out', 0):
            signal['signals'].append({
                'market': 'Away +0.5 AH',
                'confidence': 0.55,
                'reason': "Home team missing key player",
            })
        elif row.get('pm_away_key_player_out', 0) and not row.get('pm_home_key_player_out', 0):
            signal['signals'].append({
                'market': 'Home -0.5 AH',
                'confidence': 0.55,
                'reason': "Away team missing key player",
            })

        # Goals signals
        expected_goals = row.get('pm_expected_total_goals', 2.7)
        if expected_goals > 3.0:
            signal['signals'].append({
                'market': 'Over 2.5 Goals',
                'confidence': 0.6,
                'reason': f"Expected {expected_goals:.1f} total goals based on team averages",
            })
        elif expected_goals < 2.2:
            signal['signals'].append({
                'market': 'Under 2.5 Goals',
                'confidence': 0.6,
                'reason': f"Expected {expected_goals:.1f} total goals based on team averages",
            })

        # Lineup-based signals (only if lineups available)
        lineup_adj = row.get('lineup_adj_factor', 1.0)
        home_strength = row.get('home_strength', 1.0)
        away_strength = row.get('away_strength', 1.0)
        lineup_reason = row.get('lineup_adj_reason', '')

        if home_strength < 0.7 and away_strength >= 0.9:
            # Home team significantly weakened, away team full strength
            adjusted_confidence = min(0.65, away_pct + 0.1) if away_pct else 0.6
            signal['signals'].append({
                'market': 'Away Win (Lineup Boost)',
                'confidence': adjusted_confidence,
                'reason': f"Home team at {home_strength*100:.0f}% strength, "
                         f"away at {away_strength*100:.0f}%. {lineup_reason}",
            })
        elif away_strength < 0.7 and home_strength >= 0.9:
            # Away team significantly weakened
            signal['signals'].append({
                'market': 'Home Win (Lineup Boost)',
                'confidence': 0.6,
                'reason': f"Away team at {away_strength*100:.0f}% strength, "
                         f"home at {home_strength*100:.0f}%. {lineup_reason}",
            })

        if signal['signals']:
            signals.append(signal)

    return pd.DataFrame(signals)


# =============================================================================
# OUTPUT
# =============================================================================

def save_results(
    prematch_data: List[Dict[str, Any]],
    features: pd.DataFrame,
    output_dir: Path
) -> None:
    """Save all results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save raw pre-match data
    raw_output = []
    for data in prematch_data:
        raw_record = {
            'fixture_id': data['fixture_id'],
            'collected_at': data['collected_at'],
            'fixture_info': data.get('fixture_info', {}),
            'injuries_count': len(data.get('injuries', pd.DataFrame())),
            'lineups_available': data.get('lineups', {}).get('available', False),
            'predictions': data.get('predictions', {}),
            'h2h_summary': data.get('h2h_summary', {}),
        }
        raw_output.append(raw_record)

    with open(output_dir / f'prematch_raw_{timestamp}.json', 'w') as f:
        json.dump(raw_output, f, indent=2, default=str)

    # Save features
    features.to_csv(output_dir / f'prematch_features_{timestamp}.csv', index=False)

    # Save injury analysis
    injury_analysis = analyze_injuries(prematch_data)
    injury_analysis.to_csv(output_dir / f'injuries_{timestamp}.csv', index=False)

    # Save prediction analysis
    pred_analysis = analyze_predictions(prematch_data)
    pred_analysis.to_csv(output_dir / f'predictions_{timestamp}.csv', index=False)

    # Save betting signals
    signals = generate_betting_signals(features, prematch_data)
    signals.to_json(output_dir / f'signals_{timestamp}.json', orient='records', indent=2)

    logger.info(f"Results saved to {output_dir}")


def print_summary(
    prematch_data: List[Dict[str, Any]],
    features: pd.DataFrame
) -> None:
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("PRE-MATCH INTELLIGENCE SUMMARY")
    print("=" * 70)

    for data in prematch_data:
        fixture_info = data.get('fixture_info', {})
        predictions = data.get('predictions', {})
        injuries = data.get('injuries', pd.DataFrame())

        print(f"\n{fixture_info.get('home_team_name')} vs {fixture_info.get('away_team_name')}")
        print(f"  Date: {fixture_info.get('date')}")
        print(f"  Fixture ID: {data['fixture_id']}")

        # Injuries
        print(f"  Injuries: {len(injuries)} total")
        if not injuries.empty:
            for _, inj in injuries.head(3).iterrows():
                print(f"    - {inj['player_name']} ({inj['team_name']}): {inj['injury_reason']}")

        # Lineups
        lineups = data.get('lineups', {})
        if lineups.get('available'):
            home = lineups.get('home', {})
            away = lineups.get('away', {})
            print(f"  Formations: {home.get('formation')} vs {away.get('formation')}")

            # Show strength scores from features if available
            fixture_row = features[features['fixture_id'] == data['fixture_id']]
            if not fixture_row.empty:
                home_str = fixture_row['home_strength'].values[0]
                away_str = fixture_row['away_strength'].values[0]
                adj_factor = fixture_row['lineup_adj_factor'].values[0]
                print(f"  Strength: Home={home_str:.0%} Away={away_str:.0%}")
                if abs(adj_factor - 1.0) > 0.02:
                    adj_reason = fixture_row['lineup_adj_reason'].values[0]
                    print(f"  Lineup Adjustment: {adj_factor:.2f}x - {adj_reason}")
        else:
            print("  Lineups: Not yet available")

        # Predictions
        if predictions:
            percent = predictions.get('percent', {})
            print(f"  Win %: H={percent.get('home')} D={percent.get('draw')} A={percent.get('away')}")
            print(f"  Advice: {predictions.get('advice')}")

    # Betting signals
    signals = generate_betting_signals(features, prematch_data)
    if not signals.empty:
        print("\n" + "-" * 70)
        print("BETTING SIGNALS")
        print("-" * 70)
        for _, row in signals.iterrows():
            print(f"\n{row['match']}")
            for sig in row['signals']:
                print(f"  [{sig['market']}] {sig['confidence']*100:.0f}% - {sig['reason']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate pre-match predictions')
    parser.add_argument('--league', type=str, default='premier_league',
                       choices=list(LEAGUE_IDS.keys()),
                       help='League to analyze')
    parser.add_argument('--season', type=int, default=2024,
                       help='Season year')
    parser.add_argument('--fixtures', type=int, default=10,
                       help='Number of upcoming fixtures')
    parser.add_argument('--fixture', type=int,
                       help='Specific fixture ID to analyze')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR),
                       help='Output directory')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.fixture:
        # Single fixture mode
        logger.info(f"Collecting data for fixture {args.fixture}")
        data = collect_single_fixture(args.fixture)
        prematch_data = [data]
    else:
        # League mode
        prematch_data = collect_prematch_data(args.league, args.season, args.fixtures)

    if not prematch_data:
        logger.error("No data collected")
        return 1

    # Create features
    features = create_features(prematch_data)
    logger.info(f"Created {len(features)} feature rows with {len(features.columns)} columns")

    # Apply lineup-based confidence adjustments
    features = apply_lineup_adjustments(features, prematch_data)
    lineups_available = sum(1 for d in prematch_data if d.get('lineups', {}).get('available', False))
    logger.info(f"Applied lineup adjustments ({lineups_available} fixtures with lineups)")

    # Print summary
    print_summary(prematch_data, features)

    # Save results
    if not args.no_save:
        save_results(prematch_data, features, output_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
