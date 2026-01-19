#!/usr/bin/env python
"""
Create Cross-Market Features

Based on xgbfir analysis findings:
- CORNERS: shots predict corners (away_shots × home_shots = 3364.6 gain)
- SHOTS: corners predict shots (away_corners × home_corners = 1883.5 gain)
- FOULS: yellows × odds_upset_potential = 1061.6 gain

Creates interaction features to capture these relationships.

Usage:
    python experiments/create_cross_market_features.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def create_cross_market_features():
    """Create features based on xgbfir interaction findings."""
    print("=" * 70)
    print("CREATING CROSS-MARKET FEATURES")
    print("=" * 70)

    # Load features
    features_path = Path("data/03-features/features_all_5leagues_with_odds.csv")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} matches")

    # Check available columns
    cols = df.columns.tolist()

    # Find relevant columns
    shots_cols = [c for c in cols if 'shot' in c.lower()]
    corners_cols = [c for c in cols if 'corner' in c.lower()]
    fouls_cols = [c for c in cols if 'foul' in c.lower()]
    yellows_cols = [c for c in cols if 'yellow' in c.lower()]
    odds_cols = [c for c in cols if 'odds' in c.lower()]

    print(f"\nAvailable columns:")
    print(f"  Shots: {shots_cols[:5]}...")
    print(f"  Corners: {corners_cols[:5] if corners_cols else 'NONE'}...")
    print(f"  Fouls: {fouls_cols[:5] if fouls_cols else 'NONE'}...")
    print(f"  Yellows: {yellows_cols[:5] if yellows_cols else 'NONE'}...")
    print(f"  Odds: {odds_cols[:5]}...")

    # Create cross-market features
    features_created = []

    # 1. Shots-Corners interaction (for corner predictions)
    # xgbfir showed away_shots × home_shots predicts corners
    if 'home_shots_ema' in cols and 'away_shots_ema' in cols:
        df['cross_shots_product'] = df['home_shots_ema'] * df['away_shots_ema']
        df['cross_shots_total'] = df['home_shots_ema'] + df['away_shots_ema']
        df['cross_shots_diff'] = df['home_shots_ema'] - df['away_shots_ema']
        features_created.extend(['cross_shots_product', 'cross_shots_total', 'cross_shots_diff'])

    if 'home_total_shots_ema' in cols and 'away_total_shots_ema' in cols:
        df['cross_total_shots_product'] = df['home_total_shots_ema'] * df['away_total_shots_ema']
        df['cross_total_shots_sum'] = df['home_total_shots_ema'] + df['away_total_shots_ema']
        features_created.extend(['cross_total_shots_product', 'cross_total_shots_sum'])

    # 2. Corners-Shots interaction (for shots predictions)
    # xgbfir showed away_corners × home_corners predicts shots
    if 'home_corners_ema' in cols and 'away_corners_ema' in cols:
        df['cross_corners_product'] = df['home_corners_ema'] * df['away_corners_ema']
        df['cross_corners_total'] = df['home_corners_ema'] + df['away_corners_ema']
        df['cross_corners_diff'] = df['home_corners_ema'] - df['away_corners_ema']
        features_created.extend(['cross_corners_product', 'cross_corners_total', 'cross_corners_diff'])

    # 3. Yellows × Odds interaction (for fouls predictions)
    # xgbfir showed away_avg_yellows × odds_upset_potential predicts fouls
    if 'home_avg_yellows' in cols and 'away_avg_yellows' in cols:
        df['cross_yellows_product'] = df['home_avg_yellows'] * df['away_avg_yellows']
        df['cross_yellows_total'] = df['home_avg_yellows'] + df['away_avg_yellows']
        features_created.extend(['cross_yellows_product', 'cross_yellows_total'])

        if 'odds_upset_potential' in cols:
            df['cross_yellows_upset'] = df['away_avg_yellows'] * df['odds_upset_potential']
            df['cross_yellows_upset_home'] = df['home_avg_yellows'] * df['odds_upset_potential']
            features_created.extend(['cross_yellows_upset', 'cross_yellows_upset_home'])

    # 4. Derby/rivalry × discipline (more fouls in heated matches)
    if 'is_derby' in cols and 'home_avg_yellows' in cols:
        df['cross_derby_yellows'] = df['is_derby'] * (df['home_avg_yellows'] + df['away_avg_yellows'])
        features_created.append('cross_derby_yellows')

    # 5. Form × Shots (attacking form predicts shots)
    if 'home_goals_scored_ema' in cols and 'home_shots_ema' in cols:
        df['cross_form_shots_home'] = df['home_goals_scored_ema'] * df['home_shots_ema']
        features_created.append('cross_form_shots_home')
    if 'away_goals_scored_ema' in cols and 'away_shots_ema' in cols:
        df['cross_form_shots_away'] = df['away_goals_scored_ema'] * df['away_shots_ema']
        features_created.append('cross_form_shots_away')

    # 6. xG × Shots (expected goals correlates with shot volume)
    if 'xg_home_expected' in cols and 'home_shots_ema' in cols:
        df['cross_xg_shots_home'] = df['xg_home_expected'] * df['home_shots_ema']
        features_created.append('cross_xg_shots_home')
    if 'xg_away_expected' in cols and 'away_shots_ema' in cols:
        df['cross_xg_shots_away'] = df['xg_away_expected'] * df['away_shots_ema']
        features_created.append('cross_xg_shots_away')

    # 7. Rest days × Stats (fatigue affects play)
    if 'home_rest_days' in cols:
        if 'home_shots_ema' in cols:
            df['cross_rest_shots_home'] = df['home_rest_days'] * df['home_shots_ema']
            features_created.append('cross_rest_shots_home')
        if 'home_avg_yellows' in cols:
            df['cross_rest_yellows_home'] = df['home_rest_days'] * df['home_avg_yellows']
            features_created.append('cross_rest_yellows_home')

    # 8. ELO diff × Odds (strong teams + underdog potential)
    if 'elo_diff' in cols and 'odds_upset_potential' in cols:
        df['cross_elo_upset'] = df['elo_diff'] * df['odds_upset_potential']
        features_created.append('cross_elo_upset')

    # 9. Attack strength × Defense weakness (for goals/corners)
    if 'home_attack_strength' in cols and 'away_defense_strength' in cols:
        df['cross_attack_defense'] = df['home_attack_strength'] * df['away_defense_strength']
        features_created.append('cross_attack_defense')

    # Summary
    print(f"\n{'=' * 70}")
    print(f"FEATURES CREATED: {len(features_created)}")
    print("=" * 70)
    for feat in features_created:
        if feat in df.columns:
            print(f"  {feat}: mean={df[feat].mean():.3f}, std={df[feat].std():.3f}")

    # Save
    df.to_csv(features_path, index=False)
    print(f"\nUpdated features file: {features_path}")

    return features_created


if __name__ == "__main__":
    create_cross_market_features()
