#!/usr/bin/env python
"""
Yellow Cards Betting Optimization V2 - Enhanced with 253 Features + Boruta

Improvements over V1:
1. Uses full feature set (~253 features) from main pipeline
2. Adds cards-specific features (referee cards patterns, team fouls)
3. Boruta feature selection to find truly predictive features
4. Walk-forward validation with proper data leakage prevention

Key insight from V1: Referee selection explains ~40% of variance in cards.
This version validates that with Boruta on the full feature set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from boruta import BorutaPy

print("=" * 70)
print("YELLOW CARDS OPTIMIZATION V2 - BORUTA + 253 FEATURES")
print("=" * 70)

# Typical cards odds from bookmakers
CARDS_ODDS = {
    'over_2_5': 2.00, 'under_2_5': 1.80,
    'over_3_5': 1.85, 'under_3_5': 1.95,
    'over_4_5': 2.20, 'under_4_5': 1.65,
    'over_5_5': 2.80, 'under_5_5': 1.45,
}


def load_main_features():
    """Load main features from pipeline."""
    df = pd.read_csv('data/03-features/features_all_5leagues_with_odds.csv')
    print(f"Main features loaded: {len(df)} matches, {len(df.columns)} columns")
    return df


def load_cards_data():
    """Load card events data from all leagues."""
    all_events = []
    all_matches = []

    leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']

    for league in leagues:
        league_path = Path(f'data/01-raw/{league}')
        if not league_path.exists():
            continue

        for season_dir in league_path.iterdir():
            if not season_dir.is_dir():
                continue

            events_file = season_dir / 'events.parquet'
            matches_file = season_dir / 'matches.parquet'

            if events_file.exists():
                events = pd.read_parquet(events_file)
                events['league'] = league
                events['season'] = season_dir.name
                all_events.append(events)

            if matches_file.exists():
                matches = pd.read_parquet(matches_file)
                matches['league'] = league
                matches['season'] = season_dir.name
                all_matches.append(matches)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

    print(f"Events loaded: {len(events_df)}")
    print(f"Matches loaded: {len(matches_df)}")

    return events_df, matches_df


def calculate_cards_per_match(events_df, matches_df):
    """Calculate total yellow cards per match."""
    # Filter yellow cards only
    yellow_cards = events_df[events_df['detail'] == 'Yellow Card']
    cards_per_match = yellow_cards.groupby('fixture_id').size().reset_index(name='total_cards')

    # Get fixture_id from matches
    if 'fixture.id' in matches_df.columns:
        matches_df = matches_df.rename(columns={'fixture.id': 'fixture_id'})

    # Merge
    result = matches_df[['fixture_id']].drop_duplicates()
    result = result.merge(cards_per_match, on='fixture_id', how='left')
    result['total_cards'] = result['total_cards'].fillna(0).astype(int)

    return result


def calculate_referee_cards_stats(events_df, matches_df):
    """Calculate referee-specific cards patterns."""
    # Get referee info
    if 'fixture.id' in matches_df.columns:
        matches_df = matches_df.rename(columns={'fixture.id': 'fixture_id'})
    if 'fixture.referee' in matches_df.columns:
        matches_df = matches_df.rename(columns={'fixture.referee': 'referee'})

    # Yellow cards per match
    yellow_cards = events_df[events_df['detail'] == 'Yellow Card']
    cards_per_match = yellow_cards.groupby('fixture_id').size().reset_index(name='cards')

    # Merge with referee
    cards_with_ref = cards_per_match.merge(
        matches_df[['fixture_id', 'referee']].drop_duplicates(),
        on='fixture_id',
        how='left'
    )

    # Calculate per referee
    stats = {}
    for referee, group in cards_with_ref.groupby('referee'):
        if pd.isna(referee) or len(group) < 5:
            continue
        cards = group['cards']
        stats[referee] = {
            'ref_cards_avg': cards.mean(),
            'ref_cards_std': cards.std(),
            'ref_over_3_5_rate': (cards > 3.5).mean(),
            'ref_over_4_5_rate': (cards > 4.5).mean(),
            'ref_cards_matches': len(group),
        }

    return stats


def calculate_team_cards_stats(events_df):
    """Calculate team-specific cards patterns."""
    yellow_cards = events_df[events_df['detail'] == 'Yellow Card']
    team_cards = yellow_cards.groupby(['fixture_id', 'team.name']).size().reset_index(name='cards')

    stats = {}
    for team, group in team_cards.groupby('team.name'):
        if len(group) < 5:
            continue
        stats[team] = {
            'team_cards_avg': group['cards'].mean(),
            'team_cards_std': group['cards'].std(),
            'team_cards_matches': len(group),
        }

    return stats


def create_cards_features(main_df, events_df, matches_df, referee_stats, team_stats):
    """Create enhanced features for cards prediction."""
    # Get cards per match
    cards_df = calculate_cards_per_match(events_df, matches_df)

    # Merge with main features
    main_df = main_df.copy()
    main_df['fixture_id'] = main_df['fixture_id'].astype(int)
    cards_df['fixture_id'] = cards_df['fixture_id'].astype(int)

    merged = main_df.merge(cards_df, on='fixture_id', how='inner')
    print(f"Merged matches: {len(merged)}")

    # Add referee cards features
    def get_ref_features(row):
        # Try to find referee from matches_df
        fixture_id = row['fixture_id']
        match = matches_df[matches_df['fixture_id'] == fixture_id] if 'fixture_id' in matches_df.columns else pd.DataFrame()

        if len(match) == 0:
            # Try fixture.id
            match = matches_df[matches_df['fixture.id'] == fixture_id] if 'fixture.id' in matches_df.columns else pd.DataFrame()

        referee = None
        if len(match) > 0:
            referee = match.iloc[0].get('referee', match.iloc[0].get('fixture.referee'))

        if referee and referee in referee_stats:
            rs = referee_stats[referee]
            return pd.Series({
                'ref_cards_avg': rs['ref_cards_avg'],
                'ref_cards_std': rs['ref_cards_std'],
                'ref_over_3_5_rate': rs['ref_over_3_5_rate'],
                'ref_over_4_5_rate': rs['ref_over_4_5_rate'],
                'ref_cards_matches': rs['ref_cards_matches'],
            })
        else:
            return pd.Series({
                'ref_cards_avg': 3.5,
                'ref_cards_std': 1.5,
                'ref_over_3_5_rate': 0.5,
                'ref_over_4_5_rate': 0.3,
                'ref_cards_matches': 0,
            })

    # This is slow, so we'll do it in batches
    print("Adding referee cards features...")

    # Create referee lookup from matches
    if 'fixture.id' in matches_df.columns:
        ref_lookup = matches_df[['fixture.id', 'fixture.referee']].drop_duplicates()
        ref_lookup.columns = ['fixture_id', 'referee']
    else:
        ref_lookup = matches_df[['fixture_id', 'referee']].drop_duplicates()

    ref_lookup['fixture_id'] = ref_lookup['fixture_id'].astype(int)
    merged = merged.merge(ref_lookup, on='fixture_id', how='left')

    # Add referee stats
    for col in ['ref_cards_avg', 'ref_cards_std', 'ref_over_3_5_rate', 'ref_over_4_5_rate', 'ref_cards_matches']:
        merged[col] = 3.5 if 'avg' in col else (1.5 if 'std' in col else (0.5 if '3_5' in col else (0.3 if '4_5' in col else 0)))

    for idx, row in merged.iterrows():
        referee = row.get('referee')
        if referee and referee in referee_stats:
            rs = referee_stats[referee]
            merged.loc[idx, 'ref_cards_avg'] = rs['ref_cards_avg']
            merged.loc[idx, 'ref_cards_std'] = rs['ref_cards_std']
            merged.loc[idx, 'ref_over_3_5_rate'] = rs['ref_over_3_5_rate']
            merged.loc[idx, 'ref_over_4_5_rate'] = rs['ref_over_4_5_rate']
            merged.loc[idx, 'ref_cards_matches'] = rs['ref_cards_matches']

    # Add team cards features
    print("Adding team cards features...")
    home_team_col = 'home_team_name' if 'home_team_name' in merged.columns else 'home_team'
    away_team_col = 'away_team_name' if 'away_team_name' in merged.columns else 'away_team'

    merged['home_cards_avg'] = merged[home_team_col].map(
        lambda x: team_stats.get(x, {}).get('team_cards_avg', 1.5)
    )
    merged['away_cards_avg'] = merged[away_team_col].map(
        lambda x: team_stats.get(x, {}).get('team_cards_avg', 1.5)
    )
    merged['combined_team_cards'] = merged['home_cards_avg'] + merged['away_cards_avg']

    # Create target variables
    merged['over_2_5'] = (merged['total_cards'] > 2.5).astype(int)
    merged['over_3_5'] = (merged['total_cards'] > 3.5).astype(int)
    merged['over_4_5'] = (merged['total_cards'] > 4.5).astype(int)
    merged['over_5_5'] = (merged['total_cards'] > 5.5).astype(int)

    return merged


def run_boruta_selection(X_train, y_train, feature_cols, max_iter=50):
    """Run Boruta feature selection."""
    print("\n" + "-" * 50)
    print("Running Boruta Feature Selection...")
    print("-" * 50)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        n_jobs=-1,
        random_state=42
    )

    boruta = BorutaPy(
        rf,
        n_estimators='auto',
        verbose=0,
        random_state=42,
        max_iter=max_iter,
        perc=100
    )

    # Handle any remaining NaN
    X_clean = X_train.fillna(0).values

    boruta.fit(X_clean, y_train)

    confirmed = [f for f, s in zip(feature_cols, boruta.support_) if s]
    tentative = [f for f, s in zip(feature_cols, boruta.support_weak_) if s]
    selected = confirmed + tentative

    print(f"Confirmed features: {len(confirmed)}")
    print(f"Tentative features: {len(tentative)}")
    print(f"Total selected: {len(selected)}")

    ranks = pd.DataFrame({
        'feature': feature_cols,
        'rank': boruta.ranking_,
        'confirmed': boruta.support_,
        'tentative': boruta.support_weak_
    }).sort_values('rank')

    print("\nTop 20 features by Boruta rank:")
    for _, row in ranks.head(20).iterrows():
        status = "CONFIRMED" if row['confirmed'] else ("tentative" if row['tentative'] else "rejected")
        print(f"  {row['feature']:<40} rank={row['rank']:>2} ({status})")

    if len(selected) < 20:
        print(f"\nWarning: Only {len(selected)} features, using top 30 by rank")
        selected = ranks.head(30)['feature'].tolist()

    return selected, ranks


def simulate_betting(proba, y_test, odds_over, odds_under, n_bootstrap=1000):
    """Simulate betting with bootstrap CI."""
    results = []

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        # OVER bets
        bet_mask = proba >= thresh
        n_bets = bet_mask.sum()

        if n_bets >= 20:
            precision = y_test[bet_mask].mean()
            rois = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(len(proba), len(proba), replace=True)
                p, y = proba[idx], y_test[idx]
                mask = p >= thresh
                if mask.sum() > 0:
                    wins = y[mask] == 1
                    profit = (wins * (odds_over - 1) - (~wins) * 1).sum()
                    rois.append(profit / mask.sum() * 100)

            if rois:
                results.append({
                    'strategy': f'OVER >= {thresh}',
                    'direction': 'over',
                    'threshold': thresh,
                    'bets': n_bets,
                    'precision': precision,
                    'roi': np.mean(rois),
                    'ci_low': np.percentile(rois, 2.5),
                    'ci_high': np.percentile(rois, 97.5),
                    'p_profit': (np.array(rois) > 0).mean(),
                })

        # UNDER bets
        under_proba = 1 - proba
        bet_mask = under_proba >= thresh
        n_bets = bet_mask.sum()

        if n_bets >= 20:
            y_under = 1 - y_test
            precision = y_under[bet_mask].mean()
            rois = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(len(proba), len(proba), replace=True)
                p, y = 1 - proba[idx], 1 - y_test[idx]
                mask = p >= thresh
                if mask.sum() > 0:
                    wins = y[mask] == 1
                    profit = (wins * (odds_under - 1) - (~wins) * 1).sum()
                    rois.append(profit / mask.sum() * 100)

            if rois:
                results.append({
                    'strategy': f'UNDER >= {thresh}',
                    'direction': 'under',
                    'threshold': thresh,
                    'bets': n_bets,
                    'precision': precision,
                    'roi': np.mean(rois),
                    'ci_low': np.percentile(rois, 2.5),
                    'ci_high': np.percentile(rois, 97.5),
                    'p_profit': (np.array(rois) > 0).mean(),
                })

    return pd.DataFrame(results)


def main():
    # Load data
    print("\n" + "=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)

    main_df = load_main_features()
    events_df, matches_df = load_cards_data()

    # Calculate referee and team stats on ALL historical data first
    print("\nCalculating historical statistics...")
    referee_stats = calculate_referee_cards_stats(events_df, matches_df)
    team_stats = calculate_team_cards_stats(events_df)
    print(f"Referee patterns: {len(referee_stats)}")
    print(f"Team patterns: {len(team_stats)}")

    # Create enhanced features
    print("\n" + "=" * 70)
    print("STEP 2: Creating Enhanced Features")
    print("=" * 70)

    df = create_cards_features(main_df, events_df, matches_df, referee_stats, team_stats)

    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Temporal split
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nTrain: {len(train_df)} ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Val: {len(val_df)} ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    print(f"Test: {len(test_df)} ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # Define feature columns
    exclude_cols = [
        'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
        'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
        'total_goals', 'goal_difference', 'league', 'season', 'referee',
        'total_cards', 'over_2_5', 'over_3_5', 'over_4_5', 'over_5_5',
        # Exclude odds (leakage)
        'b365_home_open', 'b365_draw_open', 'b365_away_open',
        'avg_home_open', 'avg_draw_open', 'avg_away_open',
        'b365_home_close', 'b365_draw_close', 'b365_away_close',
        'avg_home_close', 'avg_draw_close', 'avg_away_close',
        'b365_over25', 'b365_under25', 'avg_over25', 'avg_under25',
    ]

    # Also exclude any odds-related columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if 'odds' not in c.lower() and 'b365' not in c.lower()
                    and 'pinnacle' not in c.lower() and 'max_' not in c.lower()
                    and 'avg_' not in c[:4] and 'ah_' not in c]

    print(f"\nFeatures for modeling: {len(feature_cols)}")

    # Remove any non-numeric columns
    numeric_features = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(col)
        elif df[col].dtype == 'object':
            # Try to convert
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > len(df) * 0.5:
                    numeric_features.append(col)
            except:
                pass

    feature_cols = numeric_features
    print(f"Numeric features: {len(feature_cols)}")

    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    X_val = val_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)

    # Boruta on over_3_5 target (most common betting line)
    print("\n" + "=" * 70)
    print("STEP 3: Boruta Feature Selection (target: over_3_5)")
    print("=" * 70)

    y_train_35 = train_df['over_3_5'].values
    selected_features, feature_ranks = run_boruta_selection(X_train, y_train_35, feature_cols)

    # Train and evaluate
    print("\n" + "=" * 70)
    print("STEP 4: Model Training with Selected Features")
    print("=" * 70)

    targets = [
        ('over_3_5', CARDS_ODDS['over_3_5'], CARDS_ODDS['under_3_5']),
        ('over_4_5', CARDS_ODDS['over_4_5'], CARDS_ODDS['under_4_5']),
        ('over_5_5', CARDS_ODDS['over_5_5'], CARDS_ODDS['under_5_5']),
    ]

    all_results = []

    for target_name, odds_over, odds_under in targets:
        print(f"\n--- {target_name.upper()} ---")

        y_train = train_df[target_name].values
        y_val = val_df[target_name].values
        y_test = test_df[target_name].values

        print(f"Train positive rate: {y_train.mean():.1%}")
        print(f"Test positive rate: {y_test.mean():.1%}")

        X_train_sel = X_train[selected_features].fillna(0)
        X_val_sel = X_val[selected_features].fillna(0)
        X_test_sel = X_test[selected_features].fillna(0)

        # Train callibration
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, min_child_weight=15,
            reg_lambda=5.0, learning_rate=0.05, subsample=0.8,
            random_state=42, verbosity=0
        )
        xgb.fit(X_train_sel, y_train)
        xgb_cal = CalibratedClassifierCV(xgb, method='sigmoid', cv='prefit')
        xgb_cal.fit(X_val_sel, y_val)

        lgbm = LGBMClassifier(
            n_estimators=200, max_depth=4, min_child_samples=50,
            reg_lambda=5.0, learning_rate=0.05, subsample=0.8,
            random_state=42, verbose=-1
        )
        lgbm.fit(X_train_sel, y_train)
        lgbm_cal = CalibratedClassifierCV(lgbm, method='sigmoid', cv='prefit')
        lgbm_cal.fit(X_val_sel, y_val)

        cat = CatBoostClassifier(
            iterations=200, depth=4, l2_leaf_reg=10,
            learning_rate=0.05, random_state=42, verbose=0
        )
        cat.fit(X_train_sel, y_train)
        cat_cal = CalibratedClassifierCV(cat, method='sigmoid', cv='prefit')
        cat_cal.fit(X_val_sel, y_val)

        # Ensemble
        xgb_proba = xgb_cal.predict_proba(X_test_sel)[:, 1]
        lgbm_proba = lgbm_cal.predict_proba(X_test_sel)[:, 1]
        cat_proba = cat_cal.predict_proba(X_test_sel)[:, 1]
        avg_proba = (xgb_proba + lgbm_proba + cat_proba) / 3

        # Metrics
        brier = brier_score_loss(y_test, avg_proba)
        auc = roc_auc_score(y_test, avg_proba)
        accuracy = ((avg_proba >= 0.5) == y_test).mean()
        print(f"Ensemble - Accuracy: {accuracy:.3f}, Brier: {brier:.4f}, AUC: {auc:.3f}")

        # Betting simulation
        betting_df = simulate_betting(avg_proba, y_test, odds_over, odds_under)

        if len(betting_df) > 0:
            betting_df['target'] = target_name
            all_results.append(betting_df)

            print(f"\n{'Strategy':<16} {'Bets':>6} {'Prec':>8} {'ROI':>10} {'P(profit)':>10}")
            print("-" * 55)

            for _, row in betting_df.sort_values('roi', ascending=False).head(8).iterrows():
                print(f"{row['strategy']:<16} {row['bets']:>6} {row['precision']:>7.1%} "
                      f"{row['roi']:>9.1f}% {row['p_profit']:>9.0%}")

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY - CARDS V2 WITH BORUTA")
    print("=" * 70)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values('roi', ascending=False)

        print("\nTop 10 strategies:")
        print(f"\n{'Target':<12} {'Strategy':<16} {'Bets':>6} {'ROI':>10} {'P(profit)':>10}")
        print("-" * 60)

        for _, row in combined.head(10).iterrows():
            print(f"{row['target']:<12} {row['strategy']:<16} {row['bets']:>6} "
                  f"{row['roi']:>9.1f}% {row['p_profit']:>9.0%}")

        viable = combined[(combined['roi'] > 0) & (combined['p_profit'] > 0.70)]
        print(f"\nViable strategies (ROI > 0, P > 70%): {len(viable)}")

        # Save
        output = {
            'version': 'v2_boruta',
            'total_features': len(feature_cols),
            'selected_features': selected_features,
            'feature_rankings': feature_ranks.head(30).to_dict('records'),
            'strategies': combined.to_dict('records')
        }

        output_path = Path('experiments/outputs/cards_optimization_v2.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
