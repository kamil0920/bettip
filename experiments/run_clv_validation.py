#!/usr/bin/env python
"""
CLV Validation Script

This script validates your model's predictions against closing line odds
from football-data.co.uk. This is the gold standard test for betting edge.

If your model consistently beats the closing line (positive CLV),
you likely have real predictive edge. If not, your backtested
profits are likely due to overfitting or luck.

Usage:
    python experiments/run_clv_validation.py

What it does:
1. Loads your feature data with predictions
2. Loads historical closing odds from football-data.co.uk
3. Calculates CLV for each prediction
4. Reports whether your model beats the market

Key interpretation:
    - Avg CLV > +2%: Strong evidence of edge
    - Avg CLV +0% to +2%: Promising, need more data
    - Avg CLV -2% to 0%: No clear edge
    - Avg CLV < -2%: Market is beating your model
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from src.odds.football_data_loader import FootballDataLoader, normalize_team_name


def load_feature_data(path: str = "data/03-features/features_all_5leagues_with_odds.csv") -> pd.DataFrame:
    """Load feature data with model predictions."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_closing_odds(leagues: List[str], seasons: List[int]) -> pd.DataFrame:
    """Load historical closing odds from football-data.co.uk."""
    loader = FootballDataLoader(cache_dir=Path("data/odds-cache"))

    all_odds = []
    for league in leagues:
        try:
            df = loader.load_multiple_seasons(league, seasons)
            if not df.empty:
                all_odds.append(df)
                print(f"  Loaded {len(df)} matches for {league}")
        except Exception as e:
            print(f"  Warning: Could not load {league}: {e}")

    if not all_odds:
        return pd.DataFrame()

    combined = pd.concat(all_odds, ignore_index=True)
    print(f"\nTotal matches with closing odds: {len(combined)}")

    return combined


def match_predictions_to_closing_odds(
    features_df: pd.DataFrame,
    odds_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match predictions from features data to closing odds.

    We simulate what would happen if we bet at the opening odds
    and measure CLV against closing odds.
    """
    # Normalize dates
    features_df['match_date'] = pd.to_datetime(features_df['date']).dt.date
    odds_df['match_date'] = pd.to_datetime(odds_df['date']).dt.date

    # Normalize team names in odds data
    odds_df['home_team_norm'] = odds_df['home_team'].apply(normalize_team_name)
    odds_df['away_team_norm'] = odds_df['away_team'].apply(normalize_team_name)

    results = []

    for _, row in features_df.iterrows():
        # Try to find matching odds data
        home = row.get('home_team_name', '')
        away = row.get('away_team_name', '')
        match_date = row['match_date']

        # Search for match in odds data
        match = odds_df[
            (odds_df['match_date'] == match_date) &
            (
                (odds_df['home_team_norm'].str.contains(home.split()[0] if home else 'NOMATCH', case=False, na=False)) |
                (odds_df['home_team'].str.contains(home.split()[0] if home else 'NOMATCH', case=False, na=False))
            )
        ]

        if match.empty or len(match) > 1:
            # Try matching by away team
            match = odds_df[
                (odds_df['match_date'] == match_date) &
                (
                    (odds_df['away_team_norm'].str.contains(away.split()[0] if away else 'NOMATCH', case=False, na=False)) |
                    (odds_df['away_team'].str.contains(away.split()[0] if away else 'NOMATCH', case=False, na=False))
                )
            ]

        if match.empty or len(match) > 1:
            continue

        match = match.iloc[0]

        # Get odds at prediction time (opening) and closing
        result = {
            'fixture_id': row.get('fixture_id'),
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'league': row.get('league', ''),

            # Actual result
            'home_goals': row.get('home_goals'),
            'away_goals': row.get('away_goals'),
            'home_win_actual': 1 if row.get('home_win') else 0,
            'away_win_actual': 1 if row.get('away_win') else 0,
            'draw_actual': 1 if row.get('draw') else 0,

            # Opening odds (what we'd bet at)
            'home_odds_open': row.get('avg_home_open', match.get('avg_home_open')),
            'away_odds_open': row.get('avg_away_open', match.get('avg_away_open')),
            'draw_odds_open': row.get('avg_draw_open', match.get('avg_draw_open')),

            # Closing odds (market consensus)
            'home_odds_close': match.get('avg_home_close'),
            'away_odds_close': match.get('avg_away_close'),
            'draw_odds_close': match.get('avg_draw_close'),

            # Asian Handicap
            'ah_line_open': row.get('ah_line'),
            'ah_line_close': match.get('ah_line_close'),
            'ah_home_odds_open': row.get('avg_ah_home'),
            'ah_away_odds_open': row.get('avg_ah_away'),
            'ah_home_odds_close': match.get('avg_ah_home_close'),
            'ah_away_odds_close': match.get('avg_ah_away_close'),

            # Over/Under 2.5
            'over25_odds_open': row.get('avg_over25'),
            'under25_odds_open': row.get('avg_under25'),
            'over25_odds_close': match.get('avg_over25_close'),
            'under25_odds_close': match.get('avg_under25_close'),

            # Model predictions (if available)
            'poisson_home_win_prob': row.get('poisson_home_win_prob'),
            'poisson_away_win_prob': row.get('poisson_away_win_prob'),
            'elo_home_win_prob': row.get('home_win_prob_elo'),
            'elo_away_win_prob': row.get('away_win_prob_elo'),
        }

        results.append(result)

    return pd.DataFrame(results)


def calculate_clv(open_odds: float, close_odds: float) -> float:
    """
    Calculate Closing Line Value (bettor-friendly formula).

    CLV = (your_odds / closing_odds) - 1

    Positive CLV = you got BETTER odds than closing (good!)
    Negative CLV = you got WORSE odds than closing (bad)

    Example:
        open=2.50, close=2.20 → CLV = +13.6% (odds dropped, you got value)
        open=2.50, close=2.80 → CLV = -10.7% (odds rose, missed value)
    """
    if pd.isna(open_odds) or pd.isna(close_odds):
        return np.nan
    if open_odds <= 1 or close_odds <= 1:
        return np.nan

    # Bettor-friendly: positive when your odds > closing odds
    return (open_odds / close_odds) - 1


def analyze_clv_by_bet_type(df: pd.DataFrame) -> Dict:
    """Analyze CLV for different bet types."""
    results = {}

    # Away Win CLV
    away_mask = df['away_odds_open'].notna() & df['away_odds_close'].notna()
    if away_mask.sum() > 0:
        df.loc[away_mask, 'clv_away'] = df.loc[away_mask].apply(
            lambda r: calculate_clv(r['away_odds_open'], r['away_odds_close']), axis=1
        )
        away_clv = df.loc[away_mask, 'clv_away'].dropna()
        if len(away_clv) > 0:
            results['away_win'] = {
                'n_bets': len(away_clv),
                'avg_clv': away_clv.mean() * 100,
                'median_clv': away_clv.median() * 100,
                'positive_rate': (away_clv > 0).mean() * 100,
            }

    # Home Win CLV
    home_mask = df['home_odds_open'].notna() & df['home_odds_close'].notna()
    if home_mask.sum() > 0:
        df.loc[home_mask, 'clv_home'] = df.loc[home_mask].apply(
            lambda r: calculate_clv(r['home_odds_open'], r['home_odds_close']), axis=1
        )
        home_clv = df.loc[home_mask, 'clv_home'].dropna()
        if len(home_clv) > 0:
            results['home_win'] = {
                'n_bets': len(home_clv),
                'avg_clv': home_clv.mean() * 100,
                'median_clv': home_clv.median() * 100,
                'positive_rate': (home_clv > 0).mean() * 100,
            }

    # Asian Handicap CLV
    ah_mask = df['ah_away_odds_open'].notna() & df['ah_away_odds_close'].notna()
    if ah_mask.sum() > 0:
        df.loc[ah_mask, 'clv_ah'] = df.loc[ah_mask].apply(
            lambda r: calculate_clv(r['ah_away_odds_open'], r['ah_away_odds_close']), axis=1
        )
        ah_clv = df.loc[ah_mask, 'clv_ah'].dropna()
        if len(ah_clv) > 0:
            results['asian_handicap'] = {
                'n_bets': len(ah_clv),
                'avg_clv': ah_clv.mean() * 100,
                'median_clv': ah_clv.median() * 100,
                'positive_rate': (ah_clv > 0).mean() * 100,
            }

    # Over 2.5 CLV
    over_mask = df['over25_odds_open'].notna() & df['over25_odds_close'].notna()
    if over_mask.sum() > 0:
        df.loc[over_mask, 'clv_over25'] = df.loc[over_mask].apply(
            lambda r: calculate_clv(r['over25_odds_open'], r['over25_odds_close']), axis=1
        )
        over_clv = df.loc[over_mask, 'clv_over25'].dropna()
        if len(over_clv) > 0:
            results['over_25'] = {
                'n_bets': len(over_clv),
                'avg_clv': over_clv.mean() * 100,
                'median_clv': over_clv.median() * 100,
                'positive_rate': (over_clv > 0).mean() * 100,
            }

    return results


def analyze_value_betting_clv(df: pd.DataFrame, prob_col: str, odds_open_col: str, odds_close_col: str) -> Dict:
    """
    Analyze CLV for value bets only (where model probability > market implied).

    This tests whether your model's "value" calls actually beat the close.
    """
    mask = (
        df[prob_col].notna() &
        df[odds_open_col].notna() &
        df[odds_close_col].notna()
    )

    if mask.sum() == 0:
        return {}

    subset = df.loc[mask].copy()

    # Calculate implied probability from opening odds
    subset['market_implied'] = 1 / subset[odds_open_col]

    # Value bets: where our model prob > market implied
    subset['is_value_bet'] = subset[prob_col] > subset['market_implied']

    # Calculate CLV
    subset['clv'] = subset.apply(
        lambda r: calculate_clv(r[odds_open_col], r[odds_close_col]), axis=1
    )

    value_bets = subset[subset['is_value_bet']]
    non_value_bets = subset[~subset['is_value_bet']]

    results = {
        'total_matches': len(subset),
        'value_bets': len(value_bets),
        'non_value_bets': len(non_value_bets),
    }

    if len(value_bets) > 0:
        results['value_bet_clv'] = {
            'avg_clv': value_bets['clv'].mean() * 100,
            'median_clv': value_bets['clv'].median() * 100,
            'positive_rate': (value_bets['clv'] > 0).mean() * 100,
        }

    if len(non_value_bets) > 0:
        results['non_value_bet_clv'] = {
            'avg_clv': non_value_bets['clv'].mean() * 100,
            'median_clv': non_value_bets['clv'].median() * 100,
            'positive_rate': (non_value_bets['clv'] > 0).mean() * 100,
        }

    return results


def print_clv_report(results: Dict, title: str = "CLV Analysis Results") -> None:
    """Print formatted CLV report."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for bet_type, stats in results.items():
        print(f"\n{bet_type.upper().replace('_', ' ')}")
        print("-" * 40)
        print(f"  Number of bets:    {stats['n_bets']:,}")
        print(f"  Average CLV:       {stats['avg_clv']:+.2f}%")
        print(f"  Median CLV:        {stats['median_clv']:+.2f}%")
        print(f"  Positive CLV %:    {stats['positive_rate']:.1f}%")

        # Interpretation
        avg = stats['avg_clv']
        if avg > 2:
            print(f"  Verdict:           STRONG EDGE (beats market by {avg:.1f}%)")
        elif avg > 0:
            print(f"  Verdict:           PROMISING (slight positive CLV)")
        elif avg > -2:
            print(f"  Verdict:           NEUTRAL (no clear edge)")
        else:
            print(f"  Verdict:           WARNING (market beats your model)")


def main():
    print("=" * 70)
    print("CLV VALIDATION - Testing if Your Model Beats the Closing Line")
    print("=" * 70)

    # Load feature data
    print("\n1. Loading feature data...")
    try:
        features_df = load_feature_data()
        print(f"   Loaded {len(features_df)} matches")
    except FileNotFoundError:
        print("   ERROR: Feature data not found. Run feature engineering first.")
        return

    # Load closing odds
    print("\n2. Loading closing odds from football-data.co.uk...")
    leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
    seasons = [2022, 2023, 2024, 2025]

    odds_df = load_closing_odds(leagues, seasons)

    if odds_df.empty:
        print("   ERROR: Could not load closing odds data.")
        return

    # Match predictions to closing odds
    print("\n3. Matching predictions to closing odds...")
    matched_df = match_predictions_to_closing_odds(features_df, odds_df)
    print(f"   Matched {len(matched_df)} predictions to closing odds")

    if len(matched_df) == 0:
        print("   ERROR: No matches found. Check team name mapping.")
        return

    # Analyze CLV by bet type
    print("\n4. Calculating CLV for each bet type...")
    clv_results = analyze_clv_by_bet_type(matched_df)

    # Print results
    print_clv_report(clv_results, "CLV BY BET TYPE (All Matches)")

    # Analyze value betting CLV specifically
    print("\n" + "=" * 70)
    print("CLV FOR VALUE BETS ONLY")
    print("(Where model probability > market implied probability)")
    print("=" * 70)

    # Away win value betting
    if 'elo_away_win_prob' in matched_df.columns:
        value_results = analyze_value_betting_clv(
            matched_df,
            'elo_away_win_prob',
            'away_odds_open',
            'away_odds_close'
        )
        if value_results:
            print(f"\nAWAY WIN (using ELO probability)")
            print(f"  Total matches:     {value_results['total_matches']}")
            print(f"  Value bets found:  {value_results['value_bets']}")
            if 'value_bet_clv' in value_results:
                vb = value_results['value_bet_clv']
                print(f"  Value bet avg CLV: {vb['avg_clv']:+.2f}%")
                print(f"  Value bet pos %:   {vb['positive_rate']:.1f}%")
            if 'non_value_bet_clv' in value_results:
                nvb = value_results['non_value_bet_clv']
                print(f"  Non-value avg CLV: {nvb['avg_clv']:+.2f}%")

    # Save detailed results
    output_path = Path("experiments/outputs/clv_validation_results.csv")
    matched_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if clv_results:
        avg_clvs = [r['avg_clv'] for r in clv_results.values()]
        overall_avg = np.mean(avg_clvs)

        print(f"\nOverall Average CLV: {overall_avg:+.2f}%")

        if overall_avg > 2:
            print("\nCONCLUSION: Your model appears to have REAL EDGE.")
            print("You consistently beat the closing line across bet types.")
            print("This suggests your backtested results may be achievable live.")
        elif overall_avg > 0:
            print("\nCONCLUSION: PROMISING but inconclusive.")
            print("Slight positive CLV, but could be noise.")
            print("Recommend: Continue paper trading for 200+ more bets.")
        elif overall_avg > -2:
            print("\nCONCLUSION: NO CLEAR EDGE detected.")
            print("Your model performs similarly to the market.")
            print("Backtested profits may be due to luck or overfitting.")
        else:
            print("\nCONCLUSION: WARNING - Market beats your model.")
            print("Negative CLV suggests your predictions are worse than closing.")
            print("Re-evaluate your feature engineering and model selection.")
    else:
        print("\nCould not calculate CLV. Check data quality.")


if __name__ == "__main__":
    main()
