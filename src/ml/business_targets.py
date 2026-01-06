"""
Business-focused betting targets for ML models.

Creates targets for:
1. Value Bet Detection - when model probability > implied odds probability
2. Profitable Bet - whether betting on each outcome returns profit
3. Expected Value - positive EV detection for each outcome
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_implied_probability(odds: pd.Series) -> pd.Series:
    """Convert decimal odds to implied probability."""
    return 1.0 / odds


def calculate_profit(odds: pd.Series, won: pd.Series, stake: float = 1.0) -> pd.Series:
    """
    Calculate profit/loss for a bet.

    Args:
        odds: Decimal odds for the bet
        won: Boolean series indicating if bet won
        stake: Stake amount (default 1.0 for unit stake)

    Returns:
        Profit/loss series (positive = profit, negative = loss)
    """
    return np.where(won, (odds - 1) * stake, -stake)


def calculate_expected_value(probability: pd.Series, odds: pd.Series) -> pd.Series:
    """
    Calculate expected value of a bet.

    EV = (probability * (odds - 1)) - (1 - probability)
       = probability * odds - 1

    Args:
        probability: Model's predicted probability of outcome
        odds: Decimal odds offered

    Returns:
        Expected value (positive = profitable in long run)
    """
    return probability * odds - 1


def create_business_targets(
    df: pd.DataFrame,
    home_odds_col: str = 'b365_home_close',
    draw_odds_col: str = 'b365_draw_close',
    away_odds_col: str = 'b365_away_close',
    result_col: str = 'match_result',
    model_probs: Optional[Dict[str, pd.Series]] = None
) -> pd.DataFrame:
    """
    Create business-focused betting targets.

    Args:
        df: DataFrame with match data and odds
        home_odds_col: Column name for home odds
        draw_odds_col: Column name for draw odds
        away_odds_col: Column name for away odds
        result_col: Column with match result (1=home, 0=draw, -1=away)
        model_probs: Optional dict with model probabilities for value bet detection

    Returns:
        DataFrame with additional business target columns
    """
    result = df.copy()

    odds_cols = [home_odds_col, draw_odds_col, away_odds_col]
    missing_cols = [c for c in odds_cols if c not in df.columns]

    if missing_cols:
        print(f"Warning: Missing odds columns: {missing_cols}")
        print("Business targets will have NaN for rows without odds")

    home_odds = df.get(home_odds_col, pd.Series([np.nan] * len(df)))
    draw_odds = df.get(draw_odds_col, pd.Series([np.nan] * len(df)))
    away_odds = df.get(away_odds_col, pd.Series([np.nan] * len(df)))

    if result_col in df.columns:
        home_won = df[result_col] == 1
        draw = df[result_col] == 0
        away_won = df[result_col] == -1
    else:
        home_won = df.get('home_win', pd.Series([False] * len(df))) == 1
        draw = df.get('draw', pd.Series([False] * len(df))) == 1
        away_won = df.get('away_win', pd.Series([False] * len(df))) == 1

    result['implied_prob_home'] = calculate_implied_probability(home_odds)
    result['implied_prob_draw'] = calculate_implied_probability(draw_odds)
    result['implied_prob_away'] = calculate_implied_probability(away_odds)

    total_prob = result['implied_prob_home'] + result['implied_prob_draw'] + result['implied_prob_away']
    result['fair_prob_home'] = result['implied_prob_home'] / total_prob
    result['fair_prob_draw'] = result['implied_prob_draw'] / total_prob
    result['fair_prob_away'] = result['implied_prob_away'] / total_prob

    result['profit_bet_home'] = calculate_profit(home_odds, home_won)
    result['profit_bet_draw'] = calculate_profit(draw_odds, draw)
    result['profit_bet_away'] = calculate_profit(away_odds, away_won)

    result['profitable_home'] = (result['profit_bet_home'] > 0).astype(int)
    result['profitable_draw'] = (result['profit_bet_draw'] > 0).astype(int)
    result['profitable_away'] = (result['profit_bet_away'] > 0).astype(int)

    profits = result[['profit_bet_home', 'profit_bet_draw', 'profit_bet_away']]
    result['best_bet'] = profits.idxmax(axis=1).map({
        'profit_bet_home': 1,
        'profit_bet_draw': 0,
        'profit_bet_away': -1
    })

    result['value_margin_home'] = result['fair_prob_home'] - result['implied_prob_home']
    result['value_margin_draw'] = result['fair_prob_draw'] - result['implied_prob_draw']
    result['value_margin_away'] = result['fair_prob_away'] - result['implied_prob_away']

    result['ev_home'] = calculate_expected_value(result['fair_prob_home'], home_odds)
    result['ev_draw'] = calculate_expected_value(result['fair_prob_draw'], draw_odds)
    result['ev_away'] = calculate_expected_value(result['fair_prob_away'], away_odds)

    result['positive_ev_home'] = (result['ev_home'] > 0).astype(int)
    result['positive_ev_draw'] = (result['ev_draw'] > 0).astype(int)
    result['positive_ev_away'] = (result['ev_away'] > 0).astype(int)

    result['has_positive_ev'] = (
        (result['ev_home'] > 0) |
        (result['ev_draw'] > 0) |
        (result['ev_away'] > 0)
    ).astype(int)

    evs = result[['ev_home', 'ev_draw', 'ev_away']]
    result['best_ev_outcome'] = evs.idxmax(axis=1).map({
        'ev_home': 1,
        'ev_draw': 0,
        'ev_away': -1
    })
    result['best_ev_value'] = evs.max(axis=1)

    result['ev_and_profitable_home'] = (
        (result['ev_home'] > 0) & (result['profit_bet_home'] > 0)
    ).astype(int)
    result['ev_and_profitable_draw'] = (
        (result['ev_draw'] > 0) & (result['profit_bet_draw'] > 0)
    ).astype(int)
    result['ev_and_profitable_away'] = (
        (result['ev_away'] > 0) & (result['profit_bet_away'] > 0)
    ).astype(int)

    if model_probs:
        for name, probs in model_probs.items():
            if 'home' in name:
                result[f'value_bet_{name}'] = (probs > result['implied_prob_home']).astype(int)
                result[f'model_ev_{name}'] = calculate_expected_value(probs, home_odds)
            elif 'draw' in name:
                result[f'value_bet_{name}'] = (probs > result['implied_prob_draw']).astype(int)
                result[f'model_ev_{name}'] = calculate_expected_value(probs, draw_odds)
            elif 'away' in name:
                result[f'value_bet_{name}'] = (probs > result['implied_prob_away']).astype(int)
                result[f'model_ev_{name}'] = calculate_expected_value(probs, away_odds)

    return result


def get_business_target_info() -> Dict[str, Dict]:
    """Get information about available business targets."""
    return {
        'profitable_home': {
            'description': 'Binary: was betting on home win profitable?',
            'type': 'binary',
            'metric': 'f1'
        },
        'profitable_draw': {
            'description': 'Binary: was betting on draw profitable?',
            'type': 'binary',
            'metric': 'f1'
        },
        'profitable_away': {
            'description': 'Binary: was betting on away win profitable?',
            'type': 'binary',
            'metric': 'f1'
        },
        'positive_ev_home': {
            'description': 'Binary: does home bet have positive expected value?',
            'type': 'binary',
            'metric': 'f1'
        },
        'positive_ev_draw': {
            'description': 'Binary: does draw bet have positive expected value?',
            'type': 'binary',
            'metric': 'f1'
        },
        'positive_ev_away': {
            'description': 'Binary: does away bet have positive expected value?',
            'type': 'binary',
            'metric': 'f1'
        },
        'has_positive_ev': {
            'description': 'Binary: is there any positive EV bet?',
            'type': 'binary',
            'metric': 'f1'
        },
        'best_bet': {
            'description': 'Which outcome was most profitable (1=home, 0=draw, -1=away)',
            'type': 'multiclass',
            'metric': 'f1_macro'
        },
        'best_ev_outcome': {
            'description': 'Which outcome has best EV (1=home, 0=draw, -1=away)',
            'type': 'multiclass',
            'metric': 'f1_macro'
        }
    }


def evaluate_betting_strategy(
    df: pd.DataFrame,
    predictions: pd.Series,
    target: str = 'profitable_home',
    stake: float = 1.0
) -> Dict[str, float]:
    """
    Evaluate a betting strategy based on model predictions.

    Args:
        df: DataFrame with business targets and odds
        predictions: Model predictions (1=bet, 0=no bet)
        target: Which target we're predicting
        stake: Stake per bet

    Returns:
        Dictionary with business metrics
    """
    if 'home' in target:
        profit_col = 'profit_bet_home'
    elif 'draw' in target:
        profit_col = 'profit_bet_draw'
    elif 'away' in target:
        profit_col = 'profit_bet_away'
    else:
        profit_col = 'profit_bet_home'

    valid_mask = df[profit_col].notna() & (predictions == 1)

    if valid_mask.sum() == 0:
        return {
            'total_bets': 0,
            'total_profit': 0,
            'roi': 0,
            'win_rate': 0,
            'avg_odds': 0
        }

    bets = df[valid_mask]
    profits = bets[profit_col]

    total_bets = len(bets)
    total_staked = total_bets * stake
    total_profit = profits.sum()
    wins = (profits > 0).sum()

    return {
        'total_bets': total_bets,
        'total_staked': total_staked,
        'total_profit': total_profit,
        'roi': (total_profit / total_staked * 100) if total_staked > 0 else 0,
        'win_rate': (wins / total_bets * 100) if total_bets > 0 else 0,
        'avg_profit_per_bet': total_profit / total_bets if total_bets > 0 else 0
    }
