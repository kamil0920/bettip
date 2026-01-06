"""
Paper Trading System for Football Betting

This system allows you to:
1. Load trained models and make predictions on upcoming matches
2. Track a simulated bankroll
3. Log all predictions with timestamps
4. Evaluate performance over time

Usage:
    # Initialize
    trader = PaperTrader(initial_bankroll=1000, bet_fraction=0.02)

    # Load models (train first if needed)
    trader.load_models()

    # Make prediction on a match
    prediction = trader.predict(match_features)

    # Record a bet
    trader.place_bet(match_id, prediction, odds)

    # After match is over, record result
    trader.record_result(match_id, actual_outcome)

    # View history
    trader.get_history()
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class PaperTrader:
    """Paper trading system for testing betting strategies without real money."""

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        bet_fraction: float = 0.02,
        probability_threshold: float = 0.45,
        output_dir: str = 'experiments/outputs/paper_trading'
    ):
        """
        Initialize paper trading system.

        Args:
            initial_bankroll: Starting bankroll amount
            bet_fraction: Fraction of initial bankroll to bet per match
            probability_threshold: Minimum probability to place a bet
            output_dir: Directory to save trading logs
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.bet_fraction = bet_fraction
        self.probability_threshold = probability_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.output_dir / 'trade_history.json'
        self.load_history()

        self.models = {}
        self.scalers = {}
        self.features_per_model = {}
        self.meta_model = None

    def load_history(self):
        """Load existing trade history."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.trades = data.get('trades', [])
                self.current_bankroll = data.get('current_bankroll', self.initial_bankroll)
        else:
            self.trades = []

    def save_history(self):
        """Save trade history to file."""
        data = {
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.current_bankroll,
            'bet_fraction': self.bet_fraction,
            'probability_threshold': self.probability_threshold,
            'trades': self.trades,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_models(self, models_dir: str = 'experiments/outputs'):
        """
        Load trained models from saved files.

        This expects:
        - optuna_best_params.json
        - features_per_model.json
        - Or pickled model files
        """
        models_path = Path(models_dir)

        features_file = models_path / 'features_per_model.json'
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.features_per_model = json.load(f)
            print(f"✓ Loaded feature configurations for {len(self.features_per_model)} models")

        params_file = models_path / 'optuna_best_params.json'
        if params_file.exists():
            with open(params_file, 'r') as f:
                self.best_params = json.load(f)
            print(f"✓ Loaded optimized parameters")

        models_pickle = models_path / 'trained_models.pkl'
        if models_pickle.exists():
            with open(models_pickle, 'rb') as f:
                saved = pickle.load(f)
                self.models = saved['models']
                self.scalers = saved.get('scalers', {})
                self.meta_model = saved.get('meta_model')
            print(f"✓ Loaded {len(self.models)} trained models")
        else:
            print("⚠ No pickled models found. Will need to train models before predictions.")

    def train_and_save_models(self, data_path: str = 'data/03-features/features_all_5leagues_with_odds.csv'):
        """Train models and save them for future use."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier

        print("Training models...")

        df = pd.read_csv(data_path)
        df_with_odds = df[df['avg_away_open'].notna() & (df['avg_away_open'] > 1)].copy()

        exclude_cols = [
            'fixture_id', 'date', 'home_team_id', 'home_team_name', 'away_team_id',
            'away_team_name', 'round', 'match_result', 'home_win', 'draw', 'away_win',
            'total_goals', 'goal_difference', 'league',
            'b365_home_open', 'b365_draw_open', 'b365_away_open',
            'avg_home_open', 'avg_draw_open', 'avg_away_open',
            'b365_home_close', 'b365_draw_close', 'b365_away_close',
            'avg_home_close', 'avg_draw_close', 'avg_away_close'
        ]

        feature_cols = [c for c in df_with_odds.columns if c not in exclude_cols]
        feature_cols = [c for c in feature_cols if df_with_odds[c].notna().sum() > len(df_with_odds) * 0.5]

        X = df_with_odds[feature_cols].copy()
        y = df_with_odds['away_win'].values
        dates = pd.to_datetime(df_with_odds['date'])

        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        sorted_indices = dates.argsort()
        n = len(X)
        train_idx = sorted_indices[:int(0.7*n)]
        val_idx = sorted_indices[int(0.7*n):int(0.85*n)]

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        with open('experiments/outputs/optuna_best_params.json', 'r') as f:
            best_params = json.load(f)

        with open('experiments/outputs/features_per_model.json', 'r') as f:
            features_per_model = json.load(f)

        self.features_per_model = features_per_model

        models = {}
        scalers = {}

        xgb_features = [f for f in features_per_model['XGBoost'] if f in X_train.columns]
        xgb_params = {**best_params['XGBoost'], 'random_state': 42, 'verbosity': 0}
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train[xgb_features], y_train)
        xgb_cal = CalibratedClassifierCV(xgb, method='sigmoid', cv='prefit')
        xgb_cal.fit(X_val[xgb_features], y_val)
        models['XGBoost'] = (xgb_cal, xgb_features)

        lgbm_features = [f for f in features_per_model['LightGBM'] if f in X_train.columns]
        lgbm_params = {**best_params['LightGBM'], 'random_state': 42, 'verbose': -1}
        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_train[lgbm_features], y_train)
        lgbm_cal = CalibratedClassifierCV(lgbm, method='sigmoid', cv='prefit')
        lgbm_cal.fit(X_val[lgbm_features], y_val)
        models['LightGBM'] = (lgbm_cal, lgbm_features)

        cat_features = [f for f in features_per_model['CatBoost'] if f in X_train.columns]
        cat_params = {**best_params['CatBoost'], 'random_state': 42, 'verbose': 0}
        cat = CatBoostClassifier(**cat_params)
        cat.fit(X_train[cat_features], y_train)
        cat_cal = CalibratedClassifierCV(cat, method='sigmoid', cv='prefit')
        cat_cal.fit(X_val[cat_features], y_val)
        models['CatBoost'] = (cat_cal, cat_features)

        lr_features = [f for f in features_per_model['LogisticReg'] if f in X_train.columns]
        scaler_lr = StandardScaler()
        X_train_lr = scaler_lr.fit_transform(X_train[lr_features])
        X_val_lr = scaler_lr.transform(X_val[lr_features])
        lr_params = {**best_params['LogisticReg'], 'solver': 'saga', 'max_iter': 1000, 'random_state': 42}
        lr = LogisticRegression(**lr_params)
        lr.fit(X_train_lr, y_train)
        lr_cal = CalibratedClassifierCV(lr, method='sigmoid', cv='prefit')
        lr_cal.fit(X_val_lr, y_val)
        models['LogisticReg'] = (lr_cal, lr_features)
        scalers['LogisticReg'] = scaler_lr

        val_preds = {}
        for name, (model, features) in models.items():
            if name == 'LogisticReg':
                X_v = scalers[name].transform(X_val[features])
            else:
                X_v = X_val[features]
            val_preds[name] = model.predict_proba(X_v)[:, 1]

        X_stack = np.column_stack([val_preds[n] for n in ['XGBoost', 'LightGBM', 'CatBoost', 'LogisticReg']])
        meta = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0])
        meta.fit(X_stack, y_val)

        self.models = models
        self.scalers = scalers
        self.meta_model = meta

        save_path = Path('experiments/outputs/trained_models.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({
                'models': models,
                'scalers': scalers,
                'meta_model': meta,
                'features_per_model': features_per_model
            }, f)

        print(f"✓ Models trained and saved to {save_path}")

    def predict(self, match_features: pd.DataFrame) -> Dict:
        """
        Make prediction for a match.

        Args:
            match_features: DataFrame with single row of features

        Returns:
            Dictionary with prediction details
        """
        if not self.models:
            raise ValueError("Models not loaded. Call load_models() first.")

        preds = {}
        for name, (model, features) in self.models.items():
            available_features = [f for f in features if f in match_features.columns]
            X = match_features[available_features]

            for col in available_features:
                if X[col].isna().any():
                    X[col] = X[col].fillna(0)

            if name == 'LogisticReg' and 'LogisticReg' in self.scalers:
                X = self.scalers[name].transform(X)

            preds[name] = float(model.predict_proba(X)[:, 1][0])

        X_stack = np.array([[preds[n] for n in ['XGBoost', 'LightGBM', 'CatBoost', 'LogisticReg']]])
        ensemble_prob = float(1 / (1 + np.exp(-self.meta_model.decision_function(X_stack)[0])))

        should_bet = ensemble_prob >= self.probability_threshold

        return {
            'individual_probs': preds,
            'ensemble_prob': ensemble_prob,
            'should_bet': should_bet,
            'bet_type': 'away_win',
            'confidence': 'high' if ensemble_prob >= 0.55 else 'medium' if ensemble_prob >= 0.50 else 'low'
        }

    def place_bet(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        match_date: str,
        prediction: Dict,
        odds: float,
        league: str = None
    ) -> Dict:
        """
        Place a paper bet on a match.

        Returns:
            Bet details
        """
        if not prediction['should_bet']:
            return {'status': 'skipped', 'reason': 'Probability below threshold'}

        bet_amount = self.initial_bankroll * self.bet_fraction

        bet = {
            'id': len(self.trades) + 1,
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'match_date': match_date,
            'league': league,
            'bet_type': 'away_win',
            'probability': prediction['ensemble_prob'],
            'odds': odds,
            'bet_amount': bet_amount,
            'potential_profit': bet_amount * (odds - 1),
            'status': 'pending',
            'placed_at': datetime.now().isoformat(),
            'result': None,
            'profit': None
        }

        self.trades.append(bet)
        self.save_history()

        print(f"✓ Bet placed: {away_team} to win @ {odds}")
        print(f"  Amount: ${bet_amount:.2f} | Potential: ${bet['potential_profit']:.2f}")

        return bet

    def record_result(self, match_id: str, away_win: bool) -> Dict:
        """
        Record the actual result of a match.

        Args:
            match_id: Match identifier
            away_win: Whether away team won
        """
        for trade in self.trades:
            if trade['match_id'] == match_id and trade['status'] == 'pending':
                trade['status'] = 'settled'
                trade['result'] = 'won' if away_win else 'lost'
                trade['settled_at'] = datetime.now().isoformat()

                if away_win:
                    trade['profit'] = trade['bet_amount'] * (trade['odds'] - 1)
                else:
                    trade['profit'] = -trade['bet_amount']

                self.current_bankroll += trade['profit']
                self.save_history()

                print(f"{'✓' if away_win else '✗'} Bet {trade['result']}: {trade['away_team']}")
                print(f"  Profit: ${trade['profit']:+.2f} | Bankroll: ${self.current_bankroll:.2f}")

                return trade

        print(f"⚠ No pending bet found for match {match_id}")
        return None

    def get_summary(self) -> Dict:
        """Get trading summary statistics."""
        settled = [t for t in self.trades if t['status'] == 'settled']
        pending = [t for t in self.trades if t['status'] == 'pending']

        if not settled:
            return {
                'total_trades': len(self.trades),
                'pending': len(pending),
                'settled': 0,
                'current_bankroll': self.current_bankroll,
                'roi': 0,
                'bankroll_change': 0,
                'wins': 0,
                'win_rate': 0,
                'total_staked': 0,
                'total_profit': 0
            }

        wins = sum(1 for t in settled if t['result'] == 'won')
        total_profit = sum(t['profit'] for t in settled)
        total_staked = sum(t['bet_amount'] for t in settled)

        return {
            'total_trades': len(self.trades),
            'pending': len(pending),
            'settled': len(settled),
            'wins': wins,
            'win_rate': wins / len(settled) * 100,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': total_profit / total_staked * 100 if total_staked > 0 else 0,
            'current_bankroll': self.current_bankroll,
            'bankroll_change': (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100
        }

    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()

        print("\n" + "=" * 50)
        print("PAPER TRADING SUMMARY")
        print("=" * 50)
        print(f"Initial Bankroll: ${self.initial_bankroll:,.2f}")
        print(f"Current Bankroll: ${summary['current_bankroll']:,.2f}")
        print(f"Change: {summary['bankroll_change']:+.1f}%")
        print("-" * 50)
        print(f"Total Bets: {summary['total_trades']}")
        print(f"  Settled: {summary['settled']}")
        print(f"  Pending: {summary['pending']}")

        if summary['settled'] > 0:
            print(f"Win Rate: {summary['win_rate']:.1f}%")
            print(f"ROI: {summary['roi']:+.1f}%")
            print(f"Total Profit: ${summary['total_profit']:+,.2f}")
        print("=" * 50)

    def get_pending_bets(self) -> List[Dict]:
        """Get all pending bets."""
        return [t for t in self.trades if t['status'] == 'pending']

    def get_history(self, limit: int = 10) -> pd.DataFrame:
        """Get recent trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        df = pd.DataFrame(self.trades[-limit:])
        cols = ['id', 'match_date', 'home_team', 'away_team', 'probability', 'odds',
                'bet_amount', 'status', 'result', 'profit']
        return df[[c for c in cols if c in df.columns]]


if __name__ == '__main__':
    print("=" * 60)
    print("PAPER TRADING SYSTEM - Demo")
    print("=" * 60)

    trader = PaperTrader(
        initial_bankroll=1000,
        bet_fraction=0.02,
        probability_threshold=0.45
    )

    try:
        trader.load_models()
        if not trader.models:
            print("\nTraining models...")
            trader.train_and_save_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Training new models...")
        trader.train_and_save_models()

    trader.print_summary()

    print("\n✅ Paper trading system ready!")
    print("   Use trader.predict(features) to make predictions")
    print("   Use trader.place_bet(...) to record bets")
    print("   Use trader.record_result(...) to settle bets")
