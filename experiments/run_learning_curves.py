#!/usr/bin/env python3
"""
Learning Curves Analysis

Plots training and validation scores as a function of training set size
to determine if more data would improve model performance.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, TimeSeriesSplit
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_FILE = "features_all.csv"
TARGET = "match_result"
RANDOM_STATE = 42

from sklearn.preprocessing import LabelEncoder


def load_data(features_file: str, target: str):
    """Load features and prepare data."""
    features_path = PROJECT_ROOT / "data" / "03-features" / features_file
    df = pd.read_csv(features_path)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    exclude_cols = ['date', 'season', 'round', 'home_team', 'away_team',
                    'fixture_id', 'match_result', 'home_win', 'draw', 'away_win',
                    'ft_home', 'ft_away', 'ht_home', 'ht_away', 'goal_difference',
                    'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name',
                    'league']

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y_raw = df[target].copy()

    X = X.fillna(0)

    print(f"Original unique classes in y: {sorted(y_raw.unique())}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Class mapping applied: {mapping}")
    print(f"Transformed unique classes in y: {sorted(np.unique(y))}")
    # --------------------------------

    print(f"Loaded {len(df)} samples, {len(feature_cols)} features")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return X, y

def get_models():
    """Get models for learning curve analysis with tuned parameters."""
    return {
        'CatBoost': CatBoostClassifier(
            iterations=422,
            depth=7,
            learning_rate=0.005068769327332112,
            bootstrap_type='Bernoulli',
            subsample=0.8373132486168499,
            colsample_bylevel=0.7820921248963657,
            l2_leaf_reg=5.6726173707934124e-05,
            random_strength=0.014910984617510477,
            verbose=0,
            random_state=RANDOM_STATE
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=148,
            max_depth=5,
            learning_rate=0.04825433596997611,
            subsample=0.9918596719520923,
            colsample_bytree=0.5410970656824947,
            min_child_samples=57,
            reg_alpha=0.004482070610918831,
            reg_lambda=0.008848390773057588,
            num_leaves=56,
            verbose=-1,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=88,
            max_depth=2,
            learning_rate=0.04039698665424471,
            subsample=0.7107558296260341,
            colsample_bytree=0.7960331793389167,
            min_child_weight=13,
            reg_alpha=0.0005689860088474209,
            reg_lambda=5.307052099351332e-08,
            verbosity=0,
            random_state=RANDOM_STATE,
            n_jobs=1
        )
    }


def plot_learning_curves(X, y, models, output_dir: Path):
    """Generate and plot learning curves for all models."""

    train_sizes = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    cv = TimeSeriesSplit(n_splits=3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    results = {}

    for idx, (name, model) in enumerate(models.items()):
        print(f"\nComputing learning curve for {name}...")

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            shuffle=False
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        results[name] = {
            'train_sizes': train_sizes_abs,
            'train_mean': train_mean,
            'train_std': train_std,
            'val_mean': val_mean,
            'val_std': val_std
        }

        ax = axes[idx]
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training score')
        ax.plot(train_sizes_abs, val_mean, 'o-', color='orange', label='Validation score')

        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{name} Learning Curve')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        final_train = train_mean[-1]
        final_val = val_mean[-1]
        gap = final_train - final_val

        ax.text(train_sizes_abs[-1], val_mean[-1] - 0.05,
                f'Final: {final_val:.1%}\nGap: {gap:.1%}',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.3)

    output_file = output_dir / 'learning_curves.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved learning curves to {output_file}")

    try:
        plt.show()
    except Exception:
        pass

    return results


def analyze_results(results):
    """Analyze learning curve results and provide recommendations."""
    print("\n" + "="*60)
    print("LEARNING CURVE ANALYSIS")
    print("="*60)

    for name, data in results.items():
        train_mean = data['train_mean']
        val_mean = data['val_mean']

        mid_idx = len(val_mean) // 2
        val_improvement = val_mean[-1] - val_mean[mid_idx]
        train_val_gap = train_mean[-1] - val_mean[-1]

        print(f"\n{name}:")
        print(f"  Final training accuracy:   {train_mean[-1]:.2%}")
        print(f"  Final validation accuracy: {val_mean[-1]:.2%}")
        print(f"  Train-val gap:             {train_val_gap:.2%}")
        print(f"  Recent improvement:        {val_improvement:+.2%}")

        if val_improvement > 0.005:
            print(f"  → Curves still RISING - more data would likely help")
        elif train_val_gap > 0.10:
            print(f"  → Large GAP - overfitting, try regularization")
        elif train_val_gap < 0.03 and val_mean[-1] < 0.60:
            print(f"  → Low scores, small gap - underfitting, need better features")
        else:
            print(f"  → Curves PLATEAUED - more data unlikely to help much")

    print("\n" + "="*60)
    print("RECOMMENDATION:")

    avg_improvement = np.mean([r['val_mean'][-1] - r['val_mean'][len(r['val_mean'])//2] for r in results.values()])
    avg_gap = np.mean([r['train_mean'][-1] - r['val_mean'][-1] for r in results.values()])

    if avg_improvement > 0.005:
        print("Models are still improving with more data.")
        print("Consider: Collecting more historical seasons or more leagues.")
    elif avg_gap > 0.08:
        print("Models are overfitting.")
        print("Consider: Stronger regularization, feature selection, or simpler models.")
    else:
        print("Models have plateaued.")
        print("Consider: Better features, ensemble methods, or different algorithms.")
    print("="*60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate learning curves')
    parser.add_argument('--config', default='config/local.yaml', help='Config file path')
    parser.add_argument('--features', default=FEATURES_FILE, help='Features file name')
    parser.add_argument('--output-dir', default='outputs/analysis', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y = load_data(args.features, TARGET)

    print("\nPreparing models...")
    models = get_models()

    print("\nGenerating learning curves (this may take a few minutes)...")
    results = plot_learning_curves(X, y, models, output_dir)

    analyze_results(results)


if __name__ == "__main__":
    main()
