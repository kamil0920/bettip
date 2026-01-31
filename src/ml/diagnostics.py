"""
Model Diagnostics: Learning Curves and Bias-Variance Analysis

Generates learning curves per bet type to diagnose:
- Overfitting: large train-val gap
- Underfitting: both scores high (loss) / low (accuracy)
- More-data-helps: val score still improving at max training size

Results logged to MLflow as artifacts.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, learning_curve

logger = logging.getLogger(__name__)


def generate_learning_curves(
    model,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str = 'neg_log_loss',
    cv_folds: int = 3,
    n_points: int = 8,
    min_train_fraction: float = 0.2,
) -> Dict[str, Any]:
    """Generate learning curves for bias-variance diagnosis.

    Args:
        model: Unfitted sklearn-compatible estimator.
        X: Feature matrix.
        y: Target array.
        scoring: Scoring metric (e.g., 'neg_log_loss', 'f1', 'neg_mean_absolute_error').
        cv_folds: Number of time-series CV folds.
        n_points: Number of training size points to evaluate.
        min_train_fraction: Minimum training fraction (of available train data).

    Returns:
        Dict with keys:
        - train_sizes: actual training sizes used
        - train_scores_mean/std: mean/std of training scores per size
        - val_scores_mean/std: mean/std of validation scores per size
        - diagnosis: string diagnosis ('overfit', 'underfit', 'more_data', 'good')
        - gap: train-val score gap at final point
    """
    cv = TimeSeriesSplit(n_splits=cv_folds)

    train_sizes_frac = np.linspace(min_train_fraction, 1.0, n_points)

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes_frac,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        shuffle=False,  # Preserve time ordering
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # Diagnose
    gap = abs(train_mean[-1] - val_mean[-1])
    val_improving = val_mean[-1] > val_mean[-2] if len(val_mean) > 1 else False

    # For neg_log_loss: scores are negative, closer to 0 = better
    # Gap > 0.1 in absolute terms suggests overfitting
    if gap > 0.1:
        diagnosis = 'overfit'
    elif val_improving and gap < 0.05:
        diagnosis = 'more_data'
    elif abs(val_mean[-1]) > 0.5 and scoring == 'neg_log_loss':
        diagnosis = 'underfit'
    else:
        diagnosis = 'good'

    logger.info(f"Learning curve diagnosis: {diagnosis} "
                f"(train={train_mean[-1]:.4f}, val={val_mean[-1]:.4f}, gap={gap:.4f})")

    return {
        'train_sizes': train_sizes.tolist(),
        'train_scores_mean': train_mean.tolist(),
        'train_scores_std': train_std.tolist(),
        'val_scores_mean': val_mean.tolist(),
        'val_scores_std': val_std.tolist(),
        'diagnosis': diagnosis,
        'gap': float(gap),
        'scoring': scoring,
    }


def generate_learning_curve_plot(
    results: Dict[str, Any],
    title: str = 'Learning Curve',
    output_path: Optional[str] = None,
) -> Optional[Any]:
    """Generate a matplotlib learning curve plot.

    Args:
        results: Output from generate_learning_curves().
        title: Plot title.
        output_path: If provided, saves plot to this path.

    Returns:
        matplotlib Figure object, or None if matplotlib unavailable.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return None

    train_sizes = results['train_sizes']
    train_mean = np.array(results['train_scores_mean'])
    train_std = np.array(results['train_scores_std'])
    val_mean = np.array(results['val_scores_mean'])
    val_std = np.array(results['val_scores_std'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{title} [diagnosis: {results['diagnosis']}]")
    ax.set_xlabel("Training Size")
    ax.set_ylabel(results['scoring'])

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation score')

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Learning curve plot saved to {output_path}")

    plt.close(fig)
    return fig


def diagnose_all_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    scoring: str = 'neg_log_loss',
    output_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """Run learning curve diagnostics for multiple models.

    Args:
        models: Dict of model_name -> unfitted estimator.
        X: Feature matrix.
        y: Target array.
        scoring: Scoring metric.
        output_dir: Optional directory to save plots.

    Returns:
        Dict of model_name -> learning curve results.
    """
    from pathlib import Path

    results = {}
    for name, model in models.items():
        logger.info(f"Generating learning curve for {name}")
        lc = generate_learning_curves(model, X, y, scoring=scoring)
        results[name] = lc

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            plot_path = str(Path(output_dir) / f"learning_curve_{name}.png")
            generate_learning_curve_plot(lc, title=f"{name} Learning Curve", output_path=plot_path)

    return results
