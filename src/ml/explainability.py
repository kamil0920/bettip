"""
SHAP Model Explainability

Generates SHAP-based explanations for betting model predictions:
- Summary plots: global feature importance with direction
- Dependence plots: feature effect on prediction
- Waterfall plots: individual prediction explanations

All plots can be logged to MLflow as artifacts.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    model_type: str = 'tree',
    max_samples: int = 1000,
) -> Any:
    """Compute SHAP values for a model.

    Args:
        model: Fitted model (XGBoost, LightGBM, CatBoost, or sklearn).
        X: Feature matrix.
        feature_names: Feature names matching X columns.
        model_type: 'tree' for GBDT models, 'kernel' for generic models.
        max_samples: Max samples to explain (for speed).

    Returns:
        shap.Explanation object.
    """
    import shap

    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X = X[indices]

    # Use CatBoost native SHAP when available (faster, exact)
    if model_type == 'tree' and hasattr(model, 'get_feature_importance'):
        try:
            from catboost import Pool
            pool = Pool(X)
            shap_vals_raw = model.get_feature_importance(type='ShapValues', data=pool)
            # CatBoost returns (n_samples, n_features+1) — last col is bias
            import shap as shap_module
            shap_values = shap_module.Explanation(
                values=shap_vals_raw[:, :-1],
                base_values=shap_vals_raw[:, -1],
                data=X,
                feature_names=feature_names,
            )
            logger.info("Used CatBoost native SHAP (exact)")
            return shap_values
        except Exception:
            pass  # Fall through to standard SHAP

    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer(X)

    # Attach feature names
    shap_values.feature_names = feature_names

    return shap_values


def generate_summary_plot(
    shap_values: Any,
    output_path: str,
    max_features: int = 20,
    title: Optional[str] = None,
) -> str:
    """Generate SHAP beeswarm summary plot.

    Args:
        shap_values: SHAP Explanation object.
        output_path: Path to save PNG.
        max_features: Max features to show.
        title: Optional plot title.

    Returns:
        Path to saved plot.
    """
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=max_features, show=False)
    if title:
        plt.title(title)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    logger.info(f"SHAP summary plot saved to {output_path}")
    return output_path


def generate_dependence_plots(
    shap_values: Any,
    feature_names: List[str],
    output_dir: str,
    top_n: int = 10,
) -> List[str]:
    """Generate SHAP dependence plots for top features.

    Args:
        shap_values: SHAP Explanation object.
        feature_names: All feature names.
        output_dir: Directory to save plots.
        top_n: Number of top features to plot.

    Returns:
        List of paths to saved plots.
    """
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get top features by mean absolute SHAP value
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs[:, 1]  # Class 1 for binary classification
    top_indices = np.argsort(mean_abs)[-top_n:][::-1]

    paths = []
    for idx in top_indices:
        feat_name = feature_names[idx]
        fig, ax = plt.subplots(figsize=(8, 5))

        sv = shap_values
        if sv.values.ndim == 3:
            # Binary classification: use class 1
            import copy
            sv = copy.deepcopy(sv)
            sv.values = sv.values[:, :, 1]

        shap.plots.scatter(sv[:, idx], show=False)
        plt.title(f"SHAP Dependence: {feat_name}")

        path = str(Path(output_dir) / f"shap_dep_{feat_name}.png")
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        paths.append(path)

    logger.info(f"Generated {len(paths)} dependence plots in {output_dir}")
    return paths


def generate_waterfall_plot(
    shap_values: Any,
    sample_idx: int,
    output_path: str,
    title: Optional[str] = None,
) -> str:
    """Generate SHAP waterfall plot for a single prediction.

    Args:
        shap_values: SHAP Explanation object.
        sample_idx: Index of sample to explain.
        output_path: Path to save PNG.
        title: Optional plot title.

    Returns:
        Path to saved plot.
    """
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sv = shap_values[sample_idx]
    if sv.values.ndim > 1:
        # Binary: take class 1
        import copy
        sv = copy.deepcopy(sv)
        sv.values = sv.values[:, 1]
        sv.base_values = sv.base_values[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(sv, show=False)
    if title:
        plt.title(title)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    logger.info(f"SHAP waterfall plot saved to {output_path}")
    return output_path


def explain_model(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    bet_type: str = '',
    model_type: str = 'tree',
) -> Dict[str, Any]:
    """Full SHAP explanation pipeline: summary + dependence + waterfall.

    Args:
        model: Fitted model.
        X: Feature matrix (test set recommended).
        feature_names: Feature names.
        output_dir: Directory to save all plots.
        bet_type: Bet type name for titles.
        model_type: 'tree' or 'kernel'.

    Returns:
        Dict with paths to all generated plots and SHAP summary statistics.
    """
    output_dir = str(Path(output_dir) / bet_type)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    shap_values = compute_shap_values(model, X, feature_names, model_type)

    # Summary
    summary_path = generate_summary_plot(
        shap_values,
        output_path=str(Path(output_dir) / 'shap_summary.png'),
        title=f'{bet_type} SHAP Summary',
    )

    # Dependence
    dep_paths = generate_dependence_plots(
        shap_values, feature_names,
        output_dir=str(Path(output_dir) / 'dependence'),
    )

    # Waterfall for first sample
    waterfall_path = generate_waterfall_plot(
        shap_values, sample_idx=0,
        output_path=str(Path(output_dir) / 'shap_waterfall_sample0.png'),
        title=f'{bet_type} Prediction Explanation (sample 0)',
    )

    # Feature importance ranking
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs[:, 1]
    importance = dict(sorted(
        zip(feature_names, mean_abs.tolist()),
        key=lambda x: x[1], reverse=True,
    ))

    return {
        'summary_plot': summary_path,
        'dependence_plots': dep_paths,
        'waterfall_plot': waterfall_path,
        'feature_importance': dict(list(importance.items())[:20]),
    }
