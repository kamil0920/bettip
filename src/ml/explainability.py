"""
Model Explainability

Generates explanations for betting model predictions:
- SHAP: summary plots, dependence plots, waterfall plots
- PDP: partial dependence diagnostics with monotonic constraint validation

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
    feature_perturbation: str = 'tree_path_dependent',
    background_samples: int = 200,
) -> Any:
    """Compute SHAP values for a model.

    Args:
        model: Fitted model (XGBoost, LightGBM, CatBoost, or sklearn).
        X: Feature matrix.
        feature_names: Feature names matching X columns.
        model_type: 'tree' for GBDT models, 'kernel' for generic models.
        max_samples: Max samples to explain (for speed).
        feature_perturbation: 'tree_path_dependent' (default, observational) or
            'interventional' (breaks feature correlations, better for
            multicollinear features like ELO/form/odds).
        background_samples: Number of background samples for interventional SHAP
            (ignored for tree_path_dependent). Default 200.

    Returns:
        shap.Explanation object.
    """
    import shap

    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X = X[indices]

    # Use CatBoost native SHAP when available (faster, exact)
    # Skip native CatBoost SHAP for interventional mode — native API only supports
    # tree_path_dependent. Use generic shap.TreeExplainer instead.
    use_native_catboost = (
        model_type == 'tree'
        and hasattr(model, 'get_feature_importance')
        and feature_perturbation != 'interventional'
    )
    if use_native_catboost:
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
        if feature_perturbation == 'interventional':
            # Interventional SHAP requires a background dataset to marginalize over
            n_bg = min(background_samples, len(X))
            bg_indices = np.random.RandomState(42).choice(len(X), n_bg, replace=False)
            background = X[bg_indices]
            try:
                explainer = shap.TreeExplainer(
                    model, data=background, feature_perturbation='interventional',
                )
                logger.info(
                    f"Using interventional SHAP with {n_bg} background samples"
                )
            except Exception as e:
                logger.warning(
                    f"Interventional SHAP failed ({e}), "
                    "falling back to tree_path_dependent"
                )
                explainer = shap.TreeExplainer(model)
        else:
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
    feature_perturbation: str = 'tree_path_dependent',
    background_samples: int = 200,
) -> Dict[str, Any]:
    """Full SHAP explanation pipeline: summary + dependence + waterfall.

    Args:
        model: Fitted model.
        X: Feature matrix (test set recommended).
        feature_names: Feature names.
        output_dir: Directory to save all plots.
        bet_type: Bet type name for titles.
        model_type: 'tree' or 'kernel'.
        feature_perturbation: 'tree_path_dependent' or 'interventional'.
        background_samples: Background samples for interventional SHAP.

    Returns:
        Dict with paths to all generated plots and SHAP summary statistics.
    """
    output_dir = str(Path(output_dir) / bet_type)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    shap_values = compute_shap_values(
        model, X, feature_names, model_type,
        feature_perturbation=feature_perturbation,
        background_samples=background_samples,
    )

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


def compute_pdp_diagnostics(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    monotonic_constraints: Optional[Dict[str, int]] = None,
    grid_resolution: int = 20,
    max_samples: int = 1000,
) -> Dict[str, Dict[str, Any]]:
    """Compute Partial Dependence Plot diagnostics for each feature.

    For each feature, computes the PDP curve and derives:
    - monotonicity_score: Spearman correlation between grid values and PDP values
      (+1 = perfectly monotone increasing, -1 = perfectly monotone decreasing)
    - constraint_match: whether the PDP direction matches the declared monotonic
      constraint (True/False), or None if no constraint is declared
    - pdp_range: range of PDP values (max - min), indicating feature effect magnitude

    Logs warnings for any constraint violations.

    Args:
        model: Fitted sklearn-compatible estimator with predict_proba.
        X: Feature matrix (n_samples, n_features).
        feature_names: Feature names matching X columns.
        monotonic_constraints: Optional dict of {feature_name: 1 or -1} from
            strategies.yaml. 1 = increasing, -1 = decreasing.
        grid_resolution: Number of grid points for PDP computation. Default 20.
        max_samples: Max samples to use for PDP computation (for speed).

    Returns:
        Dict of {feature_name: {"monotonicity_score": float,
                                "constraint_match": bool or None,
                                "pdp_range": float}}
    """
    from scipy.stats import spearmanr
    from sklearn.inspection import partial_dependence

    if monotonic_constraints is None:
        monotonic_constraints = {}

    # Subsample for speed
    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X = X[indices]

    results: Dict[str, Dict[str, Any]] = {}
    n_features = len(feature_names)

    logger.info(f"Computing PDP diagnostics for {n_features} features...")

    for i, feat_name in enumerate(feature_names):
        try:
            pdp_result = partial_dependence(
                model,
                X,
                features=[i],
                grid_resolution=grid_resolution,
                response_method='auto',
                method='brute',
                kind='average',
            )

            # pdp_result.average shape: (n_outputs, n_grid_points)
            # For binary classification, n_outputs=1 (probability of positive class)
            pdp_values = pdp_result['average'][0]
            grid_values = pdp_result['grid_values'][0]

            # Monotonicity score: Spearman rank correlation between grid and PDP
            if len(grid_values) >= 3:
                corr, _ = spearmanr(grid_values, pdp_values)
                monotonicity_score = float(corr) if np.isfinite(corr) else 0.0
            else:
                monotonicity_score = 0.0

            pdp_range = float(np.max(pdp_values) - np.min(pdp_values))

            # Check constraint match
            constraint_match: Optional[bool] = None
            declared = monotonic_constraints.get(feat_name)
            if declared is not None:
                if declared == 1:
                    constraint_match = monotonicity_score > 0
                elif declared == -1:
                    constraint_match = monotonicity_score < 0

                if constraint_match is False:
                    direction_word = "positive" if declared == 1 else "negative"
                    logger.warning(
                        f"PDP constraint violation: {feat_name} declared {direction_word} "
                        f"(constraint={declared}) but PDP monotonicity={monotonicity_score:+.3f}"
                    )

            results[feat_name] = {
                "monotonicity_score": monotonicity_score,
                "constraint_match": constraint_match,
                "pdp_range": pdp_range,
            }

        except Exception as e:
            logger.warning(f"PDP computation failed for {feat_name}: {e}")
            results[feat_name] = {
                "monotonicity_score": 0.0,
                "constraint_match": None,
                "pdp_range": 0.0,
            }

    # Summary log
    n_constrained = sum(1 for v in results.values() if v["constraint_match"] is not None)
    n_violations = sum(1 for v in results.values() if v["constraint_match"] is False)
    if n_constrained > 0:
        logger.info(
            f"PDP constraint check: {n_constrained - n_violations}/{n_constrained} "
            f"constraints satisfied, {n_violations} violations"
        )

    # Log top features by PDP range
    sorted_by_range = sorted(results.items(), key=lambda x: x[1]["pdp_range"], reverse=True)
    logger.info("Top features by PDP range (effect magnitude):")
    for feat_name, diag in sorted_by_range[:10]:
        constraint_str = ""
        if diag["constraint_match"] is not None:
            constraint_str = " MATCH" if diag["constraint_match"] else " VIOLATION"
        logger.info(
            f"  {feat_name:<45} range={diag['pdp_range']:.4f}  "
            f"mono={diag['monotonicity_score']:+.3f}{constraint_str}"
        )

    return results


def cluster_features(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Cluster correlated features using hierarchical clustering.

    Groups features by correlation distance (1 - |corr|), so features
    with |corr| > threshold end up in the same cluster. This addresses
    SHAP substitution effects where correlated features steal importance
    from each other (Lopez de Prado, Clustered FI, 2020).

    Args:
        X: Feature matrix (n_samples, n_features).
        feature_names: Feature names matching X columns.
        threshold: Distance threshold for cutting the dendrogram.
            Lower = more clusters (less grouping).
            0.5 means features with |corr| > 0.5 are grouped.

    Returns:
        Dict with:
            - clusters: Dict[int, List[str]] mapping cluster_id -> feature names
            - linkage_matrix: The scipy linkage matrix for visualization
            - n_clusters: Number of clusters formed
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    X_arr = np.asarray(X, dtype=float)
    n_features = X_arr.shape[1]

    if n_features <= 1:
        return {
            "clusters": {0: list(feature_names)},
            "linkage_matrix": None,
            "n_clusters": 1,
        }

    # Correlation distance: d = 1 - |corr|
    corr = np.corrcoef(X_arr, rowvar=False)
    # Handle NaN correlations (constant features)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    # Ensure symmetry and non-negative
    dist = np.clip((dist + dist.T) / 2, 0, 2)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=threshold, criterion="distance")

    clusters: Dict[int, List[str]] = {}
    for feat, label in zip(feature_names, labels):
        clusters.setdefault(int(label), []).append(feat)

    return {
        "clusters": clusters,
        "linkage_matrix": Z,
        "n_clusters": len(clusters),
    }


def clustered_feature_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cluster_threshold: float = 0.5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Clustered Feature Importance -- shuffles entire clusters together.

    Standard permutation importance and SHAP suffer from substitution
    effects: when feature A is shuffled, correlated feature B compensates,
    understating A's true importance. CFI fixes this by shuffling all
    features in a cluster simultaneously.

    Args:
        model: Fitted sklearn-compatible model with predict_proba.
        X: Feature matrix (n_samples, n_features).
        y: True labels.
        feature_names: Feature names matching X columns.
        cluster_threshold: Threshold for feature clustering (see cluster_features).
        n_repeats: Number of permutation repeats per cluster.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with:
            - clusters: Dict[int, List[str]] from cluster_features
            - cluster_importance: Dict[int, float] -- importance per cluster
            - feature_importance: Dict[str, float] -- importance distributed to features
            - n_clusters: Number of clusters
    """
    from sklearn.metrics import log_loss

    clustering = cluster_features(X, feature_names, threshold=cluster_threshold)
    clusters = clustering["clusters"]

    # Baseline score
    try:
        y_prob = model.predict_proba(X)
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            baseline_score = log_loss(y, y_prob[:, 1])
        else:
            baseline_score = log_loss(y, y_prob)
    except Exception:
        baseline_score = log_loss(y, model.predict_proba(X))

    rng = np.random.RandomState(random_state)
    cluster_importance: Dict[int, float] = {}

    for cluster_id, cluster_feats in clusters.items():
        feat_indices = [feature_names.index(f) for f in cluster_feats]
        losses = []

        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_order = rng.permutation(len(X))
            for idx in feat_indices:
                X_perm[:, idx] = X_perm[perm_order, idx]

            try:
                y_prob_perm = model.predict_proba(X_perm)
                if y_prob_perm.ndim == 2 and y_prob_perm.shape[1] == 2:
                    perm_score = log_loss(y, y_prob_perm[:, 1])
                else:
                    perm_score = log_loss(y, y_prob_perm)
            except Exception:
                perm_score = baseline_score

            losses.append(perm_score - baseline_score)

        cluster_importance[cluster_id] = float(np.mean(losses))

    # Distribute cluster importance proportionally to individual features
    # using within-cluster variance contribution
    feature_importance: Dict[str, float] = {}
    for cluster_id, cluster_feats in clusters.items():
        imp = cluster_importance[cluster_id]
        n_in_cluster = len(cluster_feats)
        # Equal distribution within cluster (simplest, most robust)
        per_feat = imp / n_in_cluster
        for feat in cluster_feats:
            feature_importance[feat] = per_feat

    # Sort by importance (descending)
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "clusters": clusters,
        "cluster_importance": cluster_importance,
        "feature_importance": feature_importance,
        "n_clusters": clustering["n_clusters"],
    }
