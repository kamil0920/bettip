"""Adversarial validation and filtering utilities for temporal leakage detection.

Shared module used by both sniper optimization and feature parameter optimization
pipelines. Detects and removes features that distinguish time periods (temporal leakage).
"""

import logging
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _adversarial_validation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
) -> Tuple[float, List[Tuple[str, float]]]:
    """Train LGB classifier to distinguish train vs test. AUC > 0.6 = distribution shift.

    Args:
        X_train: Training features.
        X_test: Test features.
        feature_names: Feature names for interpretability.

    Returns:
        Tuple of (auc, top_shifting_features) where top_shifting_features
        is a list of (feature_name, importance) tuples sorted by importance.
    """
    y_adv = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    X_adv = np.vstack([X_train, X_test])
    clf = lgb.LGBMClassifier(n_estimators=50, max_depth=3, verbose=-1, random_state=42)
    clf.fit(X_adv, y_adv)
    auc = roc_auc_score(y_adv, clf.predict_proba(X_adv)[:, 1])

    importances = dict(zip(feature_names, clf.feature_importances_))
    top_shift = sorted(importances.items(), key=lambda x: -x[1])[:10]
    return auc, top_shift


def _adversarial_filter(
    X: np.ndarray,
    feature_names: List[str],
    max_passes: int = 2,
    auc_threshold: float = 0.75,
    importance_threshold: float = 0.05,
    max_features_per_pass: int = 10,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Pre-screen and remove temporally leaky features before model training.

    Splits data into first 70% (train) and last 30% (test) to mimic temporal split,
    then uses adversarial validation to find features that distinguish time periods.
    Features with high importance are removed iteratively.

    Args:
        X: Feature matrix (n_samples, n_features), assumed sorted by time.
        feature_names: List of feature names corresponding to columns of X.
        max_passes: Maximum number of iterative removal passes.
        auc_threshold: Only filter if adversarial AUC exceeds this threshold.
        importance_threshold: Remove features with importance > this fraction of total.
        max_features_per_pass: Cap on features removed per pass.

    Returns:
        Tuple of (filtered_X, filtered_feature_names, diagnostics_dict).
    """
    diagnostics: Dict[str, Any] = {
        "passes": [],
        "initial_n_features": len(feature_names),
        "removed_features": [],
    }

    current_X = X.copy()
    current_features = list(feature_names)

    split_idx = int(len(current_X) * 0.7)

    for pass_num in range(max_passes):
        X_train = current_X[:split_idx]
        X_test = current_X[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        auc, top_shift = _adversarial_validation(X_train_scaled, X_test_scaled, current_features)

        pass_info: Dict[str, Any] = {"pass": pass_num, "auc": float(auc), "top_features": top_shift[:5]}

        if auc <= auc_threshold:
            pass_info["action"] = "stop_below_threshold"
            diagnostics["passes"].append(pass_info)
            logger.info(
                f"  Adversarial filter pass {pass_num}: AUC={auc:.3f} <= {auc_threshold} — stopping"
            )
            break

        # Calculate total importance and find leaky features
        total_importance = sum(imp for _, imp in top_shift)
        if total_importance == 0:
            pass_info["action"] = "stop_zero_importance"
            diagnostics["passes"].append(pass_info)
            break

        to_remove = []
        for feat_name, imp in top_shift:
            if imp / total_importance > importance_threshold:
                to_remove.append(feat_name)
            if len(to_remove) >= max_features_per_pass:
                break

        if not to_remove:
            pass_info["action"] = "stop_no_features_above_threshold"
            diagnostics["passes"].append(pass_info)
            logger.info(
                f"  Adversarial filter pass {pass_num}: AUC={auc:.3f} but no features above {importance_threshold:.0%} importance"
            )
            break

        # Safety: never remove so many features that fewer than 5 remain
        max_removable = len(current_features) - 5
        if max_removable <= 0:
            pass_info["action"] = "stop_too_few_features"
            diagnostics["passes"].append(pass_info)
            logger.info(
                f"  Adversarial filter pass {pass_num}: only {len(current_features)} features left, stopping"
            )
            break
        to_remove = to_remove[:max_removable]

        # Remove features
        keep_mask = [f not in to_remove for f in current_features]
        current_X = current_X[:, keep_mask]
        current_features = [f for f, keep in zip(current_features, keep_mask) if keep]
        diagnostics["removed_features"].extend(to_remove)

        pass_info["action"] = "removed"
        pass_info["removed"] = to_remove
        diagnostics["passes"].append(pass_info)

        logger.info(
            f"  Adversarial filter pass {pass_num}: AUC={auc:.3f}, removed {len(to_remove)} features: {to_remove}"
        )

        # If AUC still high after removal, do another pass (up to max)
        if pass_num < max_passes - 1 and auc > auc_threshold:
            continue
        else:
            break

    diagnostics["final_n_features"] = len(current_features)
    diagnostics["total_removed"] = len(diagnostics["removed_features"])

    return current_X, current_features, diagnostics
