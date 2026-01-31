"""
Model Compilation for Production Inference Speedup

Uses Treelite to compile XGBoost/LightGBM models into optimized
C code for faster prediction. Typical speedup: 10-100x.

Note: Only the model prediction is compiled. Feature preprocessing
must run separately before calling the compiled model.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compile_model(
    model: Any,
    output_dir: str,
    model_name: str = 'compiled_model',
) -> str:
    """Compile an XGBoost or LightGBM model with Treelite.

    Args:
        model: Fitted XGBoost or LightGBM model.
        output_dir: Directory to save compiled model.
        model_name: Name for the compiled artifact.

    Returns:
        Path to the compiled shared library.
    """
    import treelite

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_type = type(model).__name__

    if 'XGB' in model_type:
        # Save XGBoost model to JSON, then load into treelite
        json_path = str(output_dir / f'{model_name}.json')
        model.save_model(json_path)
        tl_model = treelite.frontend.load_xgboost_model(json_path)
    elif 'LGBM' in model_type:
        # Save LightGBM model to text, then load into treelite
        txt_path = str(output_dir / f'{model_name}.txt')
        model.booster_.save_model(txt_path)
        tl_model = treelite.frontend.load_lightgbm_model(txt_path)
    else:
        raise ValueError(f"Unsupported model type for compilation: {model_type}. "
                        f"Only XGBoost and LightGBM are supported.")

    # Compile to shared library
    lib_path = str(output_dir / f'{model_name}.so')
    tl_model.export_lib(
        toolchain='gcc',
        libpath=lib_path,
        verbose=False,
    )

    logger.info(f"Compiled {model_type} model to {lib_path}")
    return lib_path


class CompiledPredictor:
    """Fast predictor using a Treelite-compiled model.

    Use for production inference where speed matters.
    Predictions match the original model exactly.
    """

    def __init__(self, lib_path: str):
        """
        Args:
            lib_path: Path to compiled .so library from compile_model().
        """
        import treelite.runtime

        self.predictor = treelite.runtime.Predictor(lib_path)
        self.lib_path = lib_path
        logger.info(f"Loaded compiled predictor from {lib_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from compiled model.

        Args:
            X: Feature matrix (must match training feature order).

        Returns:
            Predictions array.
        """
        import treelite.runtime

        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float32)
        elif X.dtype != np.float32:
            X = X.astype(np.float32)

        batch = treelite.runtime.DMatrix(X)
        return self.predictor.predict(batch)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions for classification.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with class probabilities.
        """
        raw_preds = self.predict(X)

        # For binary classification, raw_preds may be 1D (positive class prob)
        if raw_preds.ndim == 1:
            return np.column_stack([1 - raw_preds, raw_preds])
        return raw_preds


def verify_compilation(
    original_model: Any,
    compiled_predictor: CompiledPredictor,
    X_test: np.ndarray,
    tolerance: float = 1e-5,
) -> bool:
    """Verify compiled model matches original predictions.

    Args:
        original_model: Original fitted model.
        compiled_predictor: Compiled predictor.
        X_test: Test features.
        tolerance: Maximum allowed difference.

    Returns:
        True if predictions match within tolerance.
    """
    original_preds = original_model.predict_proba(X_test)
    compiled_preds = compiled_predictor.predict_proba(X_test)

    max_diff = np.max(np.abs(original_preds - compiled_preds))
    matches = max_diff < tolerance

    if matches:
        logger.info(f"Compilation verified: max_diff={max_diff:.2e} < {tolerance}")
    else:
        logger.warning(f"Compilation mismatch: max_diff={max_diff:.2e} > {tolerance}")

    return matches
