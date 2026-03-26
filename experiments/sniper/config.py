"""SniperOptimizer configuration dataclass with validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SniperConfig:
    """All parameters for a sniper optimization run.

    Grouped logically for clarity. Validated via validate() before use.
    """

    # Required
    bet_type: str = ""

    # Cross-validation
    n_folds: int = 5
    n_holdout_folds: int = 1
    cv_method: str = "walk_forward"
    embargo_days: int = 14
    temporal_buffer: int = 50

    # Feature selection
    n_rfe_features: int = 100
    auto_rfe: bool = False
    min_rfe_features: int = 20
    max_rfe_features: int = 40
    mrmr_k: int = 0
    mrmr_min: int = 0    # Minimum K for mRMR search (0 = disabled, use fixed mrmr_k)
    mrmr_step: int = 5   # Step between K values in search
    rfe_step: int = 10
    boruta_prefilter: bool = True
    boruta_max_iter: int = 100
    boruta_importance: str = "shap"  # 'shap' (ARFS Leshy) or 'native' (BorutaPy gain)
    rfecv_scoring: str = "roc_auc"  # 'roc_auc' (discrimination) or 'neg_log_loss' (calibration)

    # Hyperparameter tuning
    n_optuna_trials: int = 250
    min_bets: int = 50
    seed: int = 42
    deterministic: bool = False
    fast_mode: bool = False

    # Model selection
    only_catboost: bool = False
    no_catboost: bool = False
    no_fastai: bool = False
    use_two_stage: Optional[bool] = None

    # CatBoost advanced
    use_monotonic: bool = False
    use_transfer_learning: bool = False
    use_baseline: bool = False

    # Sample weighting
    use_sample_weights: bool = False
    sample_decay_rate: Optional[float] = None

    # Odds / thresholds
    use_odds_threshold: bool = False
    threshold_alpha: float = 0.0
    filter_missing_odds: bool = True
    max_threshold: Optional[float] = None

    # Calibration
    calibration_method: str = "beta"
    max_ece: float = 0.15
    cal_window: float = 0.0  # 0 = default 20%, >0 = use this fraction for calibration split
    importance_weighted_calibration: bool = False  # Use density ratios for post-hoc calibration weighting

    # Tracking signal (directional bias)
    max_ts: float = 4.0  # Hard gate: skip configs with |TS| > this (0 = disable)
    ts_penalty_threshold: float = 2.0  # Soft penalty ramps from here
    ts_penalty_scale: float = 2.0  # Penalty = (|TS| - threshold) / scale, capped at 1.0

    # Adversarial validation
    adversarial_filter: bool = False
    adversarial_max_passes: int = 2
    adversarial_max_features: int = 10
    adversarial_auc_threshold: float = 0.75
    no_aggressive_reg: bool = False
    aggressive_reg_auc_threshold: float = 0.8
    whitelist_features: Optional[List[str]] = None

    # Feature parameters
    feature_params_path: Optional[str] = None
    optimize_features: bool = False
    n_feature_trials: int = 20
    feature_params_dir: Optional[Path] = None

    # Analysis
    run_walkforward: bool = False
    run_shap: bool = False
    shap_threshold_pct: float = 0.01

    # Misc
    merge_params_path: Optional[str] = None
    pe_gate: float = 1.0
    exclude_leagues: Optional[List[str]] = None
    tax_rate: float = 0.0
    training_window_days: int = 0  # 0 = use all data

    # Holdout evaluation
    min_holdout_bets: int = 10  # Minimum holdout bets before triggering threshold fallback

    # Edge-based threshold for niche markets
    edge_threshold_mode: bool = False  # Use ML edge over NegBin as threshold for estimated-odds markets

    # Niche market data quality filter mode
    # "blocklist" = remove entire leagues from strategies.yaml blocklist
    # "nan_filter" = row-level NaN removal for raw stat columns
    # "hybrid" = blocklist first, then NaN filter on remaining rows
    niche_filter_mode: str = "hybrid"

    # Tunable parameters (previously hardcoded)
    embargo_multiplier: float = 3.5  # days-per-match multiplier for embargo calc
    embargo_buffer: int = 7  # safety buffer days added to embargo
    tl_base_iterations: int = 200  # transfer learning base model iterations
    calibration_methods: Optional[List[str]] = None  # Optuna calibration search space
    cluster_threshold: float = 0.5  # distance threshold for feature clustering

    def validate(self) -> List[str]:
        """Validate config for conflicting or invalid parameters.

        Returns:
            List of error messages (empty = valid).
        """
        errors = []

        if not self.bet_type:
            errors.append("bet_type is required")

        if self.min_rfe_features > self.max_rfe_features:
            errors.append(
                f"min_rfe_features ({self.min_rfe_features}) must be "
                f"<= max_rfe_features ({self.max_rfe_features})"
            )

        if self.n_holdout_folds >= self.n_folds - 1:
            errors.append(
                f"n_holdout_folds ({self.n_holdout_folds}) must be "
                f"< n_folds - 1 ({self.n_folds - 1})"
            )

        if self.only_catboost and self.no_catboost:
            errors.append("Cannot use both --only-catboost and --no-catboost")

        if not 0 <= self.max_ece <= 1:
            errors.append(f"max_ece ({self.max_ece}) must be in [0, 1]")

        if not 0 <= self.cal_window < 1:
            errors.append(f"cal_window ({self.cal_window}) must be in [0, 1)")

        if self.max_ts < 0:
            errors.append(f"max_ts ({self.max_ts}) must be >= 0")

        if self.ts_penalty_threshold < 0:
            errors.append(
                f"ts_penalty_threshold ({self.ts_penalty_threshold}) must be >= 0"
            )

        if self.ts_penalty_scale <= 0:
            errors.append(f"ts_penalty_scale ({self.ts_penalty_scale}) must be > 0")

        if self.n_folds < 3:
            errors.append(f"n_folds ({self.n_folds}) must be >= 3")

        if self.n_optuna_trials < 1:
            errors.append(f"n_optuna_trials ({self.n_optuna_trials}) must be >= 1")

        if self.threshold_alpha < 0 or self.threshold_alpha > 1:
            errors.append(
                f"threshold_alpha ({self.threshold_alpha}) must be in [0, 1]"
            )

        if self.training_window_days and self.training_window_days < 180:
            errors.append(
                f"training_window_days ({self.training_window_days}) must be >= 180 or 0 (disabled)"
            )

        if self.rfe_step < 1:
            errors.append(f"rfe_step ({self.rfe_step}) must be >= 1")

        if self.mrmr_min > 0 and self.mrmr_k > 0 and self.mrmr_min >= self.mrmr_k:
            errors.append(f"mrmr_min ({self.mrmr_min}) must be < mrmr_k ({self.mrmr_k})")
        if self.mrmr_step < 1:
            errors.append(f"mrmr_step ({self.mrmr_step}) must be >= 1")

        if self.embargo_multiplier <= 0:
            errors.append(f"embargo_multiplier ({self.embargo_multiplier}) must be > 0")

        if self.aggressive_reg_auc_threshold < 0.5 or self.aggressive_reg_auc_threshold > 1.0:
            errors.append(
                f"aggressive_reg_auc_threshold ({self.aggressive_reg_auc_threshold}) must be in [0.5, 1.0]"
            )

        if self.tl_base_iterations < 10:
            errors.append(f"tl_base_iterations ({self.tl_base_iterations}) must be >= 10")

        valid_filter_modes = ("blocklist", "nan_filter", "hybrid")
        if self.niche_filter_mode not in valid_filter_modes:
            errors.append(
                f"niche_filter_mode ({self.niche_filter_mode}) must be one of {valid_filter_modes}"
            )

        return errors
