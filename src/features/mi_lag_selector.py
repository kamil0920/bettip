"""
MI-Based Lag Selector — Model-Free Optimal Parameter Selection

Uses mutual information (MI) to directly measure how much information each
temporal parameter lag captures about the target, without fitting any model.

AFML Ch 8 insight: MI-based lag selection is model-independent, fast (zero model
fits), and less prone to overfitting than grid search through downstream model
performance.

Usage:
    python -m src.features.mi_lag_selector --bet-type corners
    python -m src.features.mi_lag_selector --all
    python -m src.features.mi_lag_selector --bet-type corners --save-config
    python -m src.features.mi_lag_selector --bet-type corners --compare-defaults
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.features.config_manager import (
    BET_TYPE_PARAM_PRIORITIES,
    BetTypeFeatureConfig,
)
from src.features.engineers.entropy import _permutation_entropy

logger = logging.getLogger(__name__)

FEATURES_FILE = Path("data/03-features/features_all_5leagues_with_odds.parquet")


class MILagSelector:
    """Compute MI profiles for temporal feature parameters.

    For each tunable temporal parameter and a given bet type target, computes
    MI(feature_at_lag_k, target) across a range of lag values. The optimal lag
    is at the MI peak. Flat profiles indicate parameter insensitivity.
    """

    # Target definitions: bet_type -> (raw_col_or_stat, threshold, direction)
    # "classification" means the column is already binary.
    TARGETS: Dict[str, Tuple[str, Optional[float], str]] = {
        "home_win": ("home_win", None, "classification"),
        "away_win": ("away_win", None, "classification"),
        "over25": ("total_goals", 2.5, "over"),
        "under25": ("total_goals", 2.5, "under"),
        "btts": ("btts", None, "classification"),
        "fouls": ("total_fouls", 24.5, "over"),
        "cards": ("total_cards", 4.5, "over"),
        "shots": ("total_shots", 24.5, "over"),
        "corners": ("total_corners", 9.5, "over"),
    }

    # Raw stats available for each stat group
    STAT_GROUPS: Dict[str, List[str]] = {
        "goals": ["home_goals", "away_goals"],
        "fouls": ["home_fouls", "away_fouls"],
        "cards": ["home_yellows", "away_yellows"],
        "shots": ["home_shots", "away_shots"],
        "corners": ["home_corners", "away_corners"],
    }

    # Map: config param -> (stat_group, computation_type, lag_range)
    # For dynamics_window and entropy_window, stat_group is resolved dynamically
    # per bet type in compute_all_profiles (e.g., "corners" for corners market).
    PARAM_PROFILES: Dict[str, Tuple[str, str, range]] = {
        "ema_span": ("goals", "ema", range(3, 36)),
        "form_window": ("goals", "rolling_mean", range(3, 31)),
        "poisson_lookback": ("goals", "rolling_mean", range(5, 31)),
        "fouls_ema_span": ("fouls", "ema", range(3, 36)),
        "cards_ema_span": ("cards", "ema", range(3, 36)),
        "shots_ema_span": ("shots", "ema", range(3, 36)),
        "corners_ema_span": ("corners", "ema", range(3, 36)),
        "dynamics_window": (None, "rolling_kurtosis", range(5, 26)),  # type: ignore[arg-type]
        "entropy_window": (None, "rolling_pe", range(10, 26)),  # type: ignore[arg-type]
    }

    # Which params to profile per base market (derived from BET_TYPE_PARAM_PRIORITIES)
    MARKET_PARAMS: Dict[str, List[str]] = {
        "home_win": ["ema_span", "form_window", "poisson_lookback"],
        "away_win": ["ema_span", "form_window", "poisson_lookback"],
        "over25": ["ema_span", "form_window", "poisson_lookback"],
        "under25": ["ema_span", "form_window", "poisson_lookback"],
        "btts": ["ema_span", "form_window", "poisson_lookback"],
        "fouls": ["fouls_ema_span", "ema_span", "form_window", "dynamics_window", "entropy_window"],
        "cards": ["cards_ema_span", "ema_span", "form_window", "dynamics_window", "entropy_window"],
        "shots": ["shots_ema_span", "ema_span", "form_window", "dynamics_window", "entropy_window"],
        "corners": ["corners_ema_span", "ema_span", "form_window", "dynamics_window", "entropy_window"],
    }

    def __init__(
        self,
        features_path: Path = FEATURES_FILE,
        n_neighbors: int = 5,
        seed: int = 42,
    ):
        self.features_path = features_path
        self.n_neighbors = n_neighbors
        self.seed = seed
        self._df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load features parquet, derive targets, sort by date."""
        if self._df is not None:
            return self._df

        logger.info("Loading features from %s", self.features_path)
        df = pd.read_parquet(self.features_path)

        # Parse date and sort chronologically
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)

        # Derive targets that aren't directly in the parquet
        # over25 / under25
        if "total_goals" in df.columns:
            df["_over25"] = (df["total_goals"] > 2.5).astype(int)
            df["_under25"] = (df["total_goals"] <= 2.5).astype(int)

        # btts: both teams scored
        if "ft_home" in df.columns and "ft_away" in df.columns:
            df["_btts"] = ((df["ft_home"] > 0) & (df["ft_away"] > 0)).astype(int)

        # Map raw goal columns: features parquet uses ft_home/ft_away, not home_goals/away_goals
        if "ft_home" in df.columns and "home_goals" not in df.columns:
            df["home_goals"] = df["ft_home"]
        if "ft_away" in df.columns and "away_goals" not in df.columns:
            df["away_goals"] = df["ft_away"]

        # Cards: parquet has home_yellow_cards, we need home_yellows alias
        if "home_yellow_cards" in df.columns and "home_yellows" not in df.columns:
            df["home_yellows"] = df["home_yellow_cards"]
        if "away_yellow_cards" in df.columns and "away_yellows" not in df.columns:
            df["away_yellows"] = df["away_yellow_cards"]

        self._df = df
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        return df

    def _get_target(self, df: pd.DataFrame, bet_type: str) -> np.ndarray:
        """Derive binary target array for a bet type."""
        raw_col, threshold, direction = self.TARGETS[bet_type]

        if direction == "classification":
            # Already binary — check derived columns first
            derived_col = f"_{raw_col}"
            if derived_col in df.columns:
                return df[derived_col].values.astype(float)
            if raw_col in df.columns:
                return df[raw_col].values.astype(float)
            raise ValueError(f"Target column '{raw_col}' not found for {bet_type}")

        # Threshold-based target
        if raw_col not in df.columns:
            raise ValueError(f"Raw column '{raw_col}' not found for {bet_type}")
        values = df[raw_col].values.astype(float)
        if direction == "over":
            return (values > threshold).astype(float)
        else:  # under
            return (values <= threshold).astype(float)

    def _compute_per_team_feature(
        self,
        df: pd.DataFrame,
        raw_col: str,
        team_col: str,
        method: str,
        lag: int,
    ) -> pd.Series:
        """Compute per-team lagged feature matching production engineer logic.

        Args:
            df: DataFrame sorted by date
            raw_col: Raw stat column (e.g., 'home_fouls')
            team_col: Team identifier column (e.g., 'home_team_name')
            method: Computation type ('ema', 'rolling_mean', 'rolling_kurtosis', 'rolling_pe')
            lag: Window/span parameter value
        """
        if raw_col not in df.columns:
            return pd.Series(np.nan, index=df.index)

        if method == "ema":
            return df.groupby(team_col)[raw_col].transform(
                lambda x: x.shift(1).ewm(span=lag, min_periods=3).mean()
            )
        elif method == "rolling_mean":
            return df.groupby(team_col)[raw_col].transform(
                lambda x: x.shift(1).rolling(lag, min_periods=3).mean()
            )
        elif method == "rolling_kurtosis":
            return df.groupby(team_col)[raw_col].transform(
                lambda x: x.shift(1).rolling(lag, min_periods=max(4, lag // 2)).kurt()
            )
        elif method == "rolling_pe":
            return df.groupby(team_col)[raw_col].transform(
                lambda x: x.shift(1)
                .rolling(lag, min_periods=lag)
                .apply(lambda w: _permutation_entropy(w), raw=True)
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_mi_profile(
        self,
        df: pd.DataFrame,
        param_name: str,
        target: np.ndarray,
        stat_group_override: Optional[str] = None,
    ) -> Dict[int, float]:
        """Compute MI(feature_at_lag_k, target) for all lags of a parameter.

        Args:
            stat_group_override: Override stat group for params like dynamics_window
                that use market-specific stats (e.g., 'corners' for corners market).

        Returns:
            Dict mapping lag value to average MI score across home/away stats.
        """
        stat_group, method, lag_range = self.PARAM_PROFILES[param_name]
        stat_group = stat_group_override or stat_group
        raw_cols = self.STAT_GROUPS[stat_group]

        mi_profile: Dict[int, float] = {}
        for lag in lag_range:
            features = []
            for raw_col in raw_cols:
                # Determine team column from raw col prefix
                if raw_col.startswith("home_"):
                    team_col = "home_team_name"
                else:
                    team_col = "away_team_name"
                feat = self._compute_per_team_feature(df, raw_col, team_col, method, lag)
                features.append(feat)

            X_lag = np.column_stack([f.fillna(0).values for f in features])
            valid = ~np.isnan(target) & np.all(np.isfinite(X_lag), axis=1)

            if valid.sum() < 200:
                continue

            mi = mutual_info_classif(
                X_lag[valid],
                target[valid],
                n_neighbors=self.n_neighbors,
                random_state=self.seed,
            )
            mi_profile[lag] = float(np.mean(mi))

        return mi_profile

    def compute_all_profiles(
        self, bet_type: str
    ) -> Dict[str, Dict[int, float]]:
        """Compute MI profiles for all relevant params for a bet type.

        Args:
            bet_type: Base market name (e.g., 'corners', 'home_win')

        Returns:
            Dict mapping param_name to {lag: MI_score}.
        """
        # Resolve base market for niche line variants
        base = self._resolve_base_market(bet_type)
        params = self.MARKET_PARAMS.get(base, ["ema_span", "form_window"])

        df = self.load_data()
        target = self._get_target(df, base)

        profiles: Dict[str, Dict[int, float]] = {}
        for param_name in params:
            if param_name not in self.PARAM_PROFILES:
                continue
            logger.info("  Computing MI profile for %s...", param_name)
            # For dynamics_window/entropy_window, use the base market's stat group
            stat_group_override = None
            configured_group = self.PARAM_PROFILES[param_name][0]
            if configured_group is None:
                # Use base market stat group; fall back to goals for H2H markets
                stat_group_override = base if base in self.STAT_GROUPS else "goals"
            profiles[param_name] = self.compute_mi_profile(
                df, param_name, target, stat_group_override=stat_group_override
            )

        return profiles

    def select_optimal_params(self, bet_type: str) -> BetTypeFeatureConfig:
        """Select MI-optimal parameters and return a BetTypeFeatureConfig.

        For params not profiled (elo, h2h, half_life), keeps defaults.
        """
        profiles = self.compute_all_profiles(bet_type)

        # Start from defaults
        config = BetTypeFeatureConfig.load_for_bet_type(bet_type)

        for param_name, profile in profiles.items():
            if not profile:
                continue
            # Pick lag with highest MI
            best_lag = max(profile, key=profile.get)  # type: ignore[arg-type]
            if hasattr(config, param_name):
                setattr(config, param_name, best_lag)

        return config

    def print_report(
        self,
        profiles: Dict[str, Dict[int, float]],
        bet_type: str,
        compare_defaults: bool = False,
    ) -> None:
        """Print human-readable MI profile report with bar chart."""
        base = self._resolve_base_market(bet_type)
        target_info = self.TARGETS[base]
        raw_col, threshold, direction = target_info

        if direction == "classification":
            target_desc = raw_col
        else:
            target_desc = f"{raw_col} {'>' if direction == 'over' else '<='} {threshold}"

        defaults = BetTypeFeatureConfig.load_for_bet_type(bet_type)

        print(f"\nMI LAG PROFILE: {bet_type} (target: {target_desc})")
        print("=" * 60)

        for param_name, profile in profiles.items():
            if not profile:
                print(f"\n{param_name}:")
                print("  No valid MI values (insufficient data)")
                continue

            max_mi = max(profile.values())
            min_mi = min(profile.values())
            best_lag = max(profile, key=profile.get)  # type: ignore[arg-type]
            current_default = getattr(defaults, param_name, "?")

            # Detect flat profile: range < 20% of max
            is_flat = (max_mi - min_mi) < 0.2 * max_mi if max_mi > 0 else True

            print(f"\n{param_name}:")
            for lag in sorted(profile.keys()):
                mi = profile[lag]
                bar_len = int(40 * mi / max_mi) if max_mi > 0 else 0
                bar = "\u2593" * bar_len
                marker = " \u2190 peak" if lag == best_lag else ""
                print(f"  lag={lag:<3d} MI={mi:.4f}  {bar}{marker}")

            print(f"  OPTIMAL: {best_lag} (current default: {current_default})")

            if is_flat:
                print(f"  NOTE: flat profile \u2014 param has low sensitivity for this market")

            if compare_defaults and current_default != "?" and current_default != best_lag:
                default_mi = profile.get(current_default, 0.0)
                if default_mi > 0:
                    pct = (max_mi - default_mi) / default_mi * 100
                    print(f"  MI improvement: {pct:+.1f}% vs default lag={current_default}")

        # Summary
        print("\nRECOMMENDED CONFIG:")
        for param_name, profile in profiles.items():
            if not profile:
                continue
            best_lag = max(profile, key=profile.get)  # type: ignore[arg-type]
            current = getattr(defaults, param_name, "?")
            changed = " (unchanged)" if best_lag == current else f" (was {current})"
            max_mi = max(profile.values())
            min_mi = min(profile.values())
            is_flat = (max_mi - min_mi) < 0.2 * max_mi if max_mi > 0 else True
            flat_note = ", flat profile" if is_flat else ""
            print(f"  {param_name}: {best_lag}{changed}{flat_note}")

    @staticmethod
    def _resolve_base_market(bet_type: str) -> str:
        """Resolve niche line variant to its base market.

        e.g., 'corners_over_85' -> 'corners', 'cards_under_25' -> 'cards'
        """
        for base in ("fouls", "cards", "shots", "corners"):
            if bet_type.startswith(base):
                return base
        if bet_type.startswith("goals_over") or bet_type.startswith("goals_under"):
            return "over25"
        return bet_type


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MI-based lag selection for feature parameters"
    )
    parser.add_argument(
        "--bet-type",
        type=str,
        help="Bet type to profile (e.g., corners, home_win)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Profile all base markets",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save MI-optimal params to YAML config",
    )
    parser.add_argument(
        "--compare-defaults",
        action="store_true",
        help="Compare MI-selected vs current default params",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=FEATURES_FILE,
        help="Path to features parquet",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=5,
        help="KNN neighbors for MI estimation (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.bet_type and not args.all:
        parser.error("Must specify --bet-type or --all")

    selector = MILagSelector(
        features_path=args.features_path,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
    )

    bet_types: List[str] = []
    if args.all:
        bet_types = list(MILagSelector.TARGETS.keys())
    else:
        bet_types = [args.bet_type]

    for bt in bet_types:
        base = MILagSelector._resolve_base_market(bt)
        if base not in MILagSelector.TARGETS:
            print(f"WARNING: Unknown bet type '{bt}' (base='{base}'), skipping")
            continue

        logger.info("Profiling %s...", bt)
        profiles = selector.compute_all_profiles(bt)
        selector.print_report(profiles, bt, compare_defaults=args.compare_defaults)

        if args.save_config:
            config = selector.select_optimal_params(bt)
            path = config.save()
            print(f"\nSaved MI-optimal config to {path}")


if __name__ == "__main__":
    main()
