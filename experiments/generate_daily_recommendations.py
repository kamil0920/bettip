#!/usr/bin/env python3
"""
Unified Daily Recommendation Generator

Uses pre-trained sniper models (ModelLoader + FeatureLookup) to generate
predictions directly, matching the same logic as match_scheduler --predict.

Output: data/05-recommendations/rec_YYYYMMDD_NNN.csv

Usage:
    python experiments/generate_daily_recommendations.py
    python experiments/generate_daily_recommendations.py --min-edge 5
    python experiments/generate_daily_recommendations.py --schedule-file data/06-prematch/today_schedule.json
"""
import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress numpy warnings from empty slices in feature EMA/median calculations
# (expected for teams with limited history; NaNs are filled downstream)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning
)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_lookup import FeatureLookup
from src.ml.model_loader import ModelLoader
from src.ml.feature_injector import ExternalFeatureInjector
from src.ml.bankroll_manager import BankrollManager, RiskConfig
from src.odds.odds_features import remove_vig_2way

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Maps sniper deployment market names to odds columns in odds_latest.parquet
MARKET_ODDS_COLUMNS = {
    "away_win": "h2h_away_avg",
    "home_win": "h2h_home_avg",
    "over25": "totals_over_avg",
    "under25": "totals_under_avg",
    "btts": "btts_yes_avg",
    "shots": "shots_over_avg",
    "corners": "corners_over_avg",
    "cards": "cards_over_avg",
    "fouls": "fouls_over_avg",
    # Niche market line variants (share odds column with base market)
    "cards_over_35": "cards_over_avg",
    "cards_over_55": "cards_over_avg",
    "cards_over_65": "cards_over_avg",
    "corners_over_85": "corners_over_avg",
    "corners_over_105": "corners_over_avg",
    "corners_over_115": "corners_over_avg",
    "shots_over_225": "shots_over_avg",
    "shots_over_265": "shots_over_avg",
    "shots_over_285": "shots_over_avg",
    "fouls_over_225": "fouls_over_avg",
    "fouls_over_265": "fouls_over_avg",
    "fouls_over_285": "fouls_over_avg",
}

# Complementary odds columns for 2-way vig removal
# Maps a market to its opposite-side odds column
MARKET_COMPLEMENT_COLUMNS = {
    "over25": "totals_under_avg",
    "under25": "totals_over_avg",
    "btts": "btts_no_avg",
    "shots": "shots_under_avg",
    "corners": "corners_under_avg",
    "cards": "cards_under_avg",
    "fouls": "fouls_under_avg",
    "cards_over_35": "cards_under_avg",
    "cards_over_55": "cards_under_avg",
    "cards_over_65": "cards_under_avg",
    "corners_over_85": "corners_under_avg",
    "corners_over_105": "corners_under_avg",
    "corners_over_115": "corners_under_avg",
    "shots_over_225": "shots_under_avg",
    "shots_over_265": "shots_under_avg",
    "shots_over_285": "shots_under_avg",
    "fouls_over_225": "fouls_under_avg",
    "fouls_over_265": "fouls_under_avg",
    "fouls_over_285": "fouls_under_avg",
}

# Default implied probabilities when no odds available (same as match_scheduler)
MARKET_BASELINES = {
    "away_win": 0.30,
    "home_win": 0.45,
    "over25": 0.50,
    "under25": 0.50,
    "btts": 0.50,
    "shots": 0.50,
    "corners": 0.50,
    "cards": 0.50,
    "fouls": 0.50,
    # Line variants share baseline with base market
    "cards_over_35": 0.50,
    "cards_over_55": 0.50,
    "cards_over_65": 0.50,
    "corners_over_85": 0.50,
    "corners_over_105": 0.50,
    "corners_over_115": 0.50,
    "shots_over_225": 0.50,
    "shots_over_265": 0.50,
    "shots_over_285": 0.50,
    "fouls_over_225": 0.50,
    "fouls_over_265": 0.50,
    "fouls_over_285": 0.50,
}

BANKROLL_STATE_PATH = project_root / "data" / "bankroll_state.json"

# Default betting config — can be overridden by "betting_config" in deployment JSON
DEFAULT_BETTING_CONFIG = {
    "kelly_fraction": 0.25,
    "max_stake_fraction": 0.05,
    "initial_bankroll": 1000,
    "stop_loss_daily": 0.10,
    "take_profit_daily": 0.20,
}


def _load_bankroll_state() -> dict:
    """Load bankroll state from disk, or return defaults."""
    if BANKROLL_STATE_PATH.exists():
        try:
            with open(BANKROLL_STATE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    return {"bankroll": DEFAULT_BETTING_CONFIG["initial_bankroll"]}


def _save_bankroll_state(state: dict) -> None:
    """Persist bankroll state to disk."""
    BANKROLL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BANKROLL_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _create_bankroll_manager(deployment_config: Optional[Dict] = None) -> BankrollManager:
    """Create BankrollManager from deployment config + bankroll state."""
    state = _load_bankroll_state()
    bankroll = state.get("bankroll", DEFAULT_BETTING_CONFIG["initial_bankroll"])

    # Merge betting_config from deployment config if present
    betting_cfg = dict(DEFAULT_BETTING_CONFIG)
    if deployment_config and "betting_config" in deployment_config:
        betting_cfg.update(deployment_config["betting_config"])

    config = RiskConfig(
        kelly_fraction=betting_cfg["kelly_fraction"],
        max_stake_fraction=betting_cfg["max_stake_fraction"],
        stop_loss_daily=betting_cfg["stop_loss_daily"],
        take_profit_daily=betting_cfg["take_profit_daily"],
    )
    return BankrollManager(total_bankroll=bankroll, config=config)


# Human-readable market labels for CSV output
MARKET_LABELS = {
    "away_win": "AWAY_WIN",
    "home_win": "HOME_WIN",
    "over25": "OVER_2.5",
    "under25": "UNDER_2.5",
    "btts": "BTTS",
    "fouls": "FOULS",
    "shots": "SHOTS",
    "corners": "CORNERS",
    "cards": "CARDS",
    # Niche market line variants
    "cards_over_35": "CARDS_O3.5",
    "cards_over_55": "CARDS_O5.5",
    "cards_over_65": "CARDS_O6.5",
    "corners_over_85": "CORNERS_O8.5",
    "corners_over_105": "CORNERS_O10.5",
    "corners_over_115": "CORNERS_O11.5",
    "shots_over_225": "SHOTS_O22.5",
    "shots_over_265": "SHOTS_O26.5",
    "shots_over_285": "SHOTS_O28.5",
    "fouls_over_225": "FOULS_O22.5",
    "fouls_over_265": "FOULS_O26.5",
    "fouls_over_285": "FOULS_O28.5",
}


def load_schedule(schedule_file: Path) -> List[Dict]:
    """Load today's match schedule."""
    if not schedule_file.exists():
        logger.warning(f"Schedule file not found: {schedule_file}")
        return []

    with open(schedule_file) as f:
        data = json.load(f)

    matches = data.get("matches", [])
    logger.info(f"Loaded {len(matches)} matches from {schedule_file}")
    return matches


def load_sniper_config(config_path: Optional[Path] = None) -> Dict:
    """Load sniper deployment config for enabled markets and thresholds."""
    if config_path is None:
        config_path = project_root / "config" / "sniper_deployment.json"
    if not config_path.exists():
        logger.warning(f"Sniper deployment config not found: {config_path}")
        return {}

    with open(config_path) as f:
        config = json.load(f)

    markets = config.get("markets", {})
    enabled = {k: v for k, v in markets.items() if v.get("enabled", False)}
    logger.info(f"Enabled markets from {config_path.name}: {list(enabled.keys())}")
    return enabled


def load_odds() -> Optional[pd.DataFrame]:
    """Load pre-match odds from parquet."""
    odds_path = project_root / "data" / "prematch_odds" / "odds_latest.parquet"
    if not odds_path.exists():
        logger.warning(f"No odds file at {odds_path}")
        return None

    df = pd.read_parquet(odds_path)
    logger.info(f"Loaded odds for {len(df)} matches")
    return df


# Common team name aliases (short name → canonical words)
_TEAM_ALIASES: Dict[str, str] = {
    "wolves": "wolverhampton",
    "spurs": "tottenham",
    "man utd": "manchester united",
    "man city": "manchester city",
    "stade rennais": "rennes",
    "athletic bilbao": "athletic club",
    "atletico": "atletico madrid",
    "inter": "inter milan internazionale",
    "napoli": "ssc napoli",
    "gladbach": "monchengladbach mönchengladbach",
    "münchen": "munich",
    "psg": "paris saint germain",
    "paris fc": "parisfc",  # distinct from PSG — synthetic token
    "paris saint germain": "psg parissaintgermain",  # distinct from Paris FC
}


def _normalize_team(name: str) -> set:
    """Extract significant words from a team name for fuzzy matching."""
    lower = name.lower().replace("-", " ").replace("'", "")

    # Apply aliases: expand known short names
    for alias, expansion in _TEAM_ALIASES.items():
        if alias in lower:
            lower = lower + " " + expansion

    noise = {
        "fc", "cf", "sc", "ac", "as", "us", "ss", "rc", "cd", "ud", "sd",
        "afc", "ssc", "ogc", "vfl", "fsv", "tsv", "sv", "bsc", "hsv",
        "1899", "1860", "1.", "de", "la", "le", "los", "real", "04",
    }
    words = set()
    for w in lower.split():
        w = w.strip(".()")
        if w and w not in noise and len(w) > 1:
            words.add(w)
    return words


def _teams_match(name_a: str, name_b: str) -> bool:
    """Check if two team names refer to the same team using word overlap."""
    words_a = _normalize_team(name_a)
    words_b = _normalize_team(name_b)
    if not words_a or not words_b:
        return False
    # At least one significant word must overlap
    overlap = words_a & words_b
    if overlap:
        return True
    # Substring check on the shortest word sets
    for wa in words_a:
        for wb in words_b:
            if len(wa) >= 4 and len(wb) >= 4 and (wa in wb or wb in wa):
                return True
    return False


def get_match_odds(
    odds_df: Optional[pd.DataFrame],
    home_team: str,
    away_team: str,
    fixture_id: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """Look up odds for a specific match, returning column_name→odds mapping."""
    # Include draw odds + complement odds for vig removal calculation
    all_cols = list(set(
        list(MARKET_ODDS_COLUMNS.values())
        + list(MARKET_COMPLEMENT_COLUMNS.values())
        + ["h2h_draw_avg"]
    ))
    result: Dict[str, Optional[float]] = {col: None for col in all_cols}

    if odds_df is None or odds_df.empty:
        return result

    row = pd.DataFrame()

    # 1. Try fixture_id match (most reliable — same data source)
    if fixture_id and "fixture_id" in odds_df.columns:
        # Handle int/str type mismatch between schedule and odds parquet
        try:
            fid = int(fixture_id)
            mask = odds_df["fixture_id"].astype(int) == fid
        except (ValueError, TypeError):
            mask = odds_df["fixture_id"] == fixture_id
        row = odds_df[mask]

    # 2. Try exact team name match
    if row.empty and "home_team" in odds_df.columns:
        mask = (odds_df["home_team"] == home_team) & (odds_df["away_team"] == away_team)
        row = odds_df[mask]

    # 3. Fuzzy word-based match (handles different naming conventions)
    if row.empty and "home_team" in odds_df.columns:
        for idx, r in odds_df.iterrows():
            oh = str(r.get("home_team", ""))
            oa = str(r.get("away_team", ""))
            if _teams_match(home_team, oh) and _teams_match(away_team, oa):
                row = odds_df.loc[[idx]]
                logger.debug(
                    f"  Fuzzy odds match: '{home_team}' → '{oh}', "
                    f"'{away_team}' → '{oa}'"
                )
                break

    if row.empty:
        logger.info(f"  No odds found for {home_team} vs {away_team}")
        return result

    row = row.iloc[0]
    for col in all_cols:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and val > 1.0:
                result[col] = float(val)

    return result


def calculate_edge(
    prob: float,
    market: str,
    match_odds: Dict[str, Optional[float]],
) -> float:
    """
    Calculate edge = model_prob - fair_prob (vig-removed).

    Main markets (home_win, away_win): 3-way vig removal (H/D/A)
    Totals/BTTS/niche (over25, under25, btts, shots, corners, cards, fouls):
        2-way vig removal using over/under pair
    Fallback: baseline implied probability when odds unavailable.
    """
    odds_col = MARKET_ODDS_COLUMNS.get(market)
    market_odds_val = match_odds.get(odds_col) if odds_col else None

    if market in ("home_win", "away_win"):
        if market_odds_val:
            # 3-way vig removal (H/D/A)
            home_col_odds = match_odds.get("h2h_home_avg")
            away_col_odds = match_odds.get("h2h_away_avg")
            if home_col_odds and away_col_odds:
                draw_odds = match_odds.get("h2h_draw_avg")
                total_implied = 1 / home_col_odds + 1 / away_col_odds + (
                    1 / draw_odds if draw_odds and draw_odds > 1.0 else 0.28
                )
                implied = (1 / market_odds_val) / total_implied
                return prob - implied
        return prob - MARKET_BASELINES.get(market, 0.5)

    # All other markets: 2-way vig removal
    complement_col = MARKET_COMPLEMENT_COLUMNS.get(market)
    complement_odds = match_odds.get(complement_col) if complement_col else None

    if market_odds_val and market_odds_val > 1.0:
        if complement_odds and complement_odds > 1.0:
            # 2-way vig removal
            fair_prob, _ = remove_vig_2way(market_odds_val, complement_odds)
            return prob - fair_prob
        else:
            # No complement available — raw implied (still better than baseline)
            return prob - (1.0 / market_odds_val)

    return prob - MARKET_BASELINES.get(market, 0.5)


def _load_prematch_lineups(fixture_id: int) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Load confirmed lineups from prematch cache.

    Checks multiple path patterns for lineup data. Returns (None, None) when
    lineups aren't available yet (morning run) — this is the expected case.

    Args:
        fixture_id: API-Football fixture ID.

    Returns:
        Tuple of (home_lineup, away_lineup) dicts, or (None, None) if unavailable.
    """
    prematch_dir = project_root / "data" / "06-prematch"

    path_patterns = [
        prematch_dir / str(fixture_id) / "lineup_window_latest.json",
        prematch_dir / f"fixture_{fixture_id}.json",
    ]

    for path in path_patterns:
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            lineups = data.get("lineups", {})
            if lineups.get("available"):
                return lineups.get("home"), lineups.get("away")
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to parse lineup from {path}: {e}")

    return None, None


def _score_all_strategies(
    model_probs: List[tuple],
    market_name: str,
    market_config: Dict,
    match_odds: Dict,
    match_info: Dict,
    min_edge: float,
    scores_file: Path,
) -> None:
    """
    Score all strategy variants for a match/market and append to JSONL file.

    For main markets (home_win, away_win, over25, under25, btts), scores 6 strategies:
    lightgbm, catboost, xgboost, stacking_avg, stacking_weighted, agreement.
    For niche markets, scores individual line models only.
    """
    threshold = market_config.get("threshold", 0.5)
    wf_data = market_config.get("walkforward", {})
    wf_best = (wf_data.get("best_model") or wf_data.get("best_model_wf", "")).lower()

    # Determine if we have real odds
    odds_col = MARKET_ODDS_COLUMNS.get(market_name)
    odds_value = match_odds.get(odds_col) if odds_col else None
    has_real_odds = odds_value is not None and odds_value > 1.0

    entry = {
        "date": match_info["date"],
        "fixture_id": match_info.get("fixture_id", ""),
        "home": match_info["home_team"],
        "away": match_info["away_team"],
        "league": match_info.get("league", ""),
        "market": market_name,
        "deployed": wf_best or "single",
        "threshold": threshold,
        "min_edge": min_edge,
        "has_real_odds": has_real_odds,
        "strategies": {},
    }

    is_niche = market_name in ("fouls", "cards")

    if is_niche:
        # Niche markets: score each line model individually
        for name, prob, conf in model_probs:
            edge = calculate_edge(prob, market_name, match_odds)
            passes = prob >= threshold and edge >= min_edge
            entry["strategies"][name] = {
                "prob": round(prob, 3),
                "edge": round(edge * 100, 1),
                "pass": passes,
            }
    else:
        # Main markets: score individual models + ensemble strategies
        # Individual models
        model_map = {}  # base_type -> (name, prob, conf)
        for name, prob, conf in model_probs:
            for base in ("lightgbm", "catboost", "xgboost"):
                if base in name.lower():
                    model_map[base] = (name, prob, conf)
                    edge = calculate_edge(prob, market_name, match_odds)
                    passes = prob >= threshold and edge >= min_edge
                    entry["strategies"][base] = {
                        "prob": round(prob, 3),
                        "edge": round(edge * 100, 1),
                        "pass": passes,
                    }
                    break

        if len(model_probs) >= 2:
            # Stacking average
            avg_prob = sum(p for _, p, _ in model_probs) / len(model_probs)
            avg_edge = calculate_edge(avg_prob, market_name, match_odds)
            avg_passes = avg_prob >= threshold and avg_edge >= min_edge
            entry["strategies"]["stacking_avg"] = {
                "prob": round(avg_prob, 3),
                "edge": round(avg_edge * 100, 1),
                "pass": avg_passes,
            }

            # Stacking weighted (Ridge meta-learner)
            weights = market_config.get("stacking_weights")
            if weights and all(b in weights for b in model_map):
                w = np.array([weights[b] for b in model_map])
                raw = sum(
                    w_i * p
                    for w_i, (_, p, _) in zip(w, (model_map[b] for b in model_map))
                )
                weighted_prob = float(1 / (1 + np.exp(-raw)))
            else:
                weighted_prob = avg_prob  # fallback
            weighted_edge = calculate_edge(weighted_prob, market_name, match_odds)
            weighted_passes = weighted_prob >= threshold and weighted_edge >= min_edge
            entry["strategies"]["stacking_weighted"] = {
                "prob": round(weighted_prob, 3),
                "edge": round(weighted_edge * 100, 1),
                "pass": weighted_passes,
            }

            # Agreement (min probability across all models — matches optimization logic)
            min_prob = min(p for _, p, _ in model_probs)
            total_count = len(model_probs)
            agree_edge = calculate_edge(min_prob, market_name, match_odds)
            agree_passes = min_prob >= threshold and agree_edge >= min_edge
            entry["strategies"]["agreement"] = {
                "prob": round(min_prob, 3),
                "edge": round(agree_edge * 100, 1),
                "pass": agree_passes,
                "models": total_count,
            }

    # Append to JSONL
    scores_file.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def generate_sniper_predictions(
    matches: List[Dict],
    min_edge_pct: float = 5.0,
    deployment_config: Optional[Path] = None,
    bankroll_manager: Optional[BankrollManager] = None,
) -> List[Dict]:
    """
    Generate predictions using pre-trained sniper models.

    Replicates the prediction logic from match_scheduler --predict but outputs
    to the recommendation CSV format. Optionally sizes bets via Kelly criterion.
    """
    enabled_markets = load_sniper_config(deployment_config)
    if not enabled_markets:
        logger.error("No enabled markets in sniper deployment config")
        return []

    # Initialize model loader, feature lookup, and feature injector
    model_loader = ModelLoader()
    feature_lookup = FeatureLookup()
    feature_injector = ExternalFeatureInjector(enable_weather=False)

    available_models = model_loader.list_available_models()
    if not available_models:
        logger.error("No models available in models/ directory")
        return []
    logger.info(f"Available models: {available_models}")

    if not feature_lookup.load():
        logger.error("Failed to load features")
        return []

    # Load odds (from The Odds API or API-Football)
    odds_df = load_odds()
    if odds_df is not None:
        logger.info(f"Using real odds for edge calculation ({len(odds_df)} matches)")
    else:
        logger.warning("No odds file found — using baseline implied probabilities for edge calculation")

    min_edge = min_edge_pct / 100.0
    predictions = []

    for match in matches:
        home_team = match["home_team"]
        away_team = match["away_team"]
        league = match.get("league", "")
        fixture_id = match.get("fixture_id", "")
        kickoff = match.get("kickoff", "")
        match_date = str(kickoff)[:10] if kickoff else datetime.now().strftime("%Y-%m-%d")

        # Get features for this match
        features_df = feature_lookup.get_team_features(home_team, away_team)
        if features_df is None:
            logger.debug(f"No features for {home_team} vs {away_team}")
            continue

        # Inject late-breaking external features (referee, lineups)
        # These features use the assigned referee's stats and confirmed lineups
        # to provide context the model learned during training
        match_referee = match.get("referee", "")
        venue_city = match.get("venue", {}).get("city", "") if isinstance(match.get("venue"), dict) else match.get("venue_city", "")

        # Load lineups from prematch cache (None if not yet available)
        home_lineup, away_lineup = _load_prematch_lineups(fixture_id) if fixture_id else (None, None)

        features_df = feature_injector.inject_features(features_df, {
            'referee': match_referee,
            'venue_city': venue_city,
            'kickoff': kickoff,
            'home_lineup': home_lineup,
            'away_lineup': away_lineup,
            'home_team': home_team,
            'away_team': away_team,
        })

        # Get odds for this match (includes draw odds for vig removal)
        match_odds = get_match_odds(odds_df, home_team, away_team, fixture_id=fixture_id)

        # Run each enabled market's models
        for market_name, market_config in enabled_markets.items():
            threshold = market_config.get("threshold", 0.5)
            model_type = market_config.get("model", "").lower()
            saved_models = market_config.get("saved_models", [])

            # Determine which model names to try
            model_names_to_try = []

            # Use saved_models from deployment config for all markets
            for saved in saved_models:
                name = saved.replace(".joblib", "")
                if name in available_models:
                    model_names_to_try.append(name)

            # No prefix fallback — only use models explicitly listed in deployment config
            if not model_names_to_try:
                if saved_models:
                    logger.warning(
                        f"[MODEL MISMATCH] {market_name}: config lists {saved_models} "
                        f"but none found in models/. Run download_data.py to sync."
                    )
                else:
                    logger.warning(f"[NO MODELS] {market_name}: no saved_models in config. Skipping.")
                continue

            # Collect predictions from all model variants for this market
            model_probs = []
            for model_name in model_names_to_try:
                result = model_loader.predict(model_name, features_df)
                if result:
                    prob, confidence = result
                    model_probs.append((model_name, prob, confidence))

            if not model_probs:
                continue

            # Score all strategies for comparison tracking
            scores_file = project_root / "data" / "06-prematch" / "strategy_scores.jsonl"
            _score_all_strategies(
                model_probs=model_probs,
                market_name=market_name,
                market_config=market_config,
                match_odds=match_odds,
                match_info={
                    "date": match_date,
                    "fixture_id": fixture_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "league": league,
                },
                min_edge=min_edge,
                scores_file=scores_file,
            )

            # Use walkforward.best_model as primary model selection strategy
            wf_data = market_config.get("walkforward", {})
            wf_best = (wf_data.get("best_model") or wf_data.get("best_model_wf", "")).lower()

            if wf_best == "stacking" and len(model_probs) >= 2:
                # Use Ridge meta-learner weights if available, else simple average
                weights = market_config.get("stacking_weights", {})
                model_map = {}
                for n, p, c in model_probs:
                    for base in ("lightgbm", "catboost", "xgboost"):
                        if base in n.lower():
                            model_map[base] = (n, p, c)
                            break
                if weights and all(b in weights for b in model_map):
                    w = np.array([weights[b] for b in model_map])
                    raw = sum(
                        w_i * p
                        for w_i, (_, p, _) in zip(w, (model_map[b] for b in model_map))
                    )
                    prob = float(1 / (1 + np.exp(-raw)))
                else:
                    prob = sum(p for _, p, _ in model_probs) / len(model_probs)
                best_model = "stacking"
                confidence = sum(c for _, _, c in model_probs) / len(model_probs)
            elif wf_best in ("average", "temporal_blend") and len(model_probs) >= 2:
                # average: mean of all model probabilities
                # temporal_blend: approximated with average at prediction time
                prob = sum(p for _, p, _ in model_probs) / len(model_probs)
                best_model = wf_best
                confidence = sum(c for _, _, c in model_probs) / len(model_probs)
            elif wf_best == "agreement" and len(model_probs) >= 2:
                # Agreement = min probability across all models (matches optimization logic)
                min_prob = min(p for _, p, _ in model_probs)
                if min_prob < threshold:
                    logger.info(
                        f"  {home_team} vs {away_team} | {market_name}: "
                        f"agreement failed (min_prob={min_prob:.3f}, "
                        f"threshold={threshold:.2f})"
                    )
                    continue
                prob = min_prob
                best_model = "agreement"
                confidence = min(c for _, _, c in model_probs)
            elif wf_best.startswith("disagree_") and wf_best.endswith("_filtered") and len(model_probs) >= 2:
                # Disagreement-filtered ensembles: average when models agree, skip when they don't
                # Thresholds from src/ml/ensemble_disagreement.py create_disagreement_ensemble()
                disagree_configs = {
                    "disagree_conservative_filtered": {
                        "agree_thresh": 0.08, "min_edge": 0.05, "min_prob": 0.55, "max_prob": 0.80,
                    },
                    "disagree_balanced_filtered": {
                        "agree_thresh": 0.12, "min_edge": 0.03, "min_prob": 0.50, "max_prob": 0.85,
                    },
                    "disagree_aggressive_filtered": {
                        "agree_thresh": 0.15, "min_edge": 0.02, "min_prob": 0.45, "max_prob": 0.90,
                    },
                }
                cfg = disagree_configs.get(wf_best, disagree_configs["disagree_balanced_filtered"])
                all_probs = [p for _, p, _ in model_probs]
                avg_prob = sum(all_probs) / len(all_probs)
                std_prob = float(np.std(all_probs))

                # Get implied market probability for edge check
                odds_col = MARKET_ODDS_COLUMNS.get(market_name)
                mkt_odds = match_odds.get(odds_col) if odds_col else None
                implied_prob = (1.0 / mkt_odds) if mkt_odds and mkt_odds > 1.0 else MARKET_BASELINES.get(market_name, 0.5)
                edge_vs_mkt = avg_prob - implied_prob

                # Check filters: models agree, edge beats market, prob in range
                if std_prob > cfg["agree_thresh"]:
                    logger.info(
                        f"  {home_team} vs {away_team} | {market_name}: "
                        f"disagree filter rejected — models disagree "
                        f"(std={std_prob:.3f} > {cfg['agree_thresh']})"
                    )
                    continue
                if edge_vs_mkt < cfg["min_edge"]:
                    logger.info(
                        f"  {home_team} vs {away_team} | {market_name}: "
                        f"disagree filter rejected — insufficient edge "
                        f"(edge={edge_vs_mkt:.3f} < {cfg['min_edge']})"
                    )
                    continue
                if not (cfg["min_prob"] <= avg_prob <= cfg["max_prob"]):
                    logger.info(
                        f"  {home_team} vs {away_team} | {market_name}: "
                        f"disagree filter rejected — prob out of range "
                        f"(prob={avg_prob:.3f}, range=[{cfg['min_prob']}, {cfg['max_prob']}])"
                    )
                    continue
                prob = avg_prob
                best_model = wf_best
                confidence = sum(c for _, _, c in model_probs) / len(model_probs)
            else:
                # Specific model name (e.g. "xgboost", "lightgbm", "catboost")
                target = wf_best or model_type.lower()
                matched = [(n, p, c) for n, p, c in model_probs if target in n.lower()]
                if matched:
                    best_model, prob, confidence = matched[0]
                else:
                    best_model, prob, confidence = model_probs[0]  # fallback

            # Check threshold
            if prob < threshold:
                logger.info(
                    f"  {home_team} vs {away_team} | {market_name}: "
                    f"below threshold (strategy={wf_best}, model={best_model}, "
                    f"prob={prob:.3f}, threshold={threshold:.2f})"
                )
                continue

            # Calculate edge
            edge = calculate_edge(prob, market_name, match_odds)
            if edge < min_edge:
                logger.info(
                    f"  {home_team} vs {away_team} | {market_name}: "
                    f"low edge (strategy={wf_best}, model={best_model}, "
                    f"prob={prob:.3f}, edge={edge*100:.1f}%, min={min_edge*100:.0f}%)"
                )
                continue

            # Determine odds value for CSV
            odds_col = MARKET_ODDS_COLUMNS.get(market_name)
            odds_value = match_odds.get(odds_col) if odds_col else None
            has_real_odds = odds_value is not None and odds_value > 1.0
            if not has_real_odds:
                odds_value = 0

            # Determine bet type and line for niche markets
            bet_type = MARKET_LABELS.get(market_name, market_name.upper())
            line = 0.0

            if market_name in ("fouls", "cards"):
                # Legacy niche markets: use the specific model that produced the best edge
                best_niche = max(model_probs, key=lambda x: calculate_edge(x[1], market_name, match_odds))
                best_model, prob, confidence = best_niche
                edge = calculate_edge(prob, market_name, match_odds)
                if edge < min_edge:
                    continue

                # Parse line from model name (e.g., fouls_over_24_5 → 24.5)
                bet_type = "OVER"
                parts = best_model.split("_")
                for i, p in enumerate(parts):
                    if p == "over" and i + 1 < len(parts):
                        try:
                            line = float("_".join(parts[i + 1:]). replace("_", "."))
                        except ValueError:
                            pass
                        break

                market_label = market_name.upper()
            else:
                market_label = MARKET_LABELS.get(market_name, market_name.upper())

            # Calculate Kelly stake if bankroll manager is available
            kelly_stake = 0.0
            if bankroll_manager and has_real_odds and odds_value > 1.0:
                kelly_stake = bankroll_manager.calculate_stake(
                    market=market_name,
                    probability=prob,
                    odds=odds_value,
                    edge=edge,
                )

            predictions.append({
                "date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "league": league,
                "market": market_label if market_name not in ("fouls", "shots", "corners", "cards") else market_name.upper(),
                "bet_type": bet_type,
                "line": line,
                "odds": odds_value,
                "probability": round(prob, 4),
                "edge": round(edge * 100, 2),
                "kelly_stake": round(kelly_stake, 2),
                "edge_source": "real" if has_real_odds else "baseline",
                "referee": "",
                "fixture_id": fixture_id,
                "result": "",
                "actual": "",
            })

            odds_source = f"odds={odds_value:.2f}" if has_real_odds else "odds=BASELINE"
            logger.info(
                f"  {home_team} vs {away_team}: {market_name} "
                f"prob={prob:.3f} edge={edge*100:.1f}% "
                f"(strategy={wf_best}, model={best_model}, {odds_source})"
            )

    return predictions


def get_next_rec_number(date_str: str) -> int:
    """Get next recommendation file number for today."""
    rec_dir = project_root / 'data/05-recommendations'
    existing = list(rec_dir.glob(f'rec_{date_str}_*.csv'))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split('_')[-1])
            numbers.append(num)
        except ValueError:
            pass
    return max(numbers) + 1 if numbers else 1


def save_recommendations(df: pd.DataFrame) -> str:
    """Save recommendations to standardized format."""
    if df.empty:
        print("No recommendations to save")
        return ""

    rec_dir = project_root / 'data/05-recommendations'
    rec_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime('%Y%m%d')
    rec_num = get_next_rec_number(date_str)
    filename = f'rec_{date_str}_{rec_num:03d}.csv'
    filepath = rec_dir / filename

    columns = [
        'date', 'home_team', 'away_team', 'league', 'market', 'bet_type',
        'line', 'odds', 'probability', 'edge', 'kelly_stake', 'edge_source',
        'referee', 'fixture_id', 'result', 'actual'
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = ''

    df = df[columns]
    df.to_csv(filepath, index=False)

    print(f"\nSaved {len(df)} recommendations to: {filepath}")
    return str(filepath)


def update_readme_index(filepath: str, count: int) -> None:
    """Update README with new file."""
    readme_path = project_root / 'data/05-recommendations/README.md'
    if not readme_path.exists():
        return

    content = readme_path.read_text()

    filename = Path(filepath).name
    date_str = filename.split('_')[1]
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

    new_entry = f"| {filename} | {date_formatted} | {count} | Generated |\n"

    if '| File | Date |' in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '| File | Date |' in line:
                insert_idx = i + 2
                lines.insert(insert_idx, new_entry.strip())
                break
        content = '\n'.join(lines)
        readme_path.write_text(content)


def print_summary(df: pd.DataFrame) -> None:
    """Print summary of recommendations."""
    if df.empty:
        print("\nNo recommendations generated")
        return

    print("\n" + "=" * 70)
    print("DAILY RECOMMENDATIONS SUMMARY")
    print("=" * 70)

    print(f"\nTotal recommendations: {len(df)}")

    print("\nBy Market:")
    for market in df['market'].unique():
        count = len(df[df['market'] == market])
        avg_edge = df[df['market'] == market]['edge'].mean()
        print(f"  {market}: {count} bets, avg edge {avg_edge:.1f}%")

    print("\nBy Date:")
    for date in sorted(df['date'].unique()):
        count = len(df[df['date'] == date])
        print(f"  {date}: {count} bets")

    # Show Kelly stake summary if present
    if 'kelly_stake' in df.columns and df['kelly_stake'].sum() > 0:
        print(f"\nKelly Staking:")
        print(f"  Total Kelly stake: {df['kelly_stake'].sum():.2f}")
        print(f"  Avg Kelly stake:   {df['kelly_stake'].mean():.2f}")
        print(f"  Max Kelly stake:   {df['kelly_stake'].max():.2f}")

    print("\nTop 10 by Edge:")
    print("-" * 70)
    top10 = df.nlargest(10, 'edge')
    for _, row in top10.iterrows():
        match_str = f"{row['home_team'][:15]} vs {row['away_team'][:15]}"
        bet = f"{row['market']} {row['bet_type']}"
        if row['line']:
            bet += f" {row['line']}"
        kelly_str = f" K={row['kelly_stake']:.1f}" if row.get('kelly_stake', 0) > 0 else ""
        print(f"  {row['date']} | {match_str:<32} | {bet:<20} | +{row['edge']:.1f}%{kelly_str}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Generate daily recommendations using sniper models')
    parser.add_argument('--min-edge', type=float, default=5.0,
                        help='Minimum edge percentage (default: 5)')
    parser.add_argument('--schedule-file', type=str,
                        default='data/06-prematch/today_schedule.json',
                        help='Path to schedule JSON file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print summary but don\'t save file')
    parser.add_argument('--deployment-config', type=str, nargs='+',
                        default=None,
                        help='Path(s) to deployment config JSON (default: config/sniper_deployment.json). '
                             'Multiple configs are merged (e.g., European + Americas).')
    parser.add_argument('--no-kelly', action='store_true',
                        help='Disable Kelly stake sizing (flat 1-unit stakes)')
    args = parser.parse_args()

    print("=" * 70)
    print("DAILY RECOMMENDATION GENERATOR (Sniper Models)")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Minimum Edge: {args.min_edge}%")
    print("=" * 70)

    # Load schedule
    schedule_path = project_root / args.schedule_file
    matches = load_schedule(schedule_path)
    if not matches:
        print("No matches in schedule")
        return 1

    # Initialize bankroll manager for Kelly staking
    bankroll_mgr = None
    if not args.no_kelly:
        bankroll_mgr = _create_bankroll_manager()
        state = _load_bankroll_state()
        print(f"Bankroll: {state.get('bankroll', DEFAULT_BETTING_CONFIG['initial_bankroll']):.0f} | "
              f"Kelly fraction: {bankroll_mgr.config.kelly_fraction}")

    # Generate predictions using sniper models
    # Support multiple deployment configs (e.g., European + Americas)
    config_paths = [Path(p) for p in args.deployment_config] if args.deployment_config else [None]
    print(f"\nRunning sniper model predictions for {len(matches)} matches...")
    all_predictions = []
    for cfg_path in config_paths:
        preds = generate_sniper_predictions(
            matches,
            min_edge_pct=args.min_edge,
            deployment_config=cfg_path,
            bankroll_manager=bankroll_mgr,
        )
        all_predictions.extend(preds)

    if not all_predictions:
        print("\nNo predictions met edge threshold")
        return 1

    df = pd.DataFrame(all_predictions)

    # Sort by edge descending
    df = df.sort_values('edge', ascending=False)

    # Remove duplicates (same match + market)
    df = df.drop_duplicates(subset=['home_team', 'away_team', 'market', 'bet_type', 'line'])

    print_summary(df)

    if not args.dry_run and not df.empty:
        filepath = save_recommendations(df)
        if filepath:
            update_readme_index(filepath, len(df))
            print(f"\nRecommendations file: {filepath}")

    return 0 if not df.empty else 1


if __name__ == '__main__':
    sys.exit(main())
