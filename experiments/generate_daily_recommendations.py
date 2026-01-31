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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_lookup import FeatureLookup
from src.ml.model_loader import ModelLoader

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
}

# Default implied probabilities when no odds available (same as match_scheduler)
MARKET_BASELINES = {
    "away_win": 0.30,
    "home_win": 0.45,
    "over25": 0.50,
    "under25": 0.50,
    "btts": 0.50,
}

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


def load_sniper_config() -> Dict:
    """Load sniper deployment config for enabled markets and thresholds."""
    config_path = project_root / "config" / "sniper_deployment.json"
    if not config_path.exists():
        logger.warning(f"Sniper deployment config not found: {config_path}")
        return {}

    with open(config_path) as f:
        config = json.load(f)

    markets = config.get("markets", {})
    enabled = {k: v for k, v in markets.items() if v.get("enabled", False)}
    logger.info(f"Enabled markets: {list(enabled.keys())}")
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
    # Include draw odds for vig removal calculation
    all_cols = list(MARKET_ODDS_COLUMNS.values()) + ["h2h_draw_avg"]
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
    Calculate edge using the same logic as match_scheduler.py.

    Main markets (home_win, away_win): edge = prob - implied_prob (vig-removed)
    Totals/BTTS (over25, under25, btts): edge = prob - implied_prob or prob - 0.5
    Niche (fouls, shots, corners, cards): edge = (prob - 0.5) * 2
    """
    # Niche markets: simple scaling
    if market in ("fouls", "shots", "corners", "cards"):
        if prob > 0.5:
            return (prob - 0.5) * 2
        return 0.0

    odds_col = MARKET_ODDS_COLUMNS.get(market)
    market_odds = match_odds.get(odds_col) if odds_col else None

    if market in ("home_win", "away_win"):
        if market_odds:
            # Calculate vig-removed implied probability
            home_col_odds = match_odds.get("h2h_home_avg")
            away_col_odds = match_odds.get("h2h_away_avg")
            if home_col_odds and away_col_odds:
                draw_odds = match_odds.get("h2h_draw_avg")
                total_implied = 1 / home_col_odds + 1 / away_col_odds + (
                    1 / draw_odds if draw_odds and draw_odds > 1.0 else 0.28
                )
                implied = (1 / market_odds) / total_implied
                return prob - implied
        # Fallback to baseline
        return prob - MARKET_BASELINES.get(market, 0.5)

    # over25, under25, btts
    if market_odds:
        implied = 1.0 / market_odds
        return prob - implied
    return prob - MARKET_BASELINES.get(market, 0.5)


def generate_sniper_predictions(
    matches: List[Dict],
    min_edge_pct: float = 5.0,
) -> List[Dict]:
    """
    Generate predictions using pre-trained sniper models.

    Replicates the prediction logic from match_scheduler --predict but outputs
    to the recommendation CSV format.
    """
    enabled_markets = load_sniper_config()
    if not enabled_markets:
        logger.error("No enabled markets in sniper deployment config")
        return []

    # Initialize model loader and feature lookup
    model_loader = ModelLoader()
    feature_lookup = FeatureLookup()

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

        # Get odds for this match (includes draw odds for vig removal)
        match_odds = get_match_odds(odds_df, home_team, away_team, fixture_id=fixture_id)

        # Run each enabled market's models
        for market_name, market_config in enabled_markets.items():
            threshold = market_config.get("threshold", 0.5)
            model_type = market_config.get("model", "").lower()
            saved_models = market_config.get("saved_models", [])

            # Determine which model names to try
            model_names_to_try = []

            # Niche markets use specific line models
            if market_name in ("fouls", "shots", "corners", "cards"):
                # Find niche models from available models
                for m in available_models:
                    if m.startswith(market_name + "_"):
                        model_names_to_try.append(m)
            else:
                # Full optimization models: try all variants from saved_models
                for saved in saved_models:
                    name = saved.replace(".joblib", "")
                    if name in available_models:
                        model_names_to_try.append(name)

            if not model_names_to_try:
                logger.debug(f"No models available for {market_name}")
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

            # Use walkforward.best_model as primary model selection strategy
            wf_best = market_config.get("walkforward", {}).get("best_model", "").lower()

            if wf_best == "stacking" and len(model_probs) >= 2:
                # Average all base model probabilities (stacking approximation)
                prob = sum(p for _, p, _ in model_probs) / len(model_probs)
                best_model = "stacking"
                confidence = sum(c for _, _, c in model_probs) / len(model_probs)
            elif wf_best == "agreement" and len(model_probs) >= 2:
                # Majority vote: only proceed if majority of models agree on threshold
                agreeing = [(n, p, c) for n, p, c in model_probs if p >= threshold]
                if len(agreeing) < len(model_probs) / 2:
                    logger.info(
                        f"  {home_team} vs {away_team} | {market_name}: "
                        f"agreement failed ({len(agreeing)}/{len(model_probs)} agree, "
                        f"threshold={threshold:.2f})"
                    )
                    continue
                prob = sum(p for _, p, _ in agreeing) / len(agreeing)
                best_model = "agreement"
                confidence = sum(c for _, _, c in agreeing) / len(agreeing)
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

            if market_name in ("fouls", "shots", "corners", "cards"):
                # Use the specific niche model that produced the best edge
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
        'line', 'odds', 'probability', 'edge', 'edge_source', 'referee',
        'fixture_id', 'result', 'actual'
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

    print("\nTop 10 by Edge:")
    print("-" * 70)
    top10 = df.nlargest(10, 'edge')
    for _, row in top10.iterrows():
        match_str = f"{row['home_team'][:15]} vs {row['away_team'][:15]}"
        bet = f"{row['market']} {row['bet_type']}"
        if row['line']:
            bet += f" {row['line']}"
        print(f"  {row['date']} | {match_str:<32} | {bet:<20} | +{row['edge']:.1f}%")

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

    # Generate predictions using sniper models
    print(f"\nRunning sniper model predictions for {len(matches)} matches...")
    all_predictions = generate_sniper_predictions(matches, min_edge_pct=args.min_edge)

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
