#!/usr/bin/env python3
"""
Monte Carlo Stress Test for Betting Strategies.

Simulates thousands of bankroll paths using historical settled bets to estimate:
- Drawdown distribution
- Ruin probability
- Expected growth rate (Kelly vs flat)
- Bankroll fan chart

Usage:
    python scripts/monte_carlo_stress_test.py
    python scripts/monte_carlo_stress_test.py --n-sims 10000 --kelly-fraction 0.25
    python scripts/monte_carlo_stress_test.py --staking flat --plot
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_settled_bets(rec_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load historical settled bets from recommendation CSVs.

    Expects columns: date, odds, probability, edge, result (W/L/P).
    Filters to only settled bets (result is W or L).
    """
    if rec_dir is None:
        rec_dir = project_root / "data" / "05-recommendations"

    csv_files = sorted(rec_dir.glob("rec_*.csv"))
    if not csv_files:
        logger.error(f"No recommendation CSVs found in {rec_dir}")
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Keep only settled bets
    if "result" not in combined.columns:
        logger.error("No 'result' column in recommendation CSVs")
        return pd.DataFrame()

    settled = combined[combined["result"].isin(["W", "L"])].copy()
    settled = settled.sort_values("date").reset_index(drop=True)

    # Ensure numeric columns
    for col in ["odds", "probability", "edge"]:
        if col in settled.columns:
            settled[col] = pd.to_numeric(settled[col], errors="coerce")

    logger.info(f"Loaded {len(settled)} settled bets from {len(csv_files)} files")
    return settled


def simulate_betting_path(
    bets_df: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    max_stake_fraction: float = 0.05,
    staking: str = "kelly",
    n_sims: int = 5000,
    model_error_std: float = 0.05,
) -> np.ndarray:
    """
    Run Monte Carlo simulation of bankroll paths.

    For each simulation:
    1. For each historical bet, perturb the model probability with Beta noise
    2. Draw outcome from Bernoulli(p_true) instead of using actual result
    3. Size bet using Kelly or flat staking
    4. Track bankroll over time

    Args:
        bets_df: Settled bets with 'odds', 'probability' columns.
        initial_bankroll: Starting bankroll.
        kelly_fraction: Fraction of full Kelly to use.
        max_stake_fraction: Maximum stake as fraction of bankroll.
        staking: 'kelly' or 'flat'.
        n_sims: Number of simulation paths.
        model_error_std: Std dev of probability perturbation.

    Returns:
        np.ndarray of shape (n_sims, n_bets + 1) — bankroll paths.
    """
    n_bets = len(bets_df)
    odds = bets_df["odds"].values
    probs = bets_df["probability"].values

    # Pre-allocate bankroll paths: (n_sims, n_bets + 1)
    paths = np.zeros((n_sims, n_bets + 1))
    paths[:, 0] = initial_bankroll

    rng = np.random.default_rng(42)

    for sim in range(n_sims):
        bankroll = initial_bankroll

        for i in range(n_bets):
            if bankroll <= 0:
                paths[sim, i + 1:] = 0
                break

            p_model = probs[i]
            bet_odds = odds[i]

            if np.isnan(p_model) or np.isnan(bet_odds) or bet_odds <= 1.0:
                paths[sim, i + 1] = bankroll
                continue

            # Add model error: perturb probability
            p_true = np.clip(
                p_model + rng.normal(0, model_error_std), 0.01, 0.99
            )

            # Size the bet
            if staking == "kelly":
                full_kelly = (p_model * bet_odds - 1) / (bet_odds - 1)
                stake_frac = kelly_fraction * max(full_kelly, 0)
                stake_frac = min(stake_frac, max_stake_fraction)
            else:
                # Flat: 1% of initial bankroll
                stake_frac = 0.01

            stake = bankroll * stake_frac

            # Draw outcome
            won = rng.random() < p_true

            if won:
                bankroll += stake * (bet_odds - 1)
            else:
                bankroll -= stake

            paths[sim, i + 1] = bankroll

    return paths


def calculate_drawdown_distribution(paths: np.ndarray) -> Dict[str, float]:
    """
    Calculate max drawdown statistics across all simulation paths.

    Returns:
        Dict with median, p5, p95 max drawdown fractions.
    """
    n_sims = paths.shape[0]
    max_drawdowns = np.zeros(n_sims)

    for i in range(n_sims):
        path = paths[i]
        # Running peak
        peak = np.maximum.accumulate(path)
        # Drawdown = (peak - current) / peak
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peak > 0, (peak - path) / peak, 0)
        max_drawdowns[i] = np.max(dd)

    return {
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "p5_max_drawdown": float(np.percentile(max_drawdowns, 5)),
        "p95_max_drawdown": float(np.percentile(max_drawdowns, 95)),
        "mean_max_drawdown": float(np.mean(max_drawdowns)),
    }


def estimate_ruin_probability(
    paths: np.ndarray, ruin_threshold: float = 0.5
) -> float:
    """
    Estimate probability that bankroll drops below ruin_threshold * initial.

    Args:
        paths: Bankroll paths array (n_sims, n_steps).
        ruin_threshold: Fraction of initial bankroll that constitutes ruin.

    Returns:
        Estimated ruin probability [0, 1].
    """
    initial = paths[:, 0]
    ruin_level = initial * (1 - ruin_threshold)
    min_bankroll = np.min(paths, axis=1)
    n_ruined = np.sum(min_bankroll <= ruin_level)
    return float(n_ruined / len(paths))


def plot_fan_chart(paths: np.ndarray, output_path: Path) -> None:
    """
    Plot bankroll fan chart showing median and confidence bands.

    Args:
        paths: Bankroll paths (n_sims, n_steps).
        output_path: Where to save the PNG.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return

    n_steps = paths.shape[1]
    x = np.arange(n_steps)

    median = np.median(paths, axis=0)
    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(x, p5, p95, alpha=0.15, color="blue", label="5th-95th pctl")
    ax.fill_between(x, p25, p75, alpha=0.3, color="blue", label="25th-75th pctl")
    ax.plot(x, median, color="blue", linewidth=2, label="Median")
    ax.axhline(y=paths[0, 0], color="gray", linestyle="--", alpha=0.5, label="Initial")

    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Bankroll")
    ax.set_title("Monte Carlo Bankroll Simulation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fan chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo stress test for betting strategy")
    parser.add_argument("--n-sims", type=int, default=5000, help="Number of simulations (default: 5000)")
    parser.add_argument("--bankroll", type=float, default=1000, help="Initial bankroll (default: 1000)")
    parser.add_argument("--kelly-fraction", type=float, default=0.25, help="Kelly fraction (default: 0.25)")
    parser.add_argument("--max-stake", type=float, default=0.05, help="Max stake fraction (default: 0.05)")
    parser.add_argument("--staking", choices=["kelly", "flat"], default="kelly", help="Staking method")
    parser.add_argument("--model-error", type=float, default=0.05, help="Probability estimation error std")
    parser.add_argument("--ruin-threshold", type=float, default=0.50, help="Ruin = losing this fraction")
    parser.add_argument("--rec-dir", type=str, default=None, help="Path to recommendations directory")
    parser.add_argument("--plot", action="store_true", help="Generate bankroll fan chart")
    args = parser.parse_args()

    print("=" * 60)
    print("MONTE CARLO STRESS TEST")
    print(f"Simulations: {args.n_sims} | Staking: {args.staking}")
    print(f"Bankroll: {args.bankroll} | Kelly fraction: {args.kelly_fraction}")
    print("=" * 60)

    rec_path = Path(args.rec_dir) if args.rec_dir else None
    bets = load_settled_bets(rec_path)

    if bets.empty:
        print("\nNo settled bets found. Need recommendation CSVs with 'result' column (W/L).")
        return 1

    print(f"\nSettled bets: {len(bets)}")
    print(f"  Wins: {(bets['result'] == 'W').sum()}")
    print(f"  Losses: {(bets['result'] == 'L').sum()}")
    if "edge" in bets.columns:
        print(f"  Avg edge: {bets['edge'].mean():.1f}%")
    if "odds" in bets.columns:
        print(f"  Avg odds: {bets['odds'].mean():.2f}")

    # Run simulation
    print(f"\nRunning {args.n_sims} simulations...")
    paths = simulate_betting_path(
        bets,
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        max_stake_fraction=args.max_stake,
        staking=args.staking,
        n_sims=args.n_sims,
        model_error_std=args.model_error,
    )

    # Results
    final_bankrolls = paths[:, -1]
    initial = args.bankroll

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nFinal Bankroll Distribution:")
    print(f"  Median:  {np.median(final_bankrolls):.0f} ({(np.median(final_bankrolls)/initial - 1)*100:+.1f}%)")
    print(f"  Mean:    {np.mean(final_bankrolls):.0f} ({(np.mean(final_bankrolls)/initial - 1)*100:+.1f}%)")
    print(f"  5th pct: {np.percentile(final_bankrolls, 5):.0f} ({(np.percentile(final_bankrolls, 5)/initial - 1)*100:+.1f}%)")
    print(f"  95th pct:{np.percentile(final_bankrolls, 95):.0f} ({(np.percentile(final_bankrolls, 95)/initial - 1)*100:+.1f}%)")

    # Drawdown
    dd = calculate_drawdown_distribution(paths)
    print(f"\nMax Drawdown Distribution:")
    print(f"  Median: {dd['median_max_drawdown']*100:.1f}%")
    print(f"  Mean:   {dd['mean_max_drawdown']*100:.1f}%")
    print(f"  5th pct: {dd['p5_max_drawdown']*100:.1f}%")
    print(f"  95th pct: {dd['p95_max_drawdown']*100:.1f}%")

    # Ruin probability
    ruin_prob = estimate_ruin_probability(paths, args.ruin_threshold)
    print(f"\nRuin Probability (losing >{args.ruin_threshold*100:.0f}%): {ruin_prob*100:.2f}%")

    # Profit probability
    profit_prob = np.mean(final_bankrolls > initial)
    print(f"Profit Probability: {profit_prob*100:.1f}%")

    # Compare Kelly vs flat if running Kelly
    if args.staking == "kelly":
        print("\n--- Comparison: Kelly vs Flat ---")
        flat_paths = simulate_betting_path(
            bets,
            initial_bankroll=args.bankroll,
            staking="flat",
            n_sims=args.n_sims,
            model_error_std=args.model_error,
        )
        flat_final = flat_paths[:, -1]
        print(f"  Kelly median: {np.median(final_bankrolls):.0f}")
        print(f"  Flat  median: {np.median(flat_final):.0f}")
        kelly_growth = np.median(np.log(final_bankrolls / initial + 1e-10))
        flat_growth = np.median(np.log(flat_final / initial + 1e-10))
        print(f"  Kelly growth rate: {kelly_growth:.4f}")
        print(f"  Flat  growth rate: {flat_growth:.4f}")

    # Plot
    if args.plot:
        out = project_root / "data" / "mc_bankroll_fan_chart.png"
        plot_fan_chart(paths, out)

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
