#!/usr/bin/env python3
"""
Daily Betting Pipeline - Automated daily workflow.

This script automates the daily betting workflow:
1. Collect new match data from API
2. Preprocess and update feature dataset
3. Generate calibrated predictions for upcoming matches
4. Track and settle completed bets
5. Update calibration factors based on results

Usage:
    # Full daily workflow
    python entrypoints/daily_pipeline.py

    # Specific steps
    python entrypoints/daily_pipeline.py --step collect
    python entrypoints/daily_pipeline.py --step predict
    python entrypoints/daily_pipeline.py --step settle
    python entrypoints/daily_pipeline.py --step report
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.calibration.market_calibrator import MarketCalibrator
from src.ml.clv_tracker import CLVTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DailyPipeline:
    """Orchestrates daily betting operations."""

    def __init__(self, config_path: str = "config/strategies.yaml"):
        self.config_path = project_root / config_path
        self.data_dir = project_root / "data"
        self.output_dir = project_root / "data/05-recommendations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize calibrator and CLV tracker
        self.calibrator = MarketCalibrator()
        self.clv_tracker = CLVTracker(
            output_dir=str(self.data_dir / "04-predictions" / "clv_tracking")
        )
        if self.config_path.exists():
            self.calibrator.load_config(self.config_path)

        # Tracking files
        self.tracking_files = {
            'fouls': project_root / 'experiments/outputs/fouls_tracking.json',
            'shots': project_root / 'experiments/outputs/shots_tracking.json',
            'corners': project_root / 'experiments/outputs/corners_tracking_v3.json',
            'away_win': project_root / 'experiments/outputs/away_win_tracking.json',
        }

    def run_full_pipeline(self) -> Dict:
        """Run the complete daily workflow."""
        logger.info("=" * 60)
        logger.info("DAILY BETTING PIPELINE")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("=" * 60)

        results = {}

        # Step 1: Collect new data
        logger.info("\n[1/6] Collecting new match data...")
        results['collect'] = self.collect_data()

        # Step 2: Preprocess
        logger.info("\n[2/6] Preprocessing data...")
        results['preprocess'] = self.preprocess_data()

        # Step 3: Settle completed bets
        logger.info("\n[3/6] Settling completed bets...")
        results['settle'] = self.settle_bets()

        # Step 4: Generate predictions
        logger.info("\n[4/6] Generating predictions...")
        results['predict'] = self.generate_predictions()

        # Step 5: Update CLV tracking
        logger.info("\n[5/6] Updating CLV tracking...")
        results['clv'] = self.update_clv_tracking()

        # Step 6: Generate report
        logger.info("\n[6/6] Generating daily report...")
        results['report'] = self.generate_report()

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        return results

    def collect_data(self) -> Dict:
        """Collect new match data from API."""
        try:
            from src.data_collection.match_collector import MatchCollector

            leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
            collected = 0

            for league in leagues:
                try:
                    collector = MatchCollector(league=league)
                    new_matches = collector.collect_recent()
                    collected += len(new_matches) if new_matches is not None else 0
                except Exception as e:
                    logger.warning(f"  Failed to collect {league}: {e}")

            logger.info(f"  Collected {collected} new matches")
            return {'status': 'success', 'matches': collected}

        except ImportError:
            logger.warning("  Match collector not available, skipping")
            return {'status': 'skipped', 'reason': 'collector not available'}
        except Exception as e:
            logger.error(f"  Collection failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def preprocess_data(self) -> Dict:
        """Run preprocessing pipeline."""
        try:
            import subprocess
            result = subprocess.run(
                ['uv', 'run', 'python', 'entrypoints/preprocess.py'],
                capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                logger.info("  Preprocessing completed")
                return {'status': 'success'}
            else:
                logger.warning(f"  Preprocessing issues: {result.stderr[:200]}")
                return {'status': 'warning', 'stderr': result.stderr[:500]}

        except subprocess.TimeoutExpired:
            logger.warning("  Preprocessing timeout")
            return {'status': 'timeout'}
        except Exception as e:
            logger.warning(f"  Preprocessing skipped: {e}")
            return {'status': 'skipped', 'reason': str(e)}

    def settle_bets(self) -> Dict:
        """Settle completed bets from tracking files."""
        results = {}

        for market, tracking_file in self.tracking_files.items():
            if not tracking_file.exists():
                continue

            try:
                with open(tracking_file, 'r') as f:
                    data = json.load(f)

                pending = [b for b in data.get('bets', []) if b.get('status') == 'pending']
                if not pending:
                    results[market] = {'pending': 0, 'settled': 0}
                    continue

                # Load match results
                settled_count = self._settle_market_bets(market, data, pending)
                results[market] = {'pending': len(pending), 'settled': settled_count}

                # Save updated data
                with open(tracking_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

                logger.info(f"  {market.upper()}: {settled_count}/{len(pending)} bets settled")

            except Exception as e:
                logger.warning(f"  Failed to settle {market}: {e}")
                results[market] = {'error': str(e)}

        return results

    def _settle_market_bets(self, market: str, data: Dict, pending: List) -> int:
        """Settle bets for a specific market."""
        # Load match stats
        all_stats = []
        for league in ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']:
            stats_file = self.data_dir / f'01-raw/{league}/2025/match_stats.parquet'
            if stats_file.exists():
                stats = pd.read_parquet(stats_file)
                all_stats.append(stats)

        if not all_stats:
            return 0

        stats_df = pd.concat(all_stats, ignore_index=True)

        # Market-specific stat column
        stat_map = {
            'fouls': ('home_fouls', 'away_fouls'),
            'shots': ('home_shots_total', 'away_shots_total'),
            'corners': ('home_corners', 'away_corners'),
        }

        if market in stat_map:
            home_col, away_col = stat_map[market]
            if home_col in stats_df.columns and away_col in stats_df.columns:
                stats_df['total_stat'] = stats_df[home_col] + stats_df[away_col]

        settled = 0
        for bet in pending:
            fixture_id = bet.get('fixture_id')
            match_stats = stats_df[stats_df['fixture_id'] == fixture_id]

            if len(match_stats) == 0:
                continue

            if market in stat_map and 'total_stat' in match_stats.columns:
                actual = match_stats.iloc[0]['total_stat']
                line = bet.get('line', 0)
                bet_type = bet.get('bet_type', 'OVER')

                if bet_type == 'OVER':
                    bet['won'] = actual > line
                else:
                    bet['won'] = actual < line

                bet['actual'] = actual
                bet['status'] = 'settled'
                settled += 1

            elif market == 'away_win':
                # Load match results
                matches_file = self.data_dir / '01-raw/premier_league/2025/matches.parquet'
                if matches_file.exists():
                    matches = pd.read_parquet(matches_file)
                    match = matches[matches['fixture.id'] == fixture_id]
                    if len(match) > 0 and match.iloc[0]['fixture.status.short'] == 'FT':
                        home_goals = match.iloc[0]['goals.home']
                        away_goals = match.iloc[0]['goals.away']
                        bet['won'] = away_goals > home_goals
                        bet['actual_result'] = f"{home_goals}-{away_goals}"
                        bet['status'] = 'settled'
                        settled += 1

        return settled

    def update_clv_tracking(self) -> Dict:
        """Record predictions and results in CLV tracker for edge validation."""
        recorded = 0
        settled = 0

        for market, tracking_file in self.tracking_files.items():
            if not tracking_file.exists():
                continue

            try:
                with open(tracking_file, 'r') as f:
                    data = json.load(f)

                for bet in data.get('bets', []):
                    match_id = str(bet.get('fixture_id', ''))
                    if not match_id:
                        continue

                    key = f"{match_id}_{market}"

                    # Record prediction if not already tracked
                    if key not in self.clv_tracker.predictions:
                        our_odds = bet.get('odds', 0)
                        our_prob = bet.get('probability', 0)
                        if our_odds > 0:
                            self.clv_tracker.record_prediction(
                                match_id=match_id,
                                home_team=bet.get('home_team', ''),
                                away_team=bet.get('away_team', ''),
                                match_date=bet.get('match_date', ''),
                                league=bet.get('league', ''),
                                bet_type=market,
                                our_probability=our_prob,
                                our_odds=our_odds,
                                market_odds=our_odds,
                            )
                            recorded += 1

                    # Record result if settled
                    if bet.get('status') == 'settled' and 'won' in bet:
                        try:
                            self.clv_tracker.record_result(
                                match_id=match_id,
                                bet_type=market,
                                won=bet['won'],
                            )
                            settled += 1
                        except Exception:
                            pass  # Already recorded or missing closing odds

            except Exception as e:
                logger.warning(f"  CLV tracking failed for {market}: {e}")

        self.clv_tracker.save_history()

        # Generate CLV summary if we have enough data
        summary = {}
        if self.clv_tracker.predictions:
            try:
                summary = self.clv_tracker.get_clv_summary()
                clv_file = self.data_dir / "04-predictions" / "clv_tracking" / "clv_summary.json"
                with open(clv_file, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                logger.info(f"  CLV summary saved: {clv_file}")
            except Exception as e:
                logger.warning(f"  CLV summary generation failed: {e}")

        logger.info(f"  Recorded {recorded} new predictions, {settled} results for CLV")
        return {'status': 'success', 'recorded': recorded, 'settled': settled, 'summary': summary}

    def generate_predictions(self) -> Dict:
        """Generate predictions for upcoming matches."""
        results = {}

        # Run paper trading scripts for each market
        scripts = {
            'fouls': 'experiments/fouls_paper_trade.py',
            'shots': 'experiments/shots_paper_trade.py',
            'corners': 'experiments/corners_paper_trade.py',
            'away_win': 'experiments/away_win_paper_trade.py',
        }

        for market, script in scripts.items():
            script_path = project_root / script
            if not script_path.exists():
                continue

            # Check if market is enabled
            if not self.calibrator.is_enabled(market.upper()):
                logger.info(f"  {market.upper()}: DISABLED in config, skipping")
                results[market] = {'status': 'disabled'}
                continue

            try:
                import subprocess
                result = subprocess.run(
                    ['uv', 'run', 'python', str(script_path), 'predict'],
                    capture_output=True, text=True, timeout=300,
                    cwd=str(project_root)
                )

                if result.returncode == 0:
                    # Count new predictions
                    output = result.stdout
                    new_count = output.count('[NEW]')
                    results[market] = {'status': 'success', 'new_predictions': new_count}
                    logger.info(f"  {market.upper()}: {new_count} new predictions")
                else:
                    results[market] = {'status': 'error', 'stderr': result.stderr[:200]}
                    logger.warning(f"  {market.upper()} failed: {result.stderr[:100]}")

            except subprocess.TimeoutExpired:
                results[market] = {'status': 'timeout'}
                logger.warning(f"  {market.upper()} timeout")
            except Exception as e:
                results[market] = {'status': 'error', 'error': str(e)}
                logger.warning(f"  {market.upper()} error: {e}")

        return results

    def generate_report(self) -> Dict:
        """Generate daily summary report."""
        report = {
            'date': datetime.now().isoformat(),
            'markets': {}
        }

        for market, tracking_file in self.tracking_files.items():
            if not tracking_file.exists():
                continue

            try:
                with open(tracking_file, 'r') as f:
                    data = json.load(f)

                bets = data.get('bets', [])
                pending = [b for b in bets if b.get('status') == 'pending']
                settled = [b for b in bets if b.get('status') == 'settled']

                if settled:
                    wins = sum(1 for b in settled if b.get('won'))
                    hit_rate = wins / len(settled)
                else:
                    wins = 0
                    hit_rate = 0

                report['markets'][market] = {
                    'total_bets': len(bets),
                    'pending': len(pending),
                    'settled': len(settled),
                    'wins': wins,
                    'hit_rate': hit_rate,
                    'calibration_factor': self.calibrator.get_calibration_factor(market.upper()),
                    'enabled': self.calibrator.is_enabled(market.upper()),
                }

            except Exception as e:
                report['markets'][market] = {'error': str(e)}

        # Print report
        logger.info("\n" + "-" * 60)
        logger.info("DAILY REPORT")
        logger.info("-" * 60)

        for market, stats in report['markets'].items():
            if 'error' in stats:
                continue

            status = "ENABLED" if stats['enabled'] else "DISABLED"
            logger.info(f"\n{market.upper()} ({status})")
            logger.info(f"  Pending: {stats['pending']}, Settled: {stats['settled']}")

            if stats['settled'] > 0:
                logger.info(f"  Wins: {stats['wins']}/{stats['settled']} ({stats['hit_rate']:.1%})")
                logger.info(f"  Calibration factor: {stats['calibration_factor']:.2f}")

        # Save report
        report_file = self.output_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\nReport saved: {report_file}")

        return report


def main():
    parser = argparse.ArgumentParser(description='Daily betting pipeline')
    parser.add_argument('--step', type=str, default='all',
                        help='Step to run: all, collect, preprocess, settle, predict, clv, report')
    args = parser.parse_args()

    pipeline = DailyPipeline()

    if args.step == 'all':
        pipeline.run_full_pipeline()
    elif args.step == 'collect':
        pipeline.collect_data()
    elif args.step == 'preprocess':
        pipeline.preprocess_data()
    elif args.step == 'settle':
        result = pipeline.settle_bets()
        print(json.dumps(result, indent=2))
    elif args.step == 'predict':
        result = pipeline.generate_predictions()
        print(json.dumps(result, indent=2))
    elif args.step == 'clv':
        result = pipeline.update_clv_tracking()
        print(json.dumps(result, indent=2, default=str))
    elif args.step == 'report':
        pipeline.generate_report()
    else:
        print(f"Unknown step: {args.step}")
        sys.exit(1)


if __name__ == "__main__":
    main()
