#!/usr/bin/env python3
"""
Telegram Bot for BetTip Alerts

Sends betting recommendations and results updates via Telegram.

Setup:
1. Create bot with @BotFather on Telegram
2. Get your chat ID by messaging @userinfobot
3. Set environment variables:
   - TELEGRAM_BOT_TOKEN: Bot token from BotFather
   - TELEGRAM_CHAT_ID: Your chat ID

Usage:
    python experiments/telegram_bot.py send-predictions    # Send today's picks
    python experiments/telegram_bot.py send-results        # Send recent results
    python experiments/telegram_bot.py send-summary        # Send portfolio summary
    python experiments/telegram_bot.py test                # Test connection
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class TelegramBot:
    """Simple Telegram bot for betting alerts."""

    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')

        if not self.token or not self.chat_id:
            print("Warning: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID not set")
            print("Set them in your environment or .env file")

    def send_message(self, text: str, parse_mode: str = 'Markdown') -> bool:
        """Send a message to the configured chat."""
        if not self.token or not self.chat_id:
            print(f"[DRY RUN] Would send:\n{text}")
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True,
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                print("Message sent successfully")
                return True
            else:
                print(f"Error sending message: {response.text}")
                return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def test_connection(self) -> bool:
        """Test the bot connection."""
        if not self.token:
            print("No token configured")
            return False

        url = f"https://api.telegram.org/bot{self.token}/getMe"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    bot_name = data['result'].get('username')
                    print(f"Connected to bot: @{bot_name}")
                    return True
            print(f"Connection failed: {response.text}")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False


def load_todays_predictions() -> List[Dict]:
    """Load today's predictions from recommendation files."""
    rec_dir = project_root / 'data/05-recommendations'
    today = datetime.now().strftime('%Y%m%d')

    predictions = []
    for file in rec_dir.glob(f'rec_{today}_*.csv'):
        try:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                predictions.append({
                    'date': row.get('date', ''),
                    'match': f"{row['home_team']} vs {row['away_team']}",
                    'market': row.get('market', ''),
                    'bet': f"{row.get('bet_type', '')} {row.get('line', '')}".strip(),
                    'odds': row.get('odds', 1.9),
                    'edge': row.get('edge', 0),
                    'league': row.get('league', ''),
                })
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return predictions


def load_recent_results(days: int = 3) -> List[Dict]:
    """Load recent settled results."""
    rec_dir = project_root / 'data/05-recommendations'
    cutoff = datetime.now() - timedelta(days=days)

    results = []
    for file in rec_dir.glob('rec_*.csv'):
        try:
            df = pd.read_csv(file)
            df = df[df['result'].isin(['WON', 'LOST'])]

            for _, row in df.iterrows():
                match_date = pd.to_datetime(row.get('date', ''))
                if pd.notna(match_date) and match_date >= cutoff:
                    results.append({
                        'date': str(row.get('date', ''))[:10],
                        'match': f"{row['home_team']} vs {row['away_team']}",
                        'market': row.get('market', ''),
                        'bet': f"{row.get('bet_type', '')} {row.get('line', '')}".strip(),
                        'odds': row.get('odds', 1.9),
                        'result': row.get('result', ''),
                        'actual': row.get('actual', ''),
                    })
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return results


def calculate_summary() -> Dict:
    """Calculate overall portfolio summary."""
    rec_dir = project_root / 'data/05-recommendations'

    all_bets = []
    for file in rec_dir.glob('rec_*.csv'):
        try:
            df = pd.read_csv(file)
            all_bets.append(df)
        except Exception:
            pass

    if not all_bets:
        return {}

    df = pd.concat(all_bets, ignore_index=True)
    settled = df[df['result'].isin(['WON', 'LOST'])]

    if settled.empty:
        return {
            'total_bets': len(df),
            'pending': len(df),
            'settled': 0,
        }

    wins = len(settled[settled['result'] == 'WON'])
    losses = len(settled[settled['result'] == 'LOST'])

    profit = 0
    for _, row in settled.iterrows():
        if row['result'] == 'WON':
            profit += (row['odds'] - 1)
        else:
            profit -= 1

    roi = (profit / len(settled)) * 100 if len(settled) > 0 else 0

    return {
        'total_bets': len(df),
        'pending': len(df[~df['result'].isin(['WON', 'LOST'])]),
        'settled': len(settled),
        'wins': wins,
        'losses': losses,
        'win_rate': (wins / len(settled)) * 100,
        'profit': profit,
        'roi': roi,
    }


def format_predictions_message(predictions: List[Dict]) -> str:
    """Format predictions for Telegram message."""
    if not predictions:
        return "No predictions for today."

    # Sort by edge
    predictions.sort(key=lambda x: x['edge'], reverse=True)

    lines = [
        f"*Daily Picks - {datetime.now().strftime('%Y-%m-%d')}*",
        f"_{len(predictions)} value bets found_",
        "",
    ]

    for i, p in enumerate(predictions[:15], 1):
        league_short = p['league'][:3].upper() if p['league'] else ''
        emoji = "" if p['edge'] >= 20 else ""

        lines.append(
            f"{i}. {emoji} *{p['match'][:35]}*\n"
            f"   {p['market']} {p['bet']} @ {p['odds']:.2f} (+{p['edge']:.0f}%)"
        )

    if len(predictions) > 15:
        lines.append(f"\n_...and {len(predictions) - 15} more_")

    lines.append(f"\n_Generated: {datetime.now().strftime('%H:%M')}_")

    return "\n".join(lines)


def format_results_message(results: List[Dict]) -> str:
    """Format results for Telegram message."""
    if not results:
        return "No recent results."

    lines = [
        "*Recent Results*",
        "",
    ]

    wins = sum(1 for r in results if r['result'] == 'WON')
    losses = len(results) - wins
    lines.append(f"_{wins}W - {losses}L ({wins/len(results)*100:.0f}% win rate)_")
    lines.append("")

    for r in results[:20]:
        emoji = "" if r['result'] == 'WON' else ""
        actual = f" ({r['actual']})" if r['actual'] else ""
        lines.append(
            f"{emoji} {r['match'][:25]}\n"
            f"   {r['market']} {r['bet']} @ {r['odds']:.2f}{actual}"
        )

    return "\n".join(lines)


def format_summary_message(summary: Dict) -> str:
    """Format portfolio summary for Telegram message."""
    if not summary:
        return "No data available."

    roi_emoji = "" if summary.get('roi', 0) > 0 else ""
    profit_sign = "+" if summary.get('profit', 0) > 0 else ""

    lines = [
        "*Portfolio Summary*",
        "",
        f" Total Bets: {summary.get('settled', 0)}",
        f" Win Rate: {summary.get('win_rate', 0):.1f}%",
        f"{roi_emoji} ROI: {summary.get('roi', 0):+.1f}%",
        f" Profit: {profit_sign}{summary.get('profit', 0):.1f} units",
        "",
        f"_Pending: {summary.get('pending', 0)} bets_",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Telegram bot for betting alerts')
    parser.add_argument('command', choices=['send-predictions', 'send-results', 'send-summary', 'test'],
                        help='Command to execute')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print message instead of sending')
    args = parser.parse_args()

    bot = TelegramBot()

    if args.dry_run:
        bot.token = None  # Force dry run

    if args.command == 'test':
        success = bot.test_connection()
        if success:
            bot.send_message(" Bot connection test successful!")
        return 0 if success else 1

    elif args.command == 'send-predictions':
        predictions = load_todays_predictions()
        message = format_predictions_message(predictions)
        bot.send_message(message)

    elif args.command == 'send-results':
        results = load_recent_results()
        message = format_results_message(results)
        bot.send_message(message)

    elif args.command == 'send-summary':
        summary = calculate_summary()
        message = format_summary_message(summary)
        bot.send_message(message)

    return 0


if __name__ == '__main__':
    sys.exit(main())
