#!/usr/bin/env python3
"""
Performance Dashboard Generator

Generates an HTML dashboard showing betting performance metrics.

Usage:
    python experiments/generate_dashboard.py
    python experiments/generate_dashboard.py --output docs/dashboard.html
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_all_recommendations() -> pd.DataFrame:
    """Load all recommendation files."""
    rec_dir = project_root / 'data/05-recommendations'
    all_dfs = []

    for file in rec_dir.glob('rec_*.csv'):
        try:
            df = pd.read_csv(file)
            df['source_file'] = file.name
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """Calculate overall performance metrics."""
    if df.empty:
        return {}

    # Filter to settled bets only
    settled = df[df['result'].isin(['WON', 'LOST'])]
    if settled.empty:
        return {'total_bets': len(df), 'pending': len(df), 'settled': 0}

    wins = len(settled[settled['result'] == 'WON'])
    losses = len(settled[settled['result'] == 'LOST'])
    win_rate = wins / len(settled) if len(settled) > 0 else 0

    # Calculate profit
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
        'win_rate': win_rate * 100,
        'profit': profit,
        'roi': roi,
        'avg_odds': settled['odds'].mean(),
        'avg_edge': settled['edge'].mean(),
    }


def calculate_metrics_by_market(df: pd.DataFrame) -> Dict:
    """Calculate metrics grouped by market."""
    if df.empty:
        return {}

    settled = df[df['result'].isin(['WON', 'LOST'])]
    if settled.empty:
        return {}

    metrics = {}
    for market in settled['market'].unique():
        market_df = settled[settled['market'] == market]
        wins = len(market_df[market_df['result'] == 'WON'])

        profit = 0
        for _, row in market_df.iterrows():
            if row['result'] == 'WON':
                profit += (row['odds'] - 1)
            else:
                profit -= 1

        roi = (profit / len(market_df)) * 100 if len(market_df) > 0 else 0

        metrics[market] = {
            'bets': len(market_df),
            'wins': wins,
            'losses': len(market_df) - wins,
            'win_rate': (wins / len(market_df)) * 100 if len(market_df) > 0 else 0,
            'profit': profit,
            'roi': roi,
        }

    return metrics


def calculate_weekly_performance(df: pd.DataFrame) -> List[Dict]:
    """Calculate weekly performance trend."""
    if df.empty:
        return []

    settled = df[df['result'].isin(['WON', 'LOST'])].copy()
    if settled.empty:
        return []

    settled['date'] = pd.to_datetime(settled['date'])
    settled['week'] = settled['date'].dt.strftime('%Y-W%V')

    weekly = []
    for week in sorted(settled['week'].unique()):
        week_df = settled[settled['week'] == week]
        wins = len(week_df[week_df['result'] == 'WON'])

        profit = 0
        for _, row in week_df.iterrows():
            if row['result'] == 'WON':
                profit += (row['odds'] - 1)
            else:
                profit -= 1

        roi = (profit / len(week_df)) * 100 if len(week_df) > 0 else 0

        weekly.append({
            'week': week,
            'bets': len(week_df),
            'wins': wins,
            'win_rate': (wins / len(week_df)) * 100,
            'profit': profit,
            'roi': roi,
        })

    return weekly


def get_recent_bets(df: pd.DataFrame, n: int = 20) -> List[Dict]:
    """Get most recent settled bets."""
    if df.empty:
        return []

    settled = df[df['result'].isin(['WON', 'LOST'])].copy()
    if settled.empty:
        return []

    settled['date'] = pd.to_datetime(settled['date'])
    settled = settled.sort_values('date', ascending=False).head(n)

    bets = []
    for _, row in settled.iterrows():
        bets.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'match': f"{row['home_team']} vs {row['away_team']}",
            'market': row['market'],
            'bet': f"{row['bet_type']} {row.get('line', '')}".strip(),
            'odds': row['odds'],
            'edge': row['edge'],
            'result': row['result'],
            'actual': row.get('actual', ''),
        })

    return bets


def generate_html(metrics: Dict, market_metrics: Dict, weekly: List[Dict],
                  recent_bets: List[Dict]) -> str:
    """Generate HTML dashboard."""

    # Calculate cumulative profit for chart
    cumulative = []
    running_profit = 0
    for week in weekly:
        running_profit += week['profit']
        cumulative.append(running_profit)

    weeks_json = json.dumps([w['week'] for w in weekly])
    weekly_roi_json = json.dumps([w['roi'] for w in weekly])
    cumulative_json = json.dumps(cumulative)

    # Market performance for chart
    market_names = json.dumps(list(market_metrics.keys()))
    market_rois = json.dumps([m['roi'] for m in market_metrics.values()])
    market_bets = json.dumps([m['bets'] for m in market_metrics.values()])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BetTip Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
            color: #00d4ff;
            font-size: 2.5em;
        }}
        .updated {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #888;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        .neutral {{ color: #00d4ff; }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .chart-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .chart-title {{
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        .market-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 40px;
        }}
        .market-table th, .market-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .market-table th {{
            background: rgba(0, 212, 255, 0.1);
            color: #00d4ff;
        }}
        .market-table tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        .recent-bets {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .bet-row {{
            display: grid;
            grid-template-columns: 100px 1fr 120px 120px 80px 80px 80px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.9em;
        }}
        .bet-header {{
            font-weight: bold;
            color: #00d4ff;
        }}
        .won {{ color: #00ff88; }}
        .lost {{ color: #ff4444; }}
        h2 {{
            color: #00d4ff;
            margin-bottom: 20px;
        }}
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            .bet-row {{
                grid-template-columns: 1fr;
                gap: 5px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BetTip Performance Dashboard</h1>
        <p class="updated">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value neutral">{metrics.get('settled', 0)}</div>
                <div class="metric-label">Settled Bets</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.get('win_rate', 0) > 52 else 'neutral'}">{metrics.get('win_rate', 0):.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.get('roi', 0) > 0 else 'negative'}">{metrics.get('roi', 0):+.1f}%</div>
                <div class="metric-label">ROI</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.get('profit', 0) > 0 else 'negative'}">{metrics.get('profit', 0):+.1f}u</div>
                <div class="metric-label">Profit (units)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{metrics.get('avg_odds', 0):.2f}</div>
                <div class="metric-label">Avg Odds</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{metrics.get('avg_edge', 0):.1f}%</div>
                <div class="metric-label">Avg Edge</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-card">
                <h3 class="chart-title">Cumulative Profit (units)</h3>
                <canvas id="profitChart"></canvas>
            </div>
            <div class="chart-card">
                <h3 class="chart-title">Weekly ROI %</h3>
                <canvas id="roiChart"></canvas>
            </div>
        </div>

        <h2>Performance by Market</h2>
        <table class="market-table">
            <thead>
                <tr>
                    <th>Market</th>
                    <th>Bets</th>
                    <th>Wins</th>
                    <th>Win Rate</th>
                    <th>Profit</th>
                    <th>ROI</th>
                </tr>
            </thead>
            <tbody>
"""

    for market, m in sorted(market_metrics.items(), key=lambda x: x[1]['roi'], reverse=True):
        roi_class = 'positive' if m['roi'] > 0 else 'negative'
        html += f"""                <tr>
                    <td>{market}</td>
                    <td>{m['bets']}</td>
                    <td>{m['wins']}</td>
                    <td>{m['win_rate']:.1f}%</td>
                    <td class="{roi_class}">{m['profit']:+.1f}u</td>
                    <td class="{roi_class}">{m['roi']:+.1f}%</td>
                </tr>
"""

    html += """            </tbody>
        </table>

        <h2>Recent Results</h2>
        <div class="recent-bets">
            <div class="bet-row bet-header">
                <div>Date</div>
                <div>Match</div>
                <div>Market</div>
                <div>Bet</div>
                <div>Odds</div>
                <div>Edge</div>
                <div>Result</div>
            </div>
"""

    for bet in recent_bets:
        result_class = 'won' if bet['result'] == 'WON' else 'lost'
        html += f"""            <div class="bet-row">
                <div>{bet['date']}</div>
                <div>{bet['match'][:40]}</div>
                <div>{bet['market']}</div>
                <div>{bet['bet']}</div>
                <div>{bet['odds']:.2f}</div>
                <div>{bet['edge']:.1f}%</div>
                <div class="{result_class}">{bet['result']}</div>
            </div>
"""

    html += f"""        </div>
    </div>

    <script>
        const weeks = {weeks_json};
        const weeklyRoi = {weekly_roi_json};
        const cumulative = {cumulative_json};
        const marketNames = {market_names};
        const marketRois = {market_rois};

        // Profit chart
        new Chart(document.getElementById('profitChart'), {{
            type: 'line',
            data: {{
                labels: weeks,
                datasets: [{{
                    label: 'Cumulative Profit',
                    data: cumulative,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }}
                }}
            }}
        }});

        // ROI chart
        new Chart(document.getElementById('roiChart'), {{
            type: 'bar',
            data: {{
                labels: weeks,
                datasets: [{{
                    label: 'Weekly ROI %',
                    data: weeklyRoi,
                    backgroundColor: weeklyRoi.map(v => v >= 0 ? '#00ff88' : '#ff4444'),
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    return html


def main():
    parser = argparse.ArgumentParser(description='Generate performance dashboard')
    parser.add_argument('--output', type=str, default='docs/dashboard.html',
                        help='Output HTML file path')
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING PERFORMANCE DASHBOARD")
    print("=" * 60)

    # Load data
    print("\nLoading recommendations...")
    df = load_all_recommendations()
    print(f"  Loaded {len(df)} total bets")

    if df.empty:
        print("No data to display")
        return 1

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(df)
    market_metrics = calculate_metrics_by_market(df)
    weekly = calculate_weekly_performance(df)
    recent_bets = get_recent_bets(df)

    print(f"  Settled bets: {metrics.get('settled', 0)}")
    print(f"  Win rate: {metrics.get('win_rate', 0):.1f}%")
    print(f"  ROI: {metrics.get('roi', 0):+.1f}%")
    print(f"  Markets: {len(market_metrics)}")

    # Generate HTML
    print("\nGenerating HTML...")
    html = generate_html(metrics, market_metrics, weekly, recent_bets)

    # Save
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    print(f"\nDashboard saved to: {output_path}")
    print(f"Open in browser: file://{output_path.absolute()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
