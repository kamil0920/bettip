# EDA_PremierLeague_Season.py
"""
Jupyter-friendly exploratory data analysis for one Premier League season.

How to use:
1. Open a new Jupyter notebook (or JupyterLab).
2. Create a cell and paste the contents of this file, or save it as `EDA_PremierLeague_Season.py`
   and in a notebook cell run: `%run EDA_PremierLeague_Season.py --season 2025`.

This script is intentionally split into notebook-style cells ("# %%") so you can paste
cell-by-cell into a notebook.

Outputs inside the notebook: tables, printed summaries, and matplotlib plots. The script
also saves an `eda_report_<season>.md` and plots into the season folder if run as a script.

Dependencies: pandas, matplotlib, numpy, pathlib

"""

# %%
# Imports
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

# %%
# Helper functions

def load_parquet_optional(p: Path):
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            return None
    return None


def sanitize_numeric_series(s):
    return pd.to_numeric(s, errors='coerce')


# small plotting helper
def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

# %%
# Configuration / CLI (for %run) or adjust season variable in notebook
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, required=True, help='Season year (e.g. 2025)')
    ap.add_argument('--base-dir', type=str, default='data/seasons')
    ap.add_argument('--save-report', action='store_true', help='Write markdown report + plots to disk')
    ap.add_argument('--out-dir', type=str, default=None, help='Report output dir (defaults to season/eda_report)')
    args = ap.parse_args()
    SEASON = args.season
    BASE_DIR = Path(args.base_dir) / str(SEASON)
    SAVE_REPORT = args.save_report
    OUT_DIR = Path(args.out_dir) if args.out_dir else BASE_DIR / 'eda_report'
else:
    # when pasted into a notebook, set SEASON variable manually
    SEASON = 2025
    BASE_DIR = Path('data/seasons') / str(SEASON)
    SAVE_REPORT = False
    OUT_DIR = BASE_DIR / 'eda_report'

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Season folder: {BASE_DIR.resolve()}")

# %%
# Load normalized tables
matches_fp = BASE_DIR / 'matches.parquet'
players_fp = BASE_DIR / 'player_stats.parquet'
events_fp = BASE_DIR / 'events.parquet'
teams_fp = BASE_DIR / 'teams.parquet'

matches = load_parquet_optional(matches_fp)
players = load_parquet_optional(players_fp)
events = load_parquet_optional(events_fp)
teams = load_parquet_optional(teams_fp)

print('Loaded:')
print(' matches:', getattr(matches, 'shape', None))
print(' players:', getattr(players, 'shape', None))
print(' events :', getattr(events, 'shape', None))
print(' teams :', getattr(teams, 'shape', None))

# %%
# Basic checks: existence and primary keys
if matches is None:
    raise SystemExit('matches.parquet not found — run normalization first')

print('\n-- matches head --')
display(matches.head())

if 'fixture_id' in matches.columns:
    print('Unique fixtures:', matches['fixture_id'].nunique(), 'rows:', len(matches))
else:
    print('Warning: fixture_id column missing in matches')

# %%
# Basic score statistics and distribution
for col in ['ft_home','ft_away','ht_home','ht_away']:
    if col not in matches.columns:
        matches[col] = np.nan

# coerce numeric
matches['ft_home'] = sanitize_numeric_series(matches['ft_home'])
matches['ft_away'] = sanitize_numeric_series(matches['ft_away'])

matches['total_goals'] = matches['ft_home'].fillna(0) + matches['ft_away'].fillna(0)
matches['result'] = matches.apply(lambda r: 'home' if r['ft_home']>r['ft_away'] else ('away' if r['ft_home']<r['ft_away'] else 'draw'), axis=1)

print('\nMatches summary:')
print(' mean total goals:', matches['total_goals'].mean())
print(' median total goals:', matches['total_goals'].median())
print(' home wins:', (matches['result']=='home').sum())
print(' draws:', (matches['result']=='draw').sum())
print(' away wins:', (matches['result']=='away').sum())

# histogram total goals
fig = plt.figure(figsize=(6,4))
matches['total_goals'].hist(bins=range(0,11))
plt.title(f'Total goals per match — {SEASON}')
plt.xlabel('Total goals')
plt.ylabel('Count')
plt.show()

if SAVE_REPORT:
    save_plot(fig, PLOTS_DIR / f'hist_total_goals_{SEASON}.png')

# %%
# Home vs Away goals distributions
fig = plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
matches['ft_home'].dropna().astype(float).hist(bins=range(0,8))
plt.title('Home goals')
plt.subplot(1,2,2)
matches['ft_away'].dropna().astype(float).hist(bins=range(0,8))
plt.suptitle(f'Home vs Away goals — {SEASON}')
plt.show()
if SAVE_REPORT:
    save_plot(fig, PLOTS_DIR / f'home_away_goals_{SEASON}.png')

# %%
# Events: goals by minute (if events available)
if events is not None:
    print('\n-- events sample --')
    display(events.head())
    # try to coerce minute
    if 'minute' in events.columns:
        events['minute_num'] = pd.to_numeric(events['minute'], errors='coerce')
    else:
        # some responses store minute in nested raw JSON; try `raw` or `time`
        if 'raw' in events.columns:
            # try parse JSON if raw is JSON string
            try:
                sample = events['raw'].dropna().iloc[0]
                # only attempt if string
                if isinstance(sample, str):
                    parsed = events['raw'].dropna().apply(lambda s: json.loads(s) if isinstance(s,str) else s)
                    # parsed may be dicts with 'time' etc. Very dataset-specific; only minimal attempt here.
            except Exception:
                pass

    # heuristic for goal events
    if 'type' in events.columns or 'detail' in events.columns:
        tcol = 'type' if 'type' in events.columns else 'detail'
        mask = events[tcol].astype(str).str.contains('Goal', na=False) | events.get('detail', '').astype(str).str.contains('Goal', na=False)
        goals = events[mask]
        if not goals.empty and 'minute_num' in goals.columns:
            gm = goals['minute_num'].dropna().astype(int)
            fig = plt.figure(figsize=(8,3))
            gm.hist(bins=range(0,100))
            plt.title('Goals by minute (sample)')
            plt.xlabel('Minute')
            plt.show()
            if SAVE_REPORT:
                save_plot(fig, PLOTS_DIR / f'goals_by_minute_{SEASON}.png')

# %%
# Top scorers from player_stats
if players is not None:
    print('\n-- player_stats sample --')
    display(players.head())
    if {'player_id','goals'}.issubset(players.columns):
        agg = players.groupby(['player_id','player_name'], dropna=False)['goals'].sum().reset_index().sort_values('goals', ascending=False)
        print('\nTop scorers (by player_stats):')
        display(agg.head(15))
        # bar chart
        top10 = agg.head(10)
        fig = plt.figure(figsize=(8,4))
        plt.bar(range(len(top10)), top10['goals'].astype(float))
        plt.xticks(range(len(top10)), [str(x) for x in (top10['player_name'].fillna(top10['player_id']))], rotation=45, ha='right')
        plt.title('Top 10 goal scorers')
        plt.show()
        if SAVE_REPORT:
            save_plot(fig, PLOTS_DIR / f'top_scorers_{SEASON}.png')
    else:
        print('player_stats missing player_id/goals columns — cannot compute top scorers')

# %%
# Cards summary
print('\n-- disciplinary (cards)')
if events is not None:
    mask_card = events['type'].astype(str).str.contains('Card|Yellow|Red|card', na=False) | events.get('detail','').astype(str).str.contains('Yellow|Red|card', na=False)
    print('Cards found in events (heuristic):', mask_card.sum())
else:
    print('No events file to analyze cards')

if players is not None:
    ycol = next((c for c in players.columns if 'yellow' in c.lower()), None)
    rcol = next((c for c in players.columns if 'red' in c.lower()), None)
    if ycol:
        print('Total yellow cards (player stats):', players[ycol].fillna(0).astype(float).sum())
    if rcol:
        print('Total red cards (player stats):', players[rcol].fillna(0).astype(float).sum())

# %%
# Teams sample and join check
if teams is not None:
    print('\n-- teams sample --')
    display(teams.head())
    # check join keys
    if 'team.id' in teams.columns:
        print('Team ids count:', teams['team.id'].nunique())
    # attempt to left-join matches to team names
    if 'home_team_id' in matches.columns:
        # find sample mapping
        id_to_name = None
        if 'team.id' in teams.columns and 'team.name' in teams.columns:
            id_to_name = teams.set_index('team.id')['team.name'].to_dict()
            matches['home_name'] = matches['home_team_id'].map(id_to_name)
            matches['away_name'] = matches['away_team_id'].map(id_to_name)
            display(matches[['fixture_id','home_team_id','home_name','away_team_id','away_name']].head())

# %%
# Missingness and data quality checks
print('\n-- missingness (matches) --')
miss = matches.isnull().mean().sort_values(ascending=False)
print(miss.head(20))

# %%
# Optional: write a short markdown summary file and save plots
if SAVE_REPORT:
    md_lines = [f"# EDA report — {SEASON}", '']
    md_lines.append('Summary stats:')
    md_lines.append(f"- matches rows: {len(matches)}")
    md_lines.append(f"- players rows: {len(players) if players is not None else 'N/A'}")
    md_lines.append(f"- events rows: {len(events) if events is not None else 'N/A'}")
    md_lines.append('')
    (OUT_DIR / f'eda_report_{SEASON}.md').write_text('\n'.join(md_lines), encoding='utf-8')
    print('Wrote report and plots to', OUT_DIR)

# %%
# End
print('\nEDA complete — inspect plots and dataframes above.')
