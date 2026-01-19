# Bettip Data Pipeline Summary

**Created:** 2026-01-19
**Purpose:** Document what data we collect, how we process it, and identify improvement opportunities

---

## 1. Data Sources

### Primary: API-Football (api-sports.io)
| Endpoint | Data | Usage |
|----------|------|-------|
| `/fixtures` | Match results, dates, venue, referee | Core match data |
| `/fixtures/statistics` | Corners, shots, fouls, possession, passes | Niche market features |
| `/fixtures/lineups` | Starting XI, formations, substitutes | Lineup features |
| `/fixtures/players` | 60+ player stats per match | Team ratings, star player tracking |
| `/fixtures/events` | Goals, cards, substitutions with timestamps | Goal timing, discipline features |

### Secondary Sources
| Source | Status | Data | Gap |
|--------|--------|------|-----|
| **Understat** | ✅ **INTEGRATED** | xG data (2019-2025) | 99% coverage via team normalizer |
| **Open-Meteo** | Implemented | Weather (temp, wind, rain) | **NOT INTEGRATED** (API slow) |
| **football-data.co.uk** | Active | Historical odds (8 leagues) | Fully used |
| **Referee websites** | Partial | Referee assignments | Premier League only |

### League Coverage
| League | Matches | Player Stats | Match Stats | xG |
|--------|---------|--------------|-------------|-----|
| Premier League | Complete | Complete | Complete | ✅ Integrated |
| La Liga | Complete | Complete | Complete | ✅ Integrated |
| Serie A | Complete | Complete | Complete | ✅ Integrated |
| Bundesliga | Complete | Complete | **Partial** (2020,2025) | ✅ Integrated |
| Ligue 1 | Complete | Complete | **Partial** (2020-22,2025) | ✅ Integrated |

**Update (Jan 19):** xG data now integrated with 99% coverage. Bundesliga/Ligue 1 match_stats partially collected.

---

## 2. Feature Engineering Pipeline

### 26 Feature Engineers (9 Categories)

#### Form Features (5 engineers)
| Engineer | Features | Window |
|----------|----------|--------|
| TeamFormFeatureEngineer | wins, draws, losses, goals, points | Last 5 matches |
| ExponentialMovingAverageFeatureEngineer | goals_ema, points_ema | EMA span=5 |
| HomeAwayFormFeatureEngineer | home/away specific form | Last 5 venue-specific |
| StreakFeatureEngineer | win/loss/scoring streaks | Running count |
| DixonColesDecayFeatureEngineer | Time-weighted stats | Half-life=60 days |

#### Rating Features (3 engineers)
| Engineer | Features | Method |
|----------|----------|--------|
| ELORatingFeatureEngineer | elo, elo_diff, win_prob_elo | K=32, home_adv=100 |
| PoissonFeatureEngineer | xg_poisson, attack/defense_strength | 10-match lookback |
| TeamRatingFeatureEngineer | team_avg_rating | Player ratings aggregated |

#### Statistics Features (4 engineers)
| Engineer | Features | Source |
|----------|----------|--------|
| TeamStatsFeatureEngineer | shots_ema, passes_ema, tackles_ema | player_stats |
| GoalDifferenceFeatureEngineer | avg_goal_diff, total_goal_diff | Match results |
| GoalTimingFeatureEngineer | early_goal_rate, late_goal_rate | events |
| DisciplineFeatureEngineer | avg_yellows, avg_reds | events |

#### Context Features (5 engineers)
| Engineer | Features | Source |
|----------|----------|--------|
| RestDaysFeatureEngineer | rest_days, short_rest, long_rest | Match dates |
| LeaguePositionFeatureEngineer | league_position, ppg, season_gd | Running table |
| SeasonPhaseFeatureEngineer | season_phase (0/1/2), round_number | Match round |
| MatchImportanceFeatureEngineer | pts_to_leader, in_cl_race, relegation_zone | Standings |
| MatchOutcomeFeatureEngineer | **TARGETS**: home_win, away_win, total_goals | Results |

#### H2H & Context (2 engineers)
| Engineer | Features | Source |
|----------|----------|--------|
| HeadToHeadFeatureEngineer | h2h_home_wins, h2h_avg_goals | Last 3 H2H matches |
| DerbyFeatureEngineer | is_derby, is_rivalry | Hardcoded pairs |

#### Lineup Features (5 engineers)
| Engineer | Features | Source |
|----------|----------|--------|
| FormationFeatureEngineer | formation_attacking/defensive | lineups |
| CoachFeatureEngineer | coach_change_recent | lineups (placeholder) |
| LineupStabilityFeatureEngineer | lineup_stability | Starting XI overlap |
| StarPlayerFeatureEngineer | stars_playing, stars_ratio | Top 3 by rating |
| KeyPlayerAbsenceFeatureEngineer | key_players_missing | Top 5 by minutes |

#### External Features (2 engineers)
| Engineer | Features | Source |
|----------|----------|--------|
| RefereeFeatureEngineer | ref_cards_avg, ref_fouls_avg, ref_bias | Match history |
| WeatherFeatureEngineer | temp, precip, wind, humidity | **NOT INTEGRATED** |

#### Niche Market Features (4 engineers)
| Engineer | Features | Source |
|----------|----------|--------|
| CornerFeatureEngineer | corners_won_ema, expected_corners | match_stats |
| FoulsFeatureEngineer | fouls_committed_ema, expected_fouls | match_stats |
| CardsFeatureEngineer | cards_ema, expected_cards | events |
| ShotsFeatureEngineer | shots_ema, expected_shots | match_stats |

**Total: ~150-200 features generated**

---

## 3. ML Models & Techniques

### Base Models
| Model | Use Case | Tuning |
|-------|----------|--------|
| XGBoost | Primary classifier | Optuna, depth=2-6 |
| LightGBM | Fast training | Optuna, depth=3-7 |
| CatBoost | Categorical handling | Optuna, depth=4-8 |
| Random Forest | Baseline ensemble | depth=8-18, 300-700 trees |
| Logistic Regression | Linear baseline | L2 regularization |

### Ensemble Strategies
- **VotingClassifier**: Soft voting with probability averaging
- **StackingClassifier**: Meta-learner (LogReg) on base model predictions
- **Weighted Voting**: Weights from cross-validation scores

### Feature Selection Methods
| Method | Implementation | Status |
|--------|----------------|--------|
| Permutation Importance | sklearn (n_repeats=15) | Active |
| SHAP Analysis | TreeExplainer + summary plots | Active |
| Boruta | BorutaPy wrapper | Active (experiments) |
| Feature Ablation | Group removal testing | Active |
| Top-N Selection | Importance ranking | Active |
| **xgbfir Interactions** | XGBoost Feature Interaction | ✅ Active (Jan 19) |

### Feature Interaction Analysis (xgbfir) - Jan 19, 2026
| Market | Top Interaction | Gain |
|--------|-----------------|------|
| FOULS | `away_avg_yellows × odds_upset_potential` | 1061.6 |
| CORNERS | `away_shots × home_shots` | 3364.6 |
| SHOTS | `away_corners × home_corners` | 1883.5 |

**Key Insight:** Cross-market signals discovered - shots predict corners, corners predict shots.

### Calibration Methods
| Method | Purpose | Status |
|--------|---------|--------|
| Beta Calibration | Flexible 3-parameter | Active |
| Platt Scaling | Standard sigmoid | Active |
| Isotonic Regression | Non-parametric | Active |
| Temperature Scaling | Single parameter | Active |
| Market-Specific Factors | Per-market corrections | Active |

---

## 4. Current Calibration Factors (from paper trading)

| Market | Factor | Gap | Status |
|--------|--------|-----|--------|
| FOULS | 1.00 | -1.0% | Active - validated edge |
| AWAY_WIN | 0.97 | -2.2% | Active |
| SHOTS | 0.92 | -5.6% | Active |
| BTTS | 0.90 | -6.5% | Active |
| CORNERS | 0.88 | -8.2% | Active |
| HOME_WIN | 0.70 | N/A | Disabled |
| OVER_2.5 | 0.50 | N/A | Disabled |

---

## 5. Data We Have But DON'T Use

### Completely Unused
1. **Weather Data** - Fetcher exists, features generated, but NOT merged into pipeline
2. ~~**xG Data (Understat)**~~ - ✅ **NOW INTEGRATED** (Jan 19, 2026)
3. **Standings Tables** - Collected for PL but not used in features
4. **Coach Information** - Engineer exists but data extraction unreliable

### xG Features Added (10 features - PROPER PRE-MATCH)
**Note:** Original Understat xG was post-match (data leakage). Fixed Jan 19 with rolling averages.

| Feature | Description | Correlation w/Goals |
|---------|-------------|---------------------|
| `xg_home_attack_avg` | Home team rolling attack xG | - |
| `xg_home_defense_avg` | Home team rolling defense xG | - |
| `xg_away_attack_avg` | Away team rolling attack xG | - |
| `xg_away_defense_avg` | Away team rolling defense xG | - |
| `xg_home_expected` | Pre-match home xG estimate | 0.31 |
| `xg_away_expected` | Pre-match away xG estimate | - |
| `xg_total_expected` | Pre-match total xG | - |
| `xg_diff_expected` | Pre-match xG differential | - |
| `xg_btts_expected` | BTTS prob from expected xG | - |
| `xg_over25_expected` | Over 2.5 prob from expected xG | - |

**Validation Results (Jan 19):**
| Market | Log Loss Δ | Brier Δ | AUC Δ | Verdict |
|--------|------------|---------|-------|---------|
| BTTS | -0.03% | +0.05% | +0.12% | No change |
| OVER25 | +0.34% | +0.44% | +0.36% | Slight improvement |
| UNDER25 | +0.34% | +0.44% | +0.36% | Slight improvement |

### Partially Used
1. **Match Events** - Only goal timing + cards; NOT used for pressure/passing metrics
2. **Player Statistics** - Aggregated to team level; individual trends ignored
3. **Formation Data** - Collected but feature reliability is low
4. **Lineups** - Only formation detection; no injury/planned changes

### Available in API But Not Collected
- Offsides (in match_stats)
- Interceptions
- Clearances
- Dribbles attempted/won
- Aerial duels
- Through balls

---

## 6. Key Files Reference

```
src/data_collection/
├── match_collector.py       # API-Football main collector
├── fetch_xg_data.py         # Understat xG fetcher
├── weather_collector.py     # Open-Meteo weather
└── referee_fetcher.py       # Referee assignments

src/features/
├── engineers/               # 26 feature engineers
│   ├── form.py             # Form-based features
│   ├── ratings.py          # ELO, Poisson, team ratings
│   ├── context.py          # Rest days, position, importance
│   ├── niche.py            # Corners, fouls, cards, shots
│   └── ...
└── registry.py              # Feature engineer factory

src/ml/
├── models.py               # Model factory
├── ensemble.py             # Stacking, voting
└── calibration.py          # Beta, Platt, Isotonic, Temperature

src/calibration/
└── market_calibrator.py    # Per-market calibration factors

data/
├── 01-raw/{league}/{season}/   # Raw API data
├── 02-preprocessed/{league}/   # Cleaned parquet files
├── 03-features/                # Final feature CSVs
├── 04-predictions/             # Model outputs
└── 05-recommendations/         # Betting recommendations
```

---

## 7. Quality Metrics

### Data Completeness
- **Matches**: 99%+ coverage for all leagues
- **Player Stats**: 95%+ (some early seasons incomplete)
- **Match Stats**: 100% for PL/LaLiga/SerieA, **0% for Bundesliga/Ligue1**
- **Odds**: 90%+ historical, 95%+ recent matches

### Model Performance (from backtests)
| Market | Predicted | Actual | Gap |
|--------|-----------|--------|-----|
| FOULS | 70% | 69% | -1% (well calibrated) |
| AWAY_WIN | 69% | 66.8% | -2.2% |
| SHOTS | 71% | 65.3% | -5.6% |
| CORNERS | 71% | 63% | -8.2% |
| BTTS | 67% | 60.5% | -6.5% |

### Validation Approach
- **Walk-forward**: Training on past, testing on future (no random shuffle)
- **Time-series splits**: 5-fold with temporal ordering
- **Leakage tests**: Automated in `tests/test_data_leakage.py`

---

*Last updated: 2026-01-19*
