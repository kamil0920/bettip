# Bettip Next Steps Plan

**Created:** 2026-01-19
**Status:** Active paper trading validation

---

## Current State

### Calibration Factors (from walk-forward backtest)
| Market | Factor | Gap | Status |
|--------|--------|-----|--------|
| FOULS | 1.00 | -1.0% | ✅ PRIMARY - validated edge |
| AWAY_WIN | 0.97 | -2.2% | ✅ Validating |
| SHOTS | 0.92 | -5.6% | ⚠️ Monitor |
| BTTS | 0.90 | -6.5% | ⚠️ Monitor |
| CORNERS | 0.88 | -8.2% | ⚠️ Monitor |
| HOME_WIN | 0.70 | N/A | ❌ Disabled |
| OVER_2.5 | 0.50 | N/A | ❌ Disabled |

### Active Predictions (Jan 19-25)
- FOULS: 12 bets (PRIMARY)
- AWAY_WIN: 10 bets
- SHOTS: 42 bets
- CORNERS: 24 bets

---

## Week 1: Jan 19-26 - Paper Trading Validation

### Daily Tasks
```bash
# Morning: Check for completed matches
python entrypoints/daily_pipeline.py --step settle

# Evening: Generate report
python entrypoints/daily_pipeline.py --step report
```

### End of Week Analysis
- [ ] Calculate hit rates per market
- [ ] Compare vs predicted probabilities
- [ ] Calculate calibration gaps
- [ ] Adjust factors if gap > 5%

### Success Criteria
| Market | Target Hit Rate | Minimum Bets |
|--------|-----------------|--------------|
| FOULS | > 65% | 10+ |
| AWAY_WIN | > 60% | 8+ |
| SHOTS | > 60% | 20+ |
| CORNERS | > 55% | 15+ |

---

## Week 2: Jan 26 - Feb 2 - Calibration Refinement

### Tasks
1. **Analyze Week 1 Results**
   - Calculate actual calibration gaps
   - Update factors: `new_factor = old_factor * (actual_rate / predicted_rate)`

2. **Regenerate Features** (if match data updated)
   ```bash
   python entrypoints/preprocess.py
   python entrypoints/features.py
   ```

3. **Retrain Models** (if >100 new matches)
   ```bash
   python experiments/run_full_optimization_pipeline.py --bet_type fouls
   ```

4. **Generate Week 2 Predictions**
   ```bash
   python entrypoints/daily_pipeline.py --step predict
   ```

---

## Week 3-4: Production Deployment

### Prerequisites
- [ ] 2 weeks of validated paper trading
- [ ] All markets show < 5% calibration gap
- [ ] FOULS maintains > 65% hit rate
- [ ] Positive expected ROI on at least 2 markets

### Deployment Steps

1. **Set Up Scheduled Automation**
   ```bash
   # Add to crontab
   0 8 * * * cd /home/kamil/projects/bettip && python entrypoints/daily_pipeline.py
   ```

2. **Configure Bankroll Management**
   - Starting bankroll: Define amount
   - Max stake per bet: 2% of bankroll
   - Kelly fraction: 0.25 (quarter Kelly)
   - Daily loss limit: 10%

3. **Enable Live Betting** (when ready)
   - Start with FOULS only (validated edge)
   - Add other markets after 1 week of live validation

---

## Risk Management Rules

### Position Sizing
- **Single bet max:** 2% of bankroll
- **Daily exposure max:** 10% of bankroll
- **Market correlation:** Max 3 bets on same match

### Stop Loss Rules
- **Daily:** Stop if down 10%
- **Weekly:** Reduce stakes 50% if down 15%
- **Monthly:** Review strategy if ROI < -5%

### Market Disabling Triggers
- Hit rate < 45% over 20+ bets
- Calibration gap > 15%
- 5 consecutive losses

---

## Key Commands Reference

```bash
# Full daily workflow
python entrypoints/daily_pipeline.py

# Individual steps
python entrypoints/daily_pipeline.py --step collect    # Fetch new data
python entrypoints/daily_pipeline.py --step settle     # Settle bets
python entrypoints/daily_pipeline.py --step predict    # New predictions
python entrypoints/daily_pipeline.py --step report     # Daily summary

# Paper trading scripts
python experiments/fouls_paper_trade.py status
python experiments/away_win_paper_trade.py status
python experiments/shots_paper_trade.py status
python experiments/corners_paper_trade.py status

# Backtest validation
python experiments/run_calibration_backtest.py --market ALL

# View recommendations
cat data/05-recommendations/rec_20260119_week.csv
```

---

## Decision Points

### After Week 1
| If... | Then... |
|-------|---------|
| FOULS hit rate > 70% | Consider live betting with small stakes |
| AWAY_WIN hit rate > 65% | Keep in validation |
| Any market < 45% | Disable and investigate |
| Calibration gap > 10% | Adjust factor immediately |

### After Week 2
| If... | Then... |
|-------|---------|
| 2+ markets validated | Begin production deployment |
| Only FOULS validated | Focus on FOULS, continue validating others |
| No markets validated | Review model architecture, consider retraining |

---

## Files to Monitor

| File | Purpose |
|------|---------|
| `experiments/outputs/fouls_tracking.json` | FOULS bet tracking |
| `experiments/outputs/away_win_tracking.json` | AWAY_WIN bet tracking |
| `data/05-recommendations/daily_report_*.json` | Daily summaries |
| `config/strategies.yaml` | Calibration config |
| `src/calibration/market_calibrator.py` | Calibration factors |

---

## Notes

- **FOULS is the priority** - only market with validated edge (81.8% weekend hit rate)
- **AWAY_WIN looks promising** - backtest shows only -2.2% gap
- **Goals markets (BTTS, OVER_2.5) are unreliable** - keep disabled
- **Collect data daily** to keep features current
