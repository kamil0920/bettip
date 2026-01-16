# Data Quality & Enhancement Implementation Plan

Based on analysis of current codebase vs. recommended improvements.

## Priority Matrix

| Priority | Effort | Expected ROI | Focus |
|----------|--------|--------------|-------|
| P0 | Low | High | Quick wins - implement immediately |
| P1 | Medium | High | High-value, moderate effort |
| P2 | Medium | Medium | Valuable but needs validation |
| P3 | High | Uncertain | Research/experimental |

---

## P0: IMMEDIATE WINS (1-2 days each)

### 1. ✅ Data Quality Framework (DONE)
- [x] NaN analysis and reporting
- [x] League-specific imputation
- [x] Prediction confidence scoring
- [x] Unit tests

### 2. Betting Market Line Movements
**Why High Priority:** Document cites this as "strong moves are often predictive" - aggregates vast information.

**Current Gap:** We use closing odds but NOT:
- Opening vs closing line movement
- Implied probability shifts
- Steam moves (sharp money indicators)

**Implementation:**
```python
# New features to add
'odds_home_opening', 'odds_home_closing',
'line_movement_home',  # closing - opening implied prob
'line_movement_magnitude',  # absolute change
'reverse_line_movement',  # flag when line moves opposite to betting %
```

**Data Source:** Historical odds from football-data.co.uk already has opening/closing odds (B365H, B365A columns).

**Estimated Impact:** +5-15% ROI improvement based on cited research.

### 3. Dixon-Coles Time Decay Enhancement
**Why:** Document specifically recommends: "compute form metrics using exponentially weighted moving average with tunable decay"

**Current:** We have EMA but with fixed spans.

**Enhancement:**
- Add configurable half-life parameter
- Tune decay factor on validation data
- Apply to goals, xG, points, shots

```python
# Example: Half-life of 30 days means ~50% weight on last month
def dixon_coles_decay(values, dates, half_life_days=30):
    lambda_param = np.log(2) / half_life_days
    weights = np.exp(-lambda_param * days_since_match)
    return np.average(values, weights=weights)
```

### 4. Match Importance Features
**Why:** "Match importance (derby, relegation six-pointer, must-win) adds nuance"

**Implementation:**
```python
# New features
'is_derby',  # Based on rival team mapping
'standings_gap',  # Points difference in table
'relegation_battle',  # Both teams in bottom 5
'title_race',  # Both teams in top 3
'match_importance_score',  # Combined metric
```

**Data Available:** We have league standings from features.

---

## P1: HIGH VALUE (3-5 days each)

### 5. Injury/Suspension Integration
**Why:** "These clearly affect match odds. Even imperfect injury flags can improve forecasts."

**Challenge:** Inconsistent data, but worth the noise.

**Implementation Options:**
1. **Scraped data:** PhysioRoom, Transfermarkt injuries
2. **API data:** API-Football has injuries endpoint
3. **Simple approach:** Count missing key players from lineup data

```python
# Features to add
'home_injuries_count',
'away_injuries_count',
'home_key_player_out',  # Boolean: is top scorer/assist missing
'away_key_player_out',
'squad_strength_drop',  # Rating of missing players
```

**Priority:** API-Football already in use - check if injuries endpoint is available.

### 6. Lineup Strength Aggregation
**Why:** "Compute team strength by aggregating player-level ratings"

**Current Gap:** We have lineup data but don't aggregate player quality.

**Implementation:**
```python
# Use existing player ratings from lineups
'home_lineup_avg_rating',
'away_lineup_avg_rating',
'home_lineup_total_rating',
'away_lineup_total_rating',
'lineup_rating_diff',
'home_bench_strength',  # Subs quality
```

**Data Source:** We have `home_rating_ema`, `away_rating_ema` - need to verify these come from lineup data.

### 7. Advanced xG Features
**Why:** "More xG features (home/away xG balance, defensive xG conceded) often boost accuracy"

**Current:** We have basic Poisson xG.

**Enhancement:**
```python
# From StatsBomb open data or calculated
'home_xg_for_avg',
'away_xg_for_avg',
'home_xg_against_avg',  # Defensive xG conceded
'away_xg_against_avg',
'xg_differential',
'xg_overperformance',  # Goals - xG (luck factor)
'non_penalty_xg',
```

**Data Source:** StatsBomb open data (free) or calculate from shot data.

---

## P2: MODERATE VALUE (1-2 weeks each)

### 8. Team Style Clustering
**Why:** "Clustering teams on standardized stats revealed distinct playing styles"

**Implementation:**
```python
# Cluster features
team_profiles = df.groupby('team').agg({
    'possession': 'mean',
    'shots_per_game': 'mean',
    'fouls_per_game': 'mean',
    'corners_per_game': 'mean',
    'pass_accuracy': 'mean',
})

# KMeans clustering into archetypes
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
team_profiles['style_cluster'] = kmeans.fit_predict(scaled_features)

# Archetypes: "High Press", "Possession Giants", "Counter Attack", "Physical", "Balanced"
```

**Use Case:** Add cluster ID as categorical feature, capture style matchups.

### 9. Dimensionality Reduction (PCA)
**Why:** "With 220+ features, many will correlate. PCA can distill information."

**Current:** 253 columns, likely high multicollinearity.

**Implementation:**
```python
# Apply PCA to correlated feature groups
from sklearn.decomposition import PCA

# Group 1: Form features → 2-3 components
# Group 2: Attack features → 2-3 components
# Group 3: Defense features → 2-3 components

# New features: 'form_pc1', 'attack_pc1', 'defense_pc1', etc.
```

**Benefit:** Reduce noise, prevent overfitting, improve interpretability.

### 10. Sentiment Analysis Pilot
**Why:** "Fan sentiment surges can predict outcomes, higher returns than odds alone"

**Implementation:**
```python
# Pre-match sentiment features
'home_sentiment_score',  # -1 to +1
'away_sentiment_score',
'sentiment_diff',
'sentiment_volume',  # Tweet count as proxy for interest
```

**Data Source:** Twitter/X API (paid), or Reddit scraping (free).

**Caveat:** Document marks this "moderate value" - pilot first before full implementation.

---

## P3: EXPERIMENTAL/RESEARCH (Weeks-Months)

### 11. Synthetic Tracking Data
**Why:** "Simulate realistic player-tracking via environments like Google Research Football"

**Complexity:** High - requires RL environment setup, model training.

**Defer until:** Core improvements implemented and validated.

### 12. Causal Counterfactual Modeling
**Why:** "What if the red card hadn't happened?"

**Simpler Approach First:**
```python
# Instead of full causal modeling, use simple indicators
'first_half_red_card_history',  # Team tendency for early reds
'red_card_impact_factor',  # Historical goal swing after reds
```

**Full causal:** Research project, not immediate ROI.

### 13. Betfair Live Streaming
**Why:** "Large late moves in betting markets are reliable indicators"

**Challenge:** Requires Betfair API setup, real-time data pipeline.

**Simpler Alternative:** Use historical closing line value (CLV) tracking already mentioned in codebase.

---

## Implementation Roadmap

### Week 1: P0 Items
- [ ] Add line movement features from football-data.co.uk
- [ ] Implement Dixon-Coles configurable decay
- [ ] Add match importance features

### Week 2-3: P1 Items
- [ ] Integrate injury data from API-Football
- [ ] Implement lineup strength aggregation
- [ ] Add advanced xG features

### Week 4: P2 Pilots
- [ ] Team style clustering prototype
- [ ] PCA dimensionality reduction experiment
- [ ] Sentiment analysis feasibility study

### Ongoing: Validation
- Every new feature must pass walk-forward validation
- Track feature importance to prune weak signals
- Monitor for data leakage with existing tests

---

## Quick Wins to Implement First

1. **Line Movement (P0)** - Data already available, high impact
2. **Match Importance (P0)** - Easy to calculate from standings
3. **Dixon-Coles Decay (P0)** - Small code change, proven technique

---

## Enhancements to Data Quality Module

Based on document recommendations, enhance `src/features/data_quality.py`:

### Add Consistency Checks
```python
def check_feature_consistency(df, feature, expected_range):
    """Detect anomalies like implausible ELO jumps."""
    outliers = df[(df[feature] < expected_range[0]) | (df[feature] > expected_range[1])]
    return outliers

def check_team_name_alignment(df1, df2):
    """Ensure team names match across datasets."""
    names_df1 = set(df1['home_team'].unique()) | set(df1['away_team'].unique())
    names_df2 = set(df2['home_team'].unique()) | set(df2['away_team'].unique())
    mismatches = names_df1.symmetric_difference(names_df2)
    return mismatches
```

### Add Feature Drift Detection
```python
def detect_feature_drift(df, feature, window_size=100):
    """Check if feature distribution changes over time."""
    rolling_mean = df[feature].rolling(window_size).mean()
    rolling_std = df[feature].rolling(window_size).std()
    # Flag if current values are > 2 std from rolling mean
    drift = abs(df[feature] - rolling_mean) > 2 * rolling_std
    return drift
```

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| ROI (SHOTS) | +44% | +50% | Walk-forward backtest |
| ROI (FOULS) | +50% | +55% | Walk-forward backtest |
| Feature NaN Rate | 56% | <20% | Data quality report |
| Prediction Confidence | Mixed | >80% high-conf | Confidence scoring |
| CLV (Closing Line Value) | Unknown | Positive | Beat closing odds |

---

## References from Document

1. Dixon-Coles time weighting: [dashee87.github.io](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)
2. Betfair market analysis: [betfair-datascientists.github.io](https://betfair-datascientists.github.io/tutorials/analysingAndPredictingMarketMovements/)
3. Sentiment prediction: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167923616300835)
4. Synthetic tracking: [ResearchGate](https://www.researchgate.net/publication/390175898)
