# Bettip Improvement Research Plan

**Created:** 2026-01-19
**Purpose:** Deep research into techniques and data sources that could improve prediction quality

---

## Executive Summary

Based on comprehensive codebase analysis, we identified **5 major improvement categories**:

| Category | Potential Impact | Effort | Priority |
|----------|-----------------|--------|----------|
| Data Integration Gaps | HIGH | Medium | 1 |
| Feature Interaction Analysis | HIGH | Low | 2 |
| Advanced Feature Selection | MEDIUM | Low | 3 |
| External Data Enrichment | HIGH | High | 4 |
| Model Architecture Improvements | MEDIUM | Medium | 5 |

---

## Category 1: Data Integration Gaps (HIGH PRIORITY)

### 1.1 Weather Data Integration
**Status:** Fetcher exists, NOT integrated into pipeline

**Current State:**
- `src/data_collection/weather_collector.py` - fully functional
- Open-Meteo API (free, no key required)
- Features available: temp, humidity, precipitation, wind, weather codes

**Research Questions:**
- Does weather affect total fouls? (rainy = more fouls?)
- Does weather affect corners? (windy = different corner patterns?)
- Does weather affect total goals? (extreme cold/heat?)

**Implementation Plan:**
```python
# In src/features/engineers/external.py
class WeatherFeatureEngineer:
    def transform(self, df):
        # Weather already has features - just need to merge!
        weather_features = [
            'weather_is_rainy', 'weather_heavy_rain',
            'weather_is_windy', 'weather_very_windy',
            'weather_extreme_cold', 'weather_extreme_hot',
            'weather_adverse_score'
        ]
        return df  # Merge with weather data
```

**Estimated Lift:** 2-5% improvement in niche markets (corners, fouls)

---

### 1.2 xG Data Integration
**Status:** Fetched from Understat, blocked by team name normalization

**Current State:**
- `src/data_collection/fetch_xg_data.py` - works
- xG data in `data/03-features/xg_data_understat.csv`
- Problem: "Manchester United" vs "Man Utd" vs "Man United"

**Research Questions:**
- Can we build a team name mapping table?
- Is fuzzy matching reliable enough?
- Would xG improve BTTS and Over/Under predictions?

**Implementation Plan:**
```python
# Create src/utils/team_normalizer.py
TEAM_NAME_MAPPING = {
    'understat': {
        'Manchester United': 'Man United',
        'Manchester City': 'Man City',
        'Atletico Madrid': 'Atl. Madrid',
        # ... complete mapping for all teams
    }
}

def normalize_team_name(name, source='understat'):
    return TEAM_NAME_MAPPING.get(source, {}).get(name, name)
```

**Estimated Lift:** 5-10% improvement in goal-based markets

---

### 1.3 Bundesliga/Ligue 1 Match Stats Collection
**Status:** MISSING - cannot build niche market predictions

**Current State:**
- API-Football HAS the data (same endpoint)
- Collection script doesn't fetch for these leagues
- Results: No corner/fouls/shots predictions for 2 major leagues

**Implementation Plan:**
```bash
# Modify src/data_collection/match_collector.py
# Add match_stats collection for Bundesliga and Ligue 1
python entrypoints/collect.py --league bundesliga --include-stats
python entrypoints/collect.py --league ligue_1 --include-stats
```

**Estimated Lift:** 40% more betting opportunities (2 more leagues)

---

## Category 2: Feature Interaction Analysis (HIGH PRIORITY)

### 2.1 XGBfir (XGBoost Feature Interaction Ranking)
**Status:** NOT implemented

**What it does:**
- Analyzes feature interactions in XGBoost models
- Ranks pairs/triplets of features by interaction strength
- Identifies non-linear relationships ML might miss

**Installation:**
```bash
pip install xgbfir
```

**Implementation:**
```python
import xgbfir

# After training XGBoost model
xgbfir.saveXgbFI(
    model,
    feature_names=feature_names,
    OutputXlsxFile='feature_interactions.xlsx'
)

# Analyze top interactions:
# - elo_diff * rest_days
# - ref_fouls_avg * home_avg_yellows
# - weather_is_rainy * expected_fouls
```

**Research Questions:**
- Which feature pairs have strongest interactions?
- Can we create explicit interaction features from top pairs?
- Does this improve fouls/corners predictions?

**Estimated Lift:** 3-8% through explicit interaction features

---

### 2.2 SHAP Interaction Values
**Status:** SHAP exists, but interaction values NOT used

**Current State:**
- `experiments/run_shap_analysis.py` uses TreeExplainer
- Only uses `shap_values`, not `shap_interaction_values`

**Enhancement:**
```python
import shap

explainer = shap.TreeExplainer(model)
shap_interaction_values = explainer.shap_interaction_values(X_test)

# Analyze strongest interactions
# shap_interaction_values shape: (n_samples, n_features, n_features)
interaction_matrix = np.abs(shap_interaction_values).mean(axis=0)
```

**Research Questions:**
- Do SHAP interactions align with xgbfir results?
- Which interactions are most important for each market?

---

### 2.3 Partial Dependence Plots (PDP)
**Status:** NOT implemented

**What it does:**
- Shows marginal effect of features on predictions
- Reveals non-linear relationships
- Identifies optimal feature ranges

**Implementation:**
```python
from sklearn.inspection import PartialDependenceDisplay

# For fouls model
PartialDependenceDisplay.from_estimator(
    model, X_test,
    features=['ref_fouls_avg', 'home_fouls_avg', 'away_fouls_avg'],
    kind='both'  # individual + average
)
```

**Research Questions:**
- What's the optimal referee fouls average threshold?
- Is there a non-linear relationship between team fouls and total fouls?

---

## Category 3: Advanced Feature Selection (MEDIUM PRIORITY)

### 3.1 Recursive Feature Elimination (RFE)
**Status:** NOT implemented

**Current State:**
- Using Boruta (random forest based)
- Using permutation importance
- Missing: systematic elimination approach

**Implementation:**
```python
from sklearn.feature_selection import RFECV

selector = RFECV(
    estimator=XGBClassifier(),
    step=1,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='neg_log_loss',
    min_features_to_select=10
)
selector.fit(X, y)

# Get optimal feature set
optimal_features = X.columns[selector.support_]
```

**Research Questions:**
- Does RFE find different features than Boruta?
- What's the optimal feature count for each market?

---

### 3.2 Mutual Information Feature Selection
**Status:** NOT implemented

**What it does:**
- Measures statistical dependency between features and target
- Captures non-linear relationships
- Works for both continuous and categorical

**Implementation:**
```python
from sklearn.feature_selection import mutual_info_classif, SelectKBest

selector = SelectKBest(mutual_info_classif, k=30)
X_selected = selector.fit_transform(X, y)

# Get scores
mi_scores = pd.Series(
    selector.scores_,
    index=X.columns
).sort_values(descending=True)
```

---

### 3.3 L1-Based Selection (Lasso)
**Status:** NOT implemented for feature selection

**What it does:**
- Uses L1 penalty to zero out irrelevant features
- Automatic feature selection during training
- Good for high-dimensional data

**Implementation:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(
    LogisticRegression(penalty='l1', solver='saga', C=0.1),
    threshold='median'
)
X_selected = selector.fit_transform(X, y)
```

---

## Category 4: External Data Enrichment (HIGH IMPACT, HIGH EFFORT)

### 4.1 Injury/Suspension Data
**Status:** NOT collected

**Sources:**
- Transfermarkt API (unofficial)
- FBRef injury tables
- Football-Lineups.com

**Impact:**
- Key player absence feature currently uses lineup data (reactive)
- With injury data, we could be proactive

**Research Questions:**
- How much does star player absence affect results?
- Can we get reliable pre-match injury lists?

---

### 4.2 Betting Market Sharp Money Indicators
**Status:** Partially implemented (line movement)

**Current State:**
- `src/features/engineers/odds.py` - LineMovementFeatureEngineer
- Tracks opening → closing line movement

**Enhancement:**
```python
# Additional sharp money indicators
class SharpMoneyFeatureEngineer:
    def transform(self, df):
        # Steam moves (rapid line changes)
        df['steam_move'] = df['line_change_speed'] > threshold

        # Reverse line movement (sharp vs public)
        df['reverse_move'] = (df['public_pct'] > 70) & (df['line_moved_against_public'])

        # Asian market signals
        df['asian_market_signal'] = self._detect_asian_signals(df)
```

---

### 4.3 Manager Tactics Data
**Status:** Coach feature exists, data unreliable

**Enhancement:**
- Track manager tactical tendencies (pressing, possession, counter)
- Track manager history with specific referees
- Track manager head-to-head records

---

### 4.4 Stadium/Venue Data
**Status:** NOT used

**Available Data:**
- Pitch dimensions (affects corner patterns)
- Altitude (affects stamina)
- Surface type (grass vs artificial)

---

## Category 5: Model Architecture Improvements (MEDIUM PRIORITY)

### 5.1 Multi-Level Stacking
**Status:** Single-level stacking implemented

**Enhancement:**
```
Level 0: XGBoost, LightGBM, CatBoost, RF
Level 1: Blend of Level 0 + new features from Level 0 predictions
Level 2: Final meta-learner
```

---

### 5.2 Target-Specific Ensembles
**Status:** Using same ensemble for all markets

**Enhancement:**
- FOULS: Emphasize referee features → CatBoost heavy
- CORNERS: Emphasize team attack features → XGBoost heavy
- BTTS: Emphasize defense features → LightGBM heavy

---

### 5.3 Bayesian Optimization for Ensemble Weights
**Status:** Using CV scores for weights

**Enhancement:**
```python
from scipy.optimize import minimize

def optimize_ensemble_weights(models, X_val, y_val):
    def objective(weights):
        weights = np.array(weights) / sum(weights)
        pred = sum(w * m.predict_proba(X_val)[:,1]
                   for w, m in zip(weights, models))
        return -roc_auc_score(y_val, pred)

    result = minimize(objective, [1]*len(models), method='SLSQP')
    return result.x / sum(result.x)
```

---

### 5.4 Neural Network for Feature Interactions
**Status:** NOT implemented

**Idea:**
- Use shallow NN to learn feature interactions
- Extract hidden layer as "interaction features"
- Feed to tree models

```python
from tensorflow import keras

# AutoEncoder for interaction learning
encoder = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),  # Interaction features
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(n_features)
])
```

---

## Category 6: Calibration Improvements

### 6.1 Per-League Calibration
**Status:** Only per-market calibration

**Research Questions:**
- Does Premier League need different calibration than La Liga?
- Are some leagues more predictable?

---

### 6.2 Temporal Calibration Drift
**Status:** Fixed calibration factors

**Enhancement:**
- Track calibration over time
- Auto-adjust factors when drift detected
- Seasonal adjustments (early season more unpredictable)

---

### 6.3 Conformal Prediction for Uncertainty
**Status:** NOT implemented

**What it does:**
- Provides prediction intervals with guaranteed coverage
- Better uncertainty quantification than point estimates

---

## Implementation Priority Matrix

### Phase 1: Quick Wins (1-2 days each)
| Task | Expected Lift | Effort |
|------|---------------|--------|
| Integrate weather data | 2-5% | Low |
| Add xgbfir analysis | 3-8% | Low |
| Add SHAP interactions | 2-5% | Low |
| Add PDP analysis | 2-3% | Low |
| Add RFE feature selection | 2-4% | Low |

### Phase 2: Medium Effort (3-5 days each)
| Task | Expected Lift | Effort |
|------|---------------|--------|
| Fix xG team name normalization | 5-10% | Medium |
| Collect Bundesliga/Ligue1 match_stats | 40% more markets | Medium |
| Implement mutual information selection | 2-4% | Medium |
| Target-specific ensembles | 3-5% | Medium |

### Phase 3: Major Investments (1-2 weeks each)
| Task | Expected Lift | Effort |
|------|---------------|--------|
| Injury/suspension data integration | 5-10% | High |
| Neural network interaction features | 3-8% | High |
| Per-league calibration system | 2-5% | High |
| Conformal prediction | Better risk management | High |

---

## Research Experiments to Run

### Experiment 1: Feature Interaction Discovery
```bash
# Run xgbfir on FOULS model
python experiments/run_xgbfir_analysis.py --market fouls

# Run SHAP interactions
python experiments/run_shap_analysis.py --market fouls --interactions

# Compare results
python experiments/compare_interaction_methods.py
```

### Experiment 2: Weather Impact Analysis
```bash
# Merge weather data
python experiments/merge_weather_features.py

# Train callibration with/without weather
python experiments/run_weather_ablation.py

# Analyze which markets benefit most
python experiments/analyze_weather_impact.py
```

### Experiment 3: xG Integration
```bash
# Build team name mapping
python experiments/build_team_name_mapping.py

# Merge xG data
python experiments/merge_xg_features.py

# Validate on BTTS and Over/Under
python experiments/validate_xg_features.py
```

### Experiment 4: Feature Selection Comparison
```bash
# Run all selection methods
python experiments/compare_feature_selection.py \
    --methods boruta rfe mutual_info lasso \
    --market fouls

# Find consensus features
python experiments/find_consensus_features.py
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| FOULS calibration gap | -1.0% | < 2% |
| CORNERS calibration gap | -8.2% | < 5% |
| SHOTS calibration gap | -5.6% | < 5% |
| BTTS calibration gap | -6.5% | < 5% |
| Overall ROI (paper trading) | TBD | > 5% |
| Hit rate (FOULS) | 82% (weekend) | > 70% sustained |

---

## Conclusion

**Is it worth pursuing these improvements?**

**YES**, because:
1. **Quick wins exist** - Weather integration, xgbfir, SHAP interactions are low-effort, high-potential
2. **Major data gaps** - xG and Bundesliga/Ligue1 stats are blocking significant value
3. **FOULS success shows potential** - 82% hit rate proves the approach works; improvements can extend to other markets
4. **Compounding effects** - Each small improvement multiplies with others

**Recommended Path:**
1. Start with xgbfir and SHAP interactions (understand current models better)
2. Integrate weather data (easiest data enhancement)
3. Fix xG normalization (biggest potential lift)
4. Collect missing league data (more betting opportunities)

---

*Last updated: 2026-01-19*
