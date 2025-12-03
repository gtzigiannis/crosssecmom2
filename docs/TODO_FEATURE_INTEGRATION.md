# Feature Engineering Integration TODO

## Overview
Integrate V2 feature families into existing codebase. Target: 8 balanced feature families × 3 horizons.

---

## Step 1: Integrate V2 into `feature_engineering.py` ✅ COMPLETE
**Goal**: Add compute functions from `feature_engineering_v2.py` into `process_ticker()`

### Tasks:
- [x] **1.1** Copy `compute_volume_features()` into `feature_engineering.py`
- [x] **1.2** Copy `compute_liquidity_features()` into `feature_engineering.py`
- [x] **1.3** Copy `compute_structure_features()` into `feature_engineering.py`
- [x] **1.4** Call new functions from `process_ticker()` after existing features
- [x] **1.5** Run tests to verify no regression

**Result**: 120 features generated (up from ~95)

---

## Step 2: Expand `data_manager.py` with FRED ✅ COMPLETE
**Goal**: Add FRED macro/sentiment data download to `CrossSecMomDataManager`

### Tasks:
- [x] **2.1** Add FRED_SERIES constant with 12 series
- [x] **2.2** Add `_get_fred_client()` lazy initialization
- [x] **2.3** Add `load_or_download_fred_data()` method
- [x] **2.4** Add caching for FRED data (parquet files in fred/ subdirectory)

**Result**: FRED_SERIES defined, load_or_download_fred_data() ready

---

## Step 3: Expand `generate_interaction_features()` ✅ COMPLETE
**Goal**: Add new feature categories and cross-family interactions

### Tasks:
- [x] **3.1** Add `volume_features` category detection
- [x] **3.2** Add `liquidity_v2_features` category detection
- [x] **3.3** Add `structure_features` category detection
- [x] **3.4** Add cross-family interaction pairs (Type 6):
  - [x] momentum × volume
  - [x] momentum × liquidity
  - [x] volatility × liquidity
  - [x] momentum × structure
  - [x] volume × liquidity
  - [x] volatility × structure
  - [x] trend × volume

**Result**: V2 cross-family interactions added

---

## Step 4: Update Family Classification ✅ COMPLETE
**Goal**: Add FACTOR_GROUPS and classification utilities

### Tasks:
- [x] **4.1** Add `FACTOR_GROUPS` constant with 10 families
- [x] **4.2** Add `classify_feature_family()` function
- [x] **4.3** Add `get_family_features()` function
- [x] **4.4** Add `print_family_summary()` function

**Result**: Family classification working:
- momentum: 34 features
- volatility: 22 features
- trend: 21 features
- volume: 14 features (NEW)
- liquidity: 11 features (NEW)
- structure: 10 features (NEW)
- risk: 5 features

---

## Validation ✅ COMPLETE

- [x] All imports successful
- [x] FRED_SERIES has 12 series
- [x] Generated 120 features
- [x] V2 families present (volume, liquidity, structure)
- [x] NaN percentage acceptable (11.2%)

---

## Current Status
- **All 4 steps COMPLETE**
- Ready to commit and push
