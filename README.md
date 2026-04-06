# Altus Model v6 README

## 1) Purpose

`Python altus_model_v6.py` builds and evaluates a daily inflow forecasting pipeline for **next-30-day cumulative inflow** at Altus-Lugert.

Target definition:

- `inflow_next30(t) = sum(inflow from t+1 to t+30)`

This is a rolling 30-day horizon forecast, not a strict calendar-month total.

## 2) Main Capabilities

- Fixed year-based train/validation/test split.
- Daily feature engineering with lagged inflow, rainfall windows, and seasonal terms.
- Optional subbasin rainfall and soil-moisture features (area-weighted alignment).
- Multiple models:
- `Linear_DailyNext30`
- `Ridge_DailyNext30`
- `RF_DailyNext30`
- `RF_Weighted_DailyNext30`
- `RF_Regime_DailyNext30`
- `XGB_DailyNext30` (if `xgboost` installed)
- Flood-focused metrics and thresholds from train-only quantiles.
- Validation-based model selection using a flood-robust metric.
- Residual-based uncertainty intervals (90% and 50%) on test predictions.
- Daily and monthly-issuance (day-of-month filter) forecast outputs.

## 3) Script Location

- Script: `C:\Users\aalec\Desktop\Altus Project\Python altus_model_v6.py`
- This README: `C:\Users\aalec\Desktop\Altus Project\model_outputs_v5\README_altus_model_v6.md`

## 4) Required Inputs

### Core dataset

- `ALTU_ALL.csv` with columns:
- `year`
- `month`
- `day`
- `inflow adj`
- `rainfall inches (7A to Dam)`
- `rainfall inches (7A to BSN)`

### Subbasin datasets

- Rain file with columns:
- `date`
- `area_m2_used`
- `rain_mm_aw`
- Soil file with columns:
- `date`
- `area_m2_used`
- `sm_aw`

The script maps subbasins by exact area overlap and falls back to rank-based matching when precision differences prevent exact overlap.

## 5) Environment and Dependencies

Python packages used:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `xgboost` (optional)

Install example:

```powershell
pip install numpy pandas matplotlib scikit-learn xgboost
```

If `xgboost` is missing, the script still runs and skips XGBoost training.

## 6) Configuration

Edit constants near the top of `Python altus_model_v6.py`.

### Paths

- `DATA_PATH`
- `OUT_DIR`
- `SUBBASIN_RAIN_PATH`
- `SUBBASIN_SM_PATH`

### Time split

- `TRAIN_END` (inclusive)
- `VAL_END` (inclusive)
- Test is `year > VAL_END`

### Flood/event controls

- `FLOOD_QUANTILE`
- `EVENT_QUANTILE`
- `TAIL_FLOOD_WEIGHT`
- `TAIL_EVENT_WEIGHT`
- `REGIME_MIN_SAMPLES`
- `MODEL_SELECTION_METRIC` (default: `FloodMAE`)
- `FORCE_BEST_MODEL` (set to model name or `None` for auto)
- `ISSUE_DAY_OF_MONTH` (used for monthly issuance slice)

### Feature switch

The manually curated subbasin list is in:

- `SELECTED_SUBBASIN_FEATURES`

If this list resolves to zero available columns, the script raises an error.

## 7) Modeling Pipeline

1. Load `ALTU_ALL.csv`, normalize date fields, sort chronologically.
2. Build daily base series:
- basin rainfall (`rain_total`)
- dam rainfall (`rain_dam`)
- daily inflow (`inflow_day`)
3. Merge subbasin rain/soil features by `date_day`.
4. Engineer rolling features:
- rain windows (1/3/7/14/30/45 day)
- dam rain windows
- lagged inflow windows (`qin_*`, with leak-safe shift)
- seasonality (`doy_sin`, `doy_cos`)
5. Build target `inflow_next30` and `log1p` transform.
6. Split by year into train/val/test.
7. Compute flood/event thresholds from train only.
8. Fit imputer on train only, transform val/test.
9. Train candidate models.
10. Evaluate train/val/test metrics on original scale.
11. Select best model:
- forced (`FORCE_BEST_MODEL`) or
- validation ranking on `MODEL_SELECTION_METRIC` with fallback to MAE.
12. Build uncertainty bands from validation residual quantiles.
13. Generate test predictions and CI outputs (daily + monthly issue-day slice).
14. Save metrics and plots.

## 8) Metrics

Standard regression metrics:

- `MAE`
- `RMSE`
- `R2`

Flood/event metrics:

- `FloodThreshold`
- `FloodN`
- `FloodMAE`
- `FloodRMSE`
- `EventThreshold`
- `EventN`
- `EventRecall`
- `EventPrecision`

Notes:

- Flood/event thresholds are derived from train split only to avoid leakage.
- If no flood events exist in a split, flood metrics can be blank/NaN.

## 9) Outputs

All outputs are written to `OUT_DIR`.

### CSV artifacts

- `metrics_daily_next30.csv`
- `predictions_daily_next30_with_ci.csv`
- `predictions_monthly_issue_day{ISSUE_DAY_OF_MONTH}_next30_with_ci.csv`
- `metrics_issue_cadence_next30.csv`

### Plot artifacts

- `daily_inflow_timeseries.png`
- `test_timeline_daily_next30_ci90.png`
- `test_timeline_daily_next30_ci50.png`
- `test_timeline_monthly_day{ISSUE_DAY_OF_MONTH}_next30_ci90.png`
- `test_timeline_monthly_day{ISSUE_DAY_OF_MONTH}_next30_ci50.png`
- `rf_line_actual_vs_pred_daily_next30.png` (if RF available)
- `models_actual_daily_next30_test.png`

### Prediction column schema

- `date_day`
- `y_true_next30`
- `y_pred_next30`
- `y_lo90`
- `y_hi90`
- `y_lo50`
- `y_hi50`

## 10) How to Run

From PowerShell:

```powershell
python "C:\Users\aalec\Desktop\Altus Project\Python altus_model_v6.py"
```

Run completes when CSV and PNG artifacts are written to `OUT_DIR`.

## 11) Operational Assumptions

- Forecast is issued with access to same-day predictors used in features.
- Input dates are complete and correctly aligned.
- Subbasin files represent comparable geographic partitioning (handled with overlap/rank matching logic).
- The fixed split strategy is representative for intended deployment regime.

## 12) Known Risks and Caveats

- A single test period can under-represent flood behavior (`FloodN` may be zero).
- Subbasin rain features can be temporally unstable and may need pruning.
- Weighted/regime models can improve event sensitivity while degrading overall MAE.
- Residual CIs are empirical and assume validation residual behavior transfers to test/deployment.

## 13) Recommended Validation Practice

- Use multiple rolling year-based backtests rather than one holdout only.
- Compare models on both:
- aggregate fit (`MAE`, `RMSE`)
- event performance (`FloodMAE`, `EventRecall`, `EventPrecision`)
- Track CI calibration coverage for 90% and 50% intervals.

## 14) Troubleshooting

### Error: missing subbasin files

- Confirm `SUBBASIN_RAIN_PATH` and `SUBBASIN_SM_PATH`.
- Ensure fallback desktop paths exist if primary path is unavailable.

### Error: no overlapping area IDs

- Happens when rain/soil area precision differs and counts do not match.
- Ensure both files cover the same number of subbasins.

### Error: no selected subbasin features

- Update `SELECTED_SUBBASIN_FEATURES` to include columns present in current subbasin feature build.

### XGBoost not training

- Install `xgboost` or ignore if not needed; script runs without it.

## 15) Version Notes (v6)

Compared with earlier versions, v6 includes:

- Stronger flood-aware training/evaluation controls.
- Validation-driven robust model selection behavior.
- Tail-aware weighted RF and regime-blended RF options.
- Subbasin area mapping logic for precision mismatch resilience.
- Daily-to-monthly issue cadence outputs in one run.

