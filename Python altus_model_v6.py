"""
Altus-Lugert Predictive Modeling (fixed-split only)
- Fixed train/val/test split
- Simple models (Linear, Ridge, optional RF/XGB)
- Robust CI bands (original-scale residual quantiles)
- Plots and metrics saved to OUT_DIR
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math import sqrt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Optional models
RF_AVAILABLE = False
XGB_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestRegressor
    RF_AVAILABLE = True
except Exception:
    pass
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    pass

# ---------------- CONFIG ----------------
DATA_PATH = Path(r"C:\Users\aalec\Desktop\Altus Project\ALTU_ALL.csv")
OUT_DIR   = Path(r"C:\Users\aalec\Desktop\Altus Project\model_outputs_v5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBBASIN_RAIN_PATH = Path(r"C:\Users\aalec\OneDrive - Oklahoma A and M System\Desktop\Altus Project\ERA5Land_daily_area_weighted_subbasin_rainfall.csv")
SUBBASIN_SM_PATH = Path(r"C:\Users\aalec\OneDrive - Oklahoma A and M System\Desktop\Altus Project\ERA5Land_daily_area_weighted_subbasin_soil_moisture.csv")
# Fixed split years
TRAIN_END = 2018 # inclusive
VAL_END   = 2023  # inclusive; test is > VAL_END

# Flood-robustness controls
FLOOD_QUANTILE = 0.90
EVENT_QUANTILE = 0.75
TAIL_FLOOD_WEIGHT = 3.0
TAIL_EVENT_WEIGHT = 1.8
REGIME_MIN_SAMPLES = 80
MODEL_SELECTION_METRIC = 'FloodMAE'  # fallback to MAE if unavailable
FORCE_BEST_MODEL = 'RF_DailyNext30'  # set to None to use automatic validation-based selection
ISSUE_DAY_OF_MONTH = 1


# Exact CSV headers
COL_DAY         = 'day'
COL_INFLOW      = 'inflow adj'
COL_RAIN_DAM    = 'rainfall inches (7A to Dam)'
COL_RAIN_BSN    = 'rainfall inches (7A to BSN)'
COL_MONTH       = 'month'
COL_YEAR        = 'year'

# ---------------- HELPERS ----------------
def extra_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # Nash-Sutcliffe Efficiency
    sse = float(np.sum((y_true - y_pred) ** 2))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    nse = np.nan if denom <= 0 else 1.0 - (sse / denom)

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "NSE": nse}

def flood_focus_metrics(y_true, y_pred, flood_threshold=None, event_threshold=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {}

    if flood_threshold is not None:
        flood_mask = y_true >= float(flood_threshold)
        out['FloodThreshold'] = float(flood_threshold)
        out['FloodN'] = int(flood_mask.sum())
        if flood_mask.sum() > 0:
            out['FloodMAE'] = float(mean_absolute_error(y_true[flood_mask], y_pred[flood_mask]))
            out['FloodRMSE'] = float(sqrt(mean_squared_error(y_true[flood_mask], y_pred[flood_mask])))
        else:
            out['FloodMAE'] = np.nan
            out['FloodRMSE'] = np.nan

    if event_threshold is not None:
        event_true = y_true >= float(event_threshold)
        event_pred = y_pred >= float(event_threshold)
        tp = int(np.sum(event_true & event_pred))
        fn = int(np.sum(event_true & (~event_pred)))
        fp = int(np.sum((~event_true) & event_pred))
        out['EventThreshold'] = float(event_threshold)
        out['EventN'] = int(event_true.sum())
        out['EventRecall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        out['EventPrecision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan

    return out

def eval_precomputed(name, y_true_raw, pred_raw, flood_threshold=None, event_threshold=None):
    res = extra_metrics(y_true_raw, pred_raw)
    res.update(flood_focus_metrics(y_true_raw, pred_raw, flood_threshold=flood_threshold, event_threshold=event_threshold))
    res['name'] = name
    return res, pred_raw

def eval_model_log(name, model, X_i, y_true_raw, flood_threshold=None, event_threshold=None):
    """Models trained on log1p(target); evaluate on original scale."""
    if flood_threshold is None and 'flood_threshold_train' in globals():
        flood_threshold = globals().get('flood_threshold_train')
    if event_threshold is None and 'event_threshold_train' in globals():
        event_threshold = globals().get('event_threshold_train')

    pred_log = model.predict(X_i)
    pred_raw = np.expm1(pred_log)
    res = extra_metrics(y_true_raw, pred_raw)
    res.update(flood_focus_metrics(y_true_raw, pred_raw, flood_threshold=flood_threshold, event_threshold=event_threshold))
    res['name'] = name
    return res, pred_raw

def eval_model(name, model, X_i, y_true_raw, flood_threshold=None, event_threshold=None):
    pred_raw = model.predict(X_i)
    res = extra_metrics(y_true_raw, pred_raw)
    res.update(flood_focus_metrics(y_true_raw, pred_raw, flood_threshold=flood_threshold, event_threshold=event_threshold))
    res['name'] = name
    return res, pred_raw


def plot_with_ci_band(df, actual_col, pred_col, lo_col, hi_col, title, fname, out_dir=OUT_DIR):
    plt.figure(figsize=(10,5))
    plt.plot(df['date'], df[actual_col], label='Actual', linewidth=2.5)
    plt.plot(df['date'], df[pred_col], label='Prediction', linewidth=1.8)
    plt.fill_between(df['date'], df[lo_col], df[hi_col], alpha=0.25, label='CI')
    plt.title(f'{title}'); plt.xlabel('Date'); plt.ylabel('Inflow')
    plt.grid(True); plt.legend(); plt.tight_layout()
    outpath = out_dir / fname
    plt.savefig(outpath, dpi=150); plt.close()
    print(f"Saved: {outpath}")

# Residual quantiles (original scale)
def winsorize(x, p=0.005):
    x = np.asarray(x, dtype=float)
    lo, hi = np.quantile(x[~np.isnan(x)], [p, 1-p])
    return np.clip(x, lo, hi)


def month_to_season(m:int)->str:
    m = int(m)
    if m in (12,1,2):  return 'DJF'
    if m in (3,4,5):   return 'MAM'
    if m in (6,7,8):   return 'JJA'
    return 'SON'

def build_quantile_maps_raw(y_true_raw, y_pred_raw, months, min_n_month=10, min_n_season=20):
    resid = np.asarray(y_true_raw, dtype=float) - np.asarray(y_pred_raw, dtype=float)
    resid_w = winsorize(resid, p=0.01)
    dfq = pd.DataFrame({
        'm': months.values.astype(int),
        'season': months.apply(month_to_season).values,
        'r': resid_w
    })
    g_lo90, g_hi90 = np.quantile(dfq['r'].dropna(), [0.05, 0.95])
    g_lo50, g_hi50 = np.quantile(dfq['r'].dropna(), [0.25, 0.75])
    season_q_90 = (dfq.groupby('season')['r'].quantile([0.05, 0.95]).unstack())
    season_q_50 = (dfq.groupby('season')['r'].quantile([0.25, 0.75]).unstack())
    season_n = dfq.groupby('season')['r'].size()
    month_q_90 = (dfq.groupby('m')['r'].quantile([0.05, 0.95]).unstack())
    month_q_50 = (dfq.groupby('m')['r'].quantile([0.25, 0.75]).unstack())
    month_n = dfq.groupby('m')['r'].size()

    def blend_quantiles(month_q, month_n, season_q, season_n, g_lo, g_hi):
        qlo_map, qhi_map = {}, {}
        for m in sorted(dfq['m'].unique()):
            s = month_to_season(m)
            if m in month_q.index and month_n.get(m, 0) >= min_n_month:
                m_lo, m_hi = float(month_q.loc[m].iloc[0]), float(month_q.loc[m].iloc[1])
                w_m = min(1.0, month_n[m] / 30.0)
            else:
                m_lo, m_hi = np.nan, np.nan
                w_m = 0.0
            if s in season_q.index and season_n.get(s, 0) >= min_n_season:
                s_lo, s_hi = float(season_q.loc[s].iloc[0]), float(season_q.loc[s].iloc[1])
                w_s = 1.0 - w_m
            else:
                s_lo, s_hi = np.nan, np.nan
                w_s = 0.0
            parts_lo, parts_hi, weights = [], [], []
            if w_m > 0 and np.isfinite(m_lo): parts_lo.append(m_lo); parts_hi.append(m_hi); weights.append(w_m)
            if w_s > 0 and np.isfinite(s_lo): parts_lo.append(s_lo); parts_hi.append(s_hi); weights.append(w_s)
            if len(parts_lo) == 0:
                qlo, qhi = g_lo, g_hi
            else:
                w = np.array(weights, dtype=float); w = w / w.sum()
                qlo = float(np.dot(w, np.array(parts_lo)))
                qhi = float(np.dot(w, np.array(parts_hi)))
            qlo_map[int(m)] = qlo
            qhi_map[int(m)] = qhi
        return qlo_map, qhi_map

    qlo90_map, qhi90_map = blend_quantiles(month_q_90, month_n, season_q_90, season_n, g_lo90, g_hi90)
    qlo50_map, qhi50_map = blend_quantiles(month_q_50, month_n, season_q_50, season_n, g_lo50, g_hi50)
    return qlo90_map, qhi90_map, g_lo90, g_hi90, qlo50_map, qhi50_map, g_lo50, g_hi50

def q_for_month(m, q_map, q_global):
    return float(q_map.get(int(m), q_global))

def resolve_first_existing_path(*candidates):
    for p in candidates:
        pp = Path(p)
        if pp.exists():
            return pp
    return None

def build_area_subbasin_features(rain_path: Path, sm_path: Path):
    rain_df = pd.read_csv(rain_path)
    sm_df = pd.read_csv(sm_path)

    need_rain = {'date', 'area_m2_used', 'rain_mm_aw'}
    need_sm = {'date', 'area_m2_used', 'sm_aw'}
    if not need_rain.issubset(set(rain_df.columns)):
        raise ValueError(
            f"Subbasin rain file missing columns {sorted(need_rain)}. Found: {rain_df.columns.tolist()}"
        )
    if not need_sm.issubset(set(sm_df.columns)):
        raise ValueError(
            f"Subbasin soil-moisture file missing columns {sorted(need_sm)}. Found: {sm_df.columns.tolist()}"
        )

    rain_df['date_day'] = pd.to_datetime(rain_df['date'], errors='coerce')
    sm_df['date_day'] = pd.to_datetime(sm_df['date'], errors='coerce')
    rain_df['area_m2_used'] = pd.to_numeric(rain_df['area_m2_used'], errors='coerce')
    sm_df['area_m2_used'] = pd.to_numeric(sm_df['area_m2_used'], errors='coerce')
    rain_df['rain_mm_aw'] = pd.to_numeric(rain_df['rain_mm_aw'], errors='coerce')
    sm_df['sm_aw'] = pd.to_numeric(sm_df['sm_aw'], errors='coerce')

    rain_unique = sorted(pd.Series(rain_df['area_m2_used'].dropna().unique(), dtype=float).tolist())
    sm_unique = sorted(pd.Series(sm_df['area_m2_used'].dropna().unique(), dtype=float).tolist())
    common_areas = sorted(set(rain_unique).intersection(set(sm_unique)))

    if common_areas:
        rain_area_to_sb = {a: f"SB{i+1}" for i, a in enumerate(common_areas)}
        sm_area_to_sb = rain_area_to_sb.copy()
        sb_cols = [rain_area_to_sb[a] for a in common_areas]
        area_weights = pd.Series({rain_area_to_sb[a]: a for a in common_areas}, dtype=float)
    else:
        if len(rain_unique) != len(sm_unique):
            raise ValueError(
                "No exact overlapping subbasin areas and counts differ between rain and soil files."
            )
        rain_area_to_sb = {a: f"SB{i+1}" for i, a in enumerate(rain_unique)}
        sm_area_to_sb = {a: f"SB{i+1}" for i, a in enumerate(sm_unique)}
        sb_cols = [f"SB{i+1}" for i in range(len(rain_unique))]
        area_weights = pd.Series({f"SB{i+1}": a for i, a in enumerate(rain_unique)}, dtype=float)

    area_weights = area_weights / area_weights.sum()

    rain_x = rain_df[['date_day', 'area_m2_used', 'rain_mm_aw']].dropna(subset=['date_day', 'area_m2_used'])
    rain_x['subbasin'] = rain_x['area_m2_used'].map(rain_area_to_sb)
    rain_wide = (
        rain_x.pivot_table(index='date_day', columns='subbasin', values='rain_mm_aw', aggfunc='mean')
        .reindex(columns=sb_cols)
        .sort_index()
    )

    sm_x = sm_df[['date_day', 'area_m2_used', 'sm_aw']].dropna(subset=['date_day', 'area_m2_used'])
    sm_x['subbasin'] = sm_x['area_m2_used'].map(sm_area_to_sb)
    sm_wide = (
        sm_x.pivot_table(index='date_day', columns='subbasin', values='sm_aw', aggfunc='mean')
        .reindex(columns=sb_cols)
        .sort_index()
    )

    feats = pd.DataFrame(index=rain_wide.index.union(sm_wide.index).sort_values())

    for sb in sb_cols:
        rain_s = rain_wide[sb].reindex(feats.index).astype(float).fillna(0.0)
        sm_s = sm_wide[sb].reindex(feats.index).astype(float)

        feats[f'arain_{sb}_7d'] = rain_s.rolling(7, min_periods=1).sum()
        feats[f'arain_{sb}_30d'] = rain_s.rolling(30, min_periods=1).sum()
        feats[f'asm_{sb}_7d'] = sm_s.rolling(7, min_periods=1).mean()
        feats[f'asm_{sb}_30d'] = sm_s.rolling(30, min_periods=1).mean()

    rain_7d_cols = [f'arain_{sb}_7d' for sb in sb_cols]
    rain_30d_cols = [f'arain_{sb}_30d' for sb in sb_cols]
    sm_7d_cols = [f'asm_{sb}_7d' for sb in sb_cols]
    sm_30d_cols = [f'asm_{sb}_30d' for sb in sb_cols]

    feats['arain_weighted_7d'] = feats[rain_7d_cols].mul(area_weights, axis=1).sum(axis=1)
    feats['arain_weighted_30d'] = feats[rain_30d_cols].mul(area_weights, axis=1).sum(axis=1)
    feats['asm_weighted_7d'] = feats[sm_7d_cols].mul(area_weights, axis=1).sum(axis=1)
    feats['asm_weighted_30d'] = feats[sm_30d_cols].mul(area_weights, axis=1).sum(axis=1)
    feats['arain_spread_7d'] = feats[rain_7d_cols].std(axis=1)
    feats['asm_spread_30d'] = feats[sm_30d_cols].std(axis=1)

    feature_cols = rain_7d_cols + rain_30d_cols + sm_7d_cols + sm_30d_cols + [
        'arain_weighted_7d',
        'arain_weighted_30d',
        'asm_weighted_7d',
        'asm_weighted_30d',
        'arain_spread_7d',
        'asm_spread_30d',
    ]

    feats = feats.reset_index().rename(columns={'index': 'date_day'})
    return feats, feature_cols, sb_cols

# ---------------- LOAD & CLEAN ----------------
df = pd.read_csv(DATA_PATH)

# Map month strings to ints if needed
month_map = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
             'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
if df[COL_MONTH].dtype == object:
    df[COL_MONTH] = df[COL_MONTH].map(month_map).astype(int)

need = [COL_YEAR, COL_MONTH, COL_DAY, COL_INFLOW, COL_RAIN_DAM, COL_RAIN_BSN]
df = df[need].copy().sort_values([COL_YEAR, COL_MONTH, COL_DAY])

# --- Daily dataframe prep (for next-30-day model) ---
df_daily = df.copy()
df_daily['date_day'] = pd.to_datetime(
    dict(year=df_daily[COL_YEAR], month=df_daily[COL_MONTH], day=df_daily[COL_DAY]),
    errors='coerce'
)
df_daily = df_daily.dropna(subset=['date_day']).sort_values('date_day')

# Use only Rain 7A to BSN (basin) as the rain feature
df_daily['rain_total'] = df_daily[COL_RAIN_BSN].astype(float).fillna(0.0)
df_daily['inflow_day'] = df_daily[COL_INFLOW].astype(float).fillna(0.0)
df_daily['rain_dam']  = df_daily[COL_RAIN_DAM].astype(float).fillna(0.0)

# DAILY → NEXT-30-DAY INFLOW MODEL (rolling-window features)

print("\n===== DAILY NEXT-30-DAY INFLOW MODEL =====")

# 1. Build / clean daily dataframe (df_daily already exists)
daily = df_daily.copy().sort_values('date_day').reset_index(drop=True)

# plot daily inflow time series (raw + 30-day running sum) and save to OUT_DIR
daily['date_day'] = pd.to_datetime(daily['date_day'])
daily['inflow_raw_plot'] = pd.to_numeric(daily[COL_INFLOW], errors='coerce').fillna(0.0)
daily['inflow_30d_sum'] = daily['inflow_raw_plot'].rolling(30, min_periods=1).sum()

plt.figure(figsize=(12,5))
ax = plt.gca()

# compute split boundary datetimes
boundary_train_to_val = pd.Timestamp(year=TRAIN_END + 1, month=1, day=1)
boundary_val_to_test  = pd.Timestamp(year=VAL_END + 1,   month=1, day=1)
min_dt = daily['date_day'].min()
max_dt = daily['date_day'].max()

# shaded regions: train / val / test
ax.axvspan(min_dt, boundary_train_to_val, color='tab:green', alpha=0.10, zorder=0)
ax.axvspan(boundary_train_to_val, boundary_val_to_test, color='tab:orange', alpha=0.10, zorder=0)
ax.axvspan(boundary_val_to_test, max_dt, color='tab:blue', alpha=0.08, zorder=0)

# plot series on top
l1, = ax.plot(daily['date_day'], daily['inflow_raw_plot'], color='tab:red', alpha=0.6, label='Daily inflow')

# add vertical boundary lines for clarity
ax.axvline(boundary_train_to_val, color='tab:green', linestyle='--', linewidth=1.0)
ax.axvline(boundary_val_to_test,  color='tab:orange', linestyle='--', linewidth=1.0)

# build legend combining lines and shaded region labels
from matplotlib.patches import Patch
patch_train = Patch(facecolor='tab:green', alpha=0.10, label=f'Train (<= {TRAIN_END})')
patch_val   = Patch(facecolor='tab:orange', alpha=0.10, label=f'Val ({TRAIN_END+1}–{VAL_END})')
patch_test  = Patch(facecolor='tab:blue', alpha=0.08, label=f'Test (>= {VAL_END+1})')

ax.set_xlabel('Date'); ax.set_ylabel('Inflow'); ax.set_title('Daily inflow time series')
ax.grid(True)
# show patches first then series
ax.legend(handles=[patch_train, patch_val, patch_test, l1], fontsize='small', loc='upper right')
plt.tight_layout()

outf = OUT_DIR / 'daily_inflow_timeseries.png'
plt.savefig(outf, dpi=150); plt.close()
print(f"Saved: {outf}")

# Make sure inflow is float
daily[COL_INFLOW] = daily[COL_INFLOW].astype(float)

# 2a. Merge in area-weighted subbasin rain/soil features
sub_rain_path = resolve_first_existing_path(
    SUBBASIN_RAIN_PATH,
    Path(r"C:\Users\aalec\Desktop\Altus Project\ERA5Land_daily_area_weighted_subbasin_rainfall.csv")
)
sub_sm_path = resolve_first_existing_path(
    SUBBASIN_SM_PATH,
    Path(r"C:\Users\aalec\Desktop\Altus Project\ERA5Land_daily_area_weighted_subbasin_soil_moisture.csv")
)
if sub_rain_path is None or sub_sm_path is None:
    raise FileNotFoundError(
        "Could not find subbasin rain/soil files. "
        f"Rain path tried: {SUBBASIN_RAIN_PATH} and desktop fallback; "
        f"Soil path tried: {SUBBASIN_SM_PATH} and desktop fallback."
    )

sub_feats, sub_feature_cols, subbasins_found = build_area_subbasin_features(sub_rain_path, sub_sm_path)
daily = daily.merge(sub_feats, on='date_day', how='left')
print(f"Added area-specific features from {len(subbasins_found)} subbasins: {subbasins_found}")


# 3. Rolling-window features (rain, inflow, soil moisture, storage)

# Base series
rain = daily['rain_total'].astype(float).fillna(0.0)
dam = daily['rain_dam'].astype(float).fillna(0.0)
qin  = daily[COL_INFLOW].astype(float).fillna(0.0)


# Rainfall rolling windows (using data up to and including today)
daily['rain_1d']   = rain
daily['rain_3d']   = rain.rolling(3,  min_periods=1).sum()
daily['rain_7d']   = rain.rolling(7,  min_periods=1).sum()
daily['rain_14d']  = rain.rolling(14, min_periods=1).sum()
daily['rain_30d']  = rain.rolling(30, min_periods=1).sum()
daily['rain_45d']  = rain.rolling(45, min_periods=1).sum()

daily['dam_1d']   = dam
daily['dam_3d']   = dam.rolling(3,  min_periods=1).sum()
daily['dam_7d']   = dam.rolling(7,  min_periods=1).sum()
daily['dam_14d']  = dam.rolling(14, min_periods=    1).sum()
daily['dam_30d']  = dam.rolling(30, min_periods=1).sum()            
daily['dam_45d']  = dam.rolling(45, min_periods=1).sum()

# Inflow rolling windows (use only past inflow: shift by 1 day)
qin_shift = qin.shift(1)
daily['qin_lag1']   = qin_shift
daily['qin_3d']     = qin_shift.rolling(3,  min_periods=1).mean()
daily['qin_7d']     = qin_shift.rolling(7,  min_periods=1).mean()
daily['qin_14d']    = qin_shift.rolling(14, min_periods=1).mean()
daily['qin_30d']    = qin_shift.rolling(30, min_periods=1).sum()



# Seasonality features (day of year)
daily['doy']     = daily['date_day'].dt.dayofyear
daily['doy_sin'] = np.sin(2*np.pi*daily['doy']/365.25)
daily['doy_cos'] = np.cos(2*np.pi*daily['doy']/365.25)

# 4. Target: sum of inflow over the NEXT 30 days (rolling window)
#
# inflow_next30(t) = sum of inflow from t+1 to t+30
daily['inflow_next30'] = (
    qin.shift(-30)  # shift back 30 days to get future inflow
       .rolling(30, min_periods=30)
       .sum()
)

daily['inflow_next30_log'] = np.log1p(daily['inflow_next30'])

# 5. Build modeling dataframe

daily['year'] = daily['date_day'].dt.year

daily_features = [
    #'rain_1d','rain_3d',
    #'rain_7d',
    # 'rain_14d','rain_30d','rain_45d',
    #'dam_7d','dam_14d',
    #'dam_30d','dam_45d',
    #'qin_lag1','qin_3d','qin_7d','qin_14d',
    'qin_30d',
    #'sm_1d','sm_7d_mean','sm_14d_mean',
    #'sm_30d_mean',
    'doy_sin','doy_cos'
]
# Editable subbasin feature list (remove any items you want to test without)
SELECTED_SUBBASIN_FEATURES = [
    #'arain_SB1_7d',
    #'arain_SB2_7d',
    #'arain_SB3_7d',
    'arain_SB4_7d',
    'arain_SB1_30d',
    #'arain_SB2_30d',
    #'arain_SB3_30d',
    'arain_SB4_30d',
    #'asm_SB1_7d',
    'asm_SB2_7d',
    'asm_SB3_7d',
    #'asm_SB4_7d',
    'asm_SB1_30d',
    'asm_SB2_30d',
    'asm_SB3_30d',
    'asm_SB4_30d',
    #'arain_weighted_7d',
    #'arain_weighted_30d',
    #'asm_weighted_7d',
    #'asm_weighted_30d',
    #'arain_spread_7d',
    #'asm_spread_30d',
]

available_sub_features = set(sub_feature_cols)
selected_sub_features = [f for f in SELECTED_SUBBASIN_FEATURES if f in available_sub_features]
missing_sub_features = [f for f in SELECTED_SUBBASIN_FEATURES if f not in available_sub_features]
if missing_sub_features:
    print(f"Warning: requested subbasin features not found in this run: {missing_sub_features}")
if len(selected_sub_features) == 0:
    raise ValueError("No subbasin features selected. Edit SELECTED_SUBBASIN_FEATURES.")

daily_features += selected_sub_features

daily_model_df = daily[['date_day','year'] + daily_features + ['inflow_next30','inflow_next30_log']].dropna().reset_index(drop=True)

# 6. Time-based split (same TRAIN_END / VAL_END logic)

train_d = daily_model_df[daily_model_df['year'] <= TRAIN_END].copy()
val_d   = daily_model_df[(daily_model_df['year'] > TRAIN_END) & (daily_model_df['year'] <= VAL_END)].copy()
test_d  = daily_model_df[daily_model_df['year'] > VAL_END].copy()

Xd_tr, Xd_v, Xd_t = train_d[daily_features], val_d[daily_features], test_d[daily_features]
yd_tr_raw, yd_v_raw, yd_t_raw = train_d['inflow_next30'], val_d['inflow_next30'], test_d['inflow_next30']
yd_tr_log, yd_v_log, yd_t_log = train_d['inflow_next30_log'], val_d['inflow_next30_log'], test_d['inflow_next30_log']

# Flood/event thresholds derived from TRAIN only (prevents leakage)
flood_threshold_train = float(np.quantile(yd_tr_raw, FLOOD_QUANTILE))
event_threshold_train = float(np.quantile(yd_tr_raw, EVENT_QUANTILE))
print(f"Flood threshold (q={FLOOD_QUANTILE:.2f}) from train: {flood_threshold_train:.2f}")
print(f"Event threshold (q={EVENT_QUANTILE:.2f}) from train: {event_threshold_train:.2f}")

# 7. Impute features (shared across models)

imp_d = SimpleImputer(strategy='mean')
Xd_tr_i = imp_d.fit_transform(Xd_tr)
Xd_v_i  = imp_d.transform(Xd_v)
Xd_t_i  = imp_d.transform(Xd_t)

metrics_d = {}
preds_train = {}
preds_val   = {}
preds_test  = {}

# 7a. Linear Regression (log target)
lin_d = LinearRegression()
lin_d.fit(Xd_tr_i, yd_tr_log)

m_tr_lin, pred_tr_lin = eval_model_log('Linear_DailyNext30', lin_d, Xd_tr_i, yd_tr_raw)
m_v_lin,  pred_v_lin  = eval_model_log('Linear_DailyNext30', lin_d, Xd_v_i,  yd_v_raw)
m_t_lin,  pred_t_lin  = eval_model_log('Linear_DailyNext30', lin_d, Xd_t_i,  yd_t_raw)

metrics_d['Linear_DailyNext30'] = [
    dict(split='train', **m_tr_lin),
    dict(split='val',   **m_v_lin),
    dict(split='test',  **m_t_lin),
]
preds_train['Linear_DailyNext30'] = pred_tr_lin
preds_val['Linear_DailyNext30']   = pred_v_lin
preds_test['Linear_DailyNext30']  = pred_t_lin

# 7b. Ridge Regression (log target)
ridge_d = Ridge(alpha=1.0, random_state=0)
ridge_d.fit(Xd_tr_i, yd_tr_log)

# use daily splits/targets (fixed variable mix-up was causing the shape error)
m_tr_ridge, pred_tr_ridge = eval_model_log('Ridge_DailyNext30', ridge_d, Xd_tr_i, yd_tr_raw)
m_v_ridge,  pred_v_ridge  = eval_model_log('Ridge_DailyNext30', ridge_d, Xd_v_i,  yd_v_raw)
m_t_ridge,  pred_t_ridge  = eval_model_log('Ridge_DailyNext30', ridge_d, Xd_t_i,  yd_t_raw)

metrics_d['Ridge_DailyNext30'] = [
    dict(split='train', **m_tr_ridge),
    dict(split='val',   **m_v_ridge),
    dict(split='test',  **m_t_ridge),
]
preds_train['Ridge_DailyNext30'] = pred_tr_ridge
preds_val['Ridge_DailyNext30']   = pred_v_ridge
preds_test['Ridge_DailyNext30']  = pred_t_ridge

# 7c. RandomForest (if available)
if RF_AVAILABLE:
    # Baseline RF
    rf_d = RandomForestRegressor(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=5,
        random_state=0,
        n_jobs=-1
    )
    rf_d.fit(Xd_tr_i, yd_tr_log)

    m_tr_rf, pred_tr_rf = eval_model_log('RF_DailyNext30', rf_d, Xd_tr_i, yd_tr_raw)
    m_v_rf,  pred_v_rf  = eval_model_log('RF_DailyNext30', rf_d, Xd_v_i,  yd_v_raw)
    m_t_rf,  pred_t_rf  = eval_model_log('RF_DailyNext30', rf_d, Xd_t_i,  yd_t_raw)

    metrics_d['RF_DailyNext30'] = [
        dict(split='train', **m_tr_rf),
        dict(split='val',   **m_v_rf),
        dict(split='test',  **m_t_rf),
    ]
    preds_train['RF_DailyNext30'] = pred_tr_rf
    preds_val['RF_DailyNext30']   = pred_v_rf
    preds_test['RF_DailyNext30']  = pred_t_rf

    # Tail-aware weighted RF (robustness to flood years)
    train_weights = np.ones(len(yd_tr_raw), dtype=float)
    train_weights[yd_tr_raw.values >= event_threshold_train] = TAIL_EVENT_WEIGHT
    train_weights[yd_tr_raw.values >= flood_threshold_train] = TAIL_FLOOD_WEIGHT

    rf_w = RandomForestRegressor(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=6,
        random_state=0,
        n_jobs=-1
    )
    rf_w.fit(Xd_tr_i, yd_tr_log, sample_weight=train_weights)

    m_tr_rfw, pred_tr_rfw = eval_model_log('RF_Weighted_DailyNext30', rf_w, Xd_tr_i, yd_tr_raw)
    m_v_rfw,  pred_v_rfw  = eval_model_log('RF_Weighted_DailyNext30', rf_w, Xd_v_i,  yd_v_raw)
    m_t_rfw,  pred_t_rfw  = eval_model_log('RF_Weighted_DailyNext30', rf_w, Xd_t_i,  yd_t_raw)

    metrics_d['RF_Weighted_DailyNext30'] = [
        dict(split='train', **m_tr_rfw),
        dict(split='val',   **m_v_rfw),
        dict(split='test',  **m_t_rfw),
    ]
    preds_train['RF_Weighted_DailyNext30'] = pred_tr_rfw
    preds_val['RF_Weighted_DailyNext30']   = pred_v_rfw
    preds_test['RF_Weighted_DailyNext30']  = pred_t_rfw

    # Regime model: classify high-event regime then blend specialist regressors
    y_regime_train = (yd_tr_raw.values >= event_threshold_train).astype(int)
    n_regime_pos = int(y_regime_train.sum())
    n_regime_neg = int((1 - y_regime_train).sum())

    if n_regime_pos >= REGIME_MIN_SAMPLES and n_regime_neg >= REGIME_MIN_SAMPLES:
        regime_clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=10,
            random_state=0,
            n_jobs=-1
        )
        regime_clf.fit(Xd_tr_i, y_regime_train, sample_weight=train_weights)

        low_mask = y_regime_train == 0
        high_mask = y_regime_train == 1

        rf_low = RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=6,
            random_state=0,
            n_jobs=-1
        )
        rf_high = RandomForestRegressor(
            n_estimators=700,
            max_depth=9,
            min_samples_leaf=4,
            random_state=0,
            n_jobs=-1
        )

        rf_low.fit(Xd_tr_i[low_mask], yd_tr_log.values[low_mask])
        rf_high.fit(
            Xd_tr_i[high_mask],
            yd_tr_log.values[high_mask],
            sample_weight=np.maximum(train_weights[high_mask], 1.0)
        )

        def predict_regime_blend(Xm):
            p_high = regime_clf.predict_proba(Xm)[:, 1]
            pred_low = np.expm1(rf_low.predict(Xm))
            pred_high = np.expm1(rf_high.predict(Xm))
            return ((1.0 - p_high) * pred_low) + (p_high * pred_high)

        pred_tr_reg = predict_regime_blend(Xd_tr_i)
        pred_v_reg = predict_regime_blend(Xd_v_i)
        pred_t_reg = predict_regime_blend(Xd_t_i)

        m_tr_reg, _ = eval_precomputed('RF_Regime_DailyNext30', yd_tr_raw, pred_tr_reg,
                                       flood_threshold=flood_threshold_train,
                                       event_threshold=event_threshold_train)
        m_v_reg, _ = eval_precomputed('RF_Regime_DailyNext30', yd_v_raw, pred_v_reg,
                                      flood_threshold=flood_threshold_train,
                                      event_threshold=event_threshold_train)
        m_t_reg, _ = eval_precomputed('RF_Regime_DailyNext30', yd_t_raw, pred_t_reg,
                                      flood_threshold=flood_threshold_train,
                                      event_threshold=event_threshold_train)

        metrics_d['RF_Regime_DailyNext30'] = [
            dict(split='train', **m_tr_reg),
            dict(split='val',   **m_v_reg),
            dict(split='test',  **m_t_reg),
        ]
        preds_train['RF_Regime_DailyNext30'] = pred_tr_reg
        preds_val['RF_Regime_DailyNext30']   = pred_v_reg
        preds_test['RF_Regime_DailyNext30']  = pred_t_reg
    else:
        print(f"Skipping RF_Regime_DailyNext30: not enough regime samples (high={n_regime_pos}, low={n_regime_neg}).")

# 7d. XGBoost (if available)
if XGB_AVAILABLE:
    xgb_d = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=0,
        n_jobs=-1
    )
    xgb_d.fit(Xd_tr_i, yd_tr_log)

    m_tr_xgb, pred_tr_xgb = eval_model_log('XGB_DailyNext30', xgb_d, Xd_tr_i, yd_tr_raw)
    m_v_xgb,  pred_v_xgb  = eval_model_log('XGB_DailyNext30', xgb_d, Xd_v_i,  yd_v_raw)
    m_t_xgb,  pred_t_xgb  = eval_model_log('XGB_DailyNext30', xgb_d, Xd_t_i,  yd_t_raw)

    metrics_d['XGB_DailyNext30'] = [
        dict(split='train', **m_tr_xgb),
        dict(split='val',   **m_v_xgb),
        dict(split='test',  **m_t_xgb),
    ]
    preds_train['XGB_DailyNext30'] = pred_tr_xgb
    preds_val['XGB_DailyNext30']   = pred_v_xgb
    preds_test['XGB_DailyNext30']  = pred_t_xgb

# 8. Collect metrics into a single DataFrame
rows = []
for name, mlist in metrics_d.items():
    for m in mlist:
        m = m.copy()
        m['name'] = name
        rows.append(m)

metrics_daily_df = pd.DataFrame(rows).sort_values(['split','RMSE','name'])
print("\n=== DAILY NEXT-30D METRICS (lower is better) ===")
print(metrics_daily_df)

metrics_daily_df.to_csv(OUT_DIR / "metrics_daily_next30.csv", index=False)

# 9. Choose best model (forced model or validation-based auto-selection)
if FORCE_BEST_MODEL is not None:
    if FORCE_BEST_MODEL not in preds_val:
        raise ValueError(f"FORCE_BEST_MODEL '{FORCE_BEST_MODEL}' not found in trained models: {list(preds_val.keys())}")
    best_name = FORCE_BEST_MODEL
    print(f"\nBest daily next-30d model forced by config: {best_name}")
else:
    val_metrics = metrics_daily_df[metrics_daily_df['split'] == 'val'].copy()
    if MODEL_SELECTION_METRIC in val_metrics.columns and val_metrics[MODEL_SELECTION_METRIC].notna().any():
        val_metrics = val_metrics.sort_values([MODEL_SELECTION_METRIC, 'MAE', 'RMSE'])
        best_name = val_metrics.iloc[0]['name']
        print(f"\nBest daily next-30d model on val ({MODEL_SELECTION_METRIC}): {best_name}")
    else:
        val_metrics = val_metrics.sort_values(['MAE', 'RMSE'])
        best_name = val_metrics.iloc[0]['name']
        print(f"\nBest daily next-30d model on val (MAE fallback): {best_name}")


# 10. Build residual-based uncertainty bands using VALIDATION set
#     (same logic as monthly CIs, but with daily-origin month)

y_true_val = yd_v_raw.values
y_pred_val = preds_val[best_name]
months_val = val_d['date_day'].dt.month

qlo90_map, qhi90_map, glo90, ghi90, qlo50_map, qhi50_map, glo50, ghi50 = build_quantile_maps_raw(
    y_true_raw=y_true_val,
    y_pred_raw=y_pred_val,
    months=months_val,
    min_n_month=10,
    min_n_season=20
)

# Apply month-conditioned residual quantiles to TEST predictions
months_test = test_d['date_day'].dt.month
qlo90 = months_test.apply(lambda m: q_for_month(m, qlo90_map, glo90)).to_numpy()
qhi90 = months_test.apply(lambda m: q_for_month(m, qhi90_map, ghi90)).to_numpy()
qlo50 = months_test.apply(lambda m: q_for_month(m, qlo50_map, glo50)).to_numpy()
qhi50 = months_test.apply(lambda m: q_for_month(m, qhi50_map, ghi50)).to_numpy()

# Build test output with CIs
test_daily_out = test_d[['date_day']].copy()
test_daily_out['y_true_next30'] = yd_t_raw.values
test_daily_out['y_pred_next30'] = preds_test[best_name]

test_daily_out['y_lo90'] = test_daily_out['y_pred_next30'] + qlo90
test_daily_out['y_hi90'] = test_daily_out['y_pred_next30'] + qhi90
test_daily_out['y_lo50'] = test_daily_out['y_pred_next30'] + qlo50
test_daily_out['y_hi50'] = test_daily_out['y_pred_next30'] + qhi50

# Save raw test predictions + CIs
test_daily_out_sorted = test_daily_out.sort_values('date_day')
test_daily_out_sorted.to_csv(OUT_DIR / "predictions_daily_next30_with_ci.csv", index=False)

# Monthly issuance slice (parallel to daily): issue on day ISSUE_DAY_OF_MONTH each month
test_monthly_issue = test_daily_out_sorted[test_daily_out_sorted['date_day'].dt.day == ISSUE_DAY_OF_MONTH].copy()
test_monthly_issue.to_csv(OUT_DIR / f"predictions_monthly_issue_day{ISSUE_DAY_OF_MONTH}_next30_with_ci.csv", index=False)

# Cadence metrics side-by-side (daily_all vs monthly_dayX)
cadence_rows = []
for cadence_name, cdf in [
    ('daily_all', test_daily_out_sorted),
    (f'monthly_day{ISSUE_DAY_OF_MONTH}', test_monthly_issue),
]:
    if len(cdf) == 0:
        continue
    y_true_c = cdf['y_true_next30'].to_numpy()
    y_pred_c = cdf['y_pred_next30'].to_numpy()
    m = extra_metrics(y_true_c, y_pred_c)
    m.update(flood_focus_metrics(y_true_c, y_pred_c, flood_threshold=flood_threshold_train, event_threshold=event_threshold_train))
    m['cadence'] = cadence_name
    m['n'] = int(len(cdf))
    m['model'] = best_name
    cadence_rows.append(m)

cadence_metrics_df = pd.DataFrame(cadence_rows)
if len(cadence_metrics_df) > 0:
    cadence_metrics_df.to_csv(OUT_DIR / "metrics_issue_cadence_next30.csv", index=False)
    print("\n=== ISSUE CADENCE METRICS (best model) ===")
    print(cadence_metrics_df)

# 11. Plot with CI bands using existing helper (needs 'date' column)
df_plot_90 = test_daily_out_sorted.rename(columns={
    'date_day': 'date',
    'y_true_next30': 'y_true',
    'y_pred_next30': 'y_pred'
})

plot_with_ci_band(
    df=df_plot_90,
    actual_col='y_true',
    pred_col='y_pred',
    lo_col='y_lo90',
    hi_col='y_hi90',
    title=f'{best_name} Next 30-Day Inflow with 90% CI (Test - Daily)',
    fname='test_timeline_daily_next30_ci90.png',
    out_dir=OUT_DIR
)

plot_with_ci_band(
    df=df_plot_90,
    actual_col='y_true',
    pred_col='y_pred',
    lo_col='y_lo50',
    hi_col='y_hi50',
    title=f'{best_name} Next 30-Day Inflow with 50% CI (Test - Daily)',
    fname='test_timeline_daily_next30_ci50.png',
    out_dir=OUT_DIR
)

if len(test_monthly_issue) > 0:
    df_plot_monthly = test_monthly_issue.rename(columns={
        'date_day': 'date',
        'y_true_next30': 'y_true',
        'y_pred_next30': 'y_pred'
    })

    plot_with_ci_band(
        df=df_plot_monthly,
        actual_col='y_true',
        pred_col='y_pred',
        lo_col='y_lo90',
        hi_col='y_hi90',
        title=f'{best_name} Next 30-Day Inflow with 90% CI (Test - Monthly Day {ISSUE_DAY_OF_MONTH})',
        fname=f'test_timeline_monthly_day{ISSUE_DAY_OF_MONTH}_next30_ci90.png',
        out_dir=OUT_DIR
    )

    plot_with_ci_band(
        df=df_plot_monthly,
        actual_col='y_true',
        pred_col='y_pred',
        lo_col='y_lo50',
        hi_col='y_hi50',
        title=f'{best_name} Next 30-Day Inflow with 50% CI (Test - Monthly Day {ISSUE_DAY_OF_MONTH})',
        fname=f'test_timeline_monthly_day{ISSUE_DAY_OF_MONTH}_next30_ci50.png',
        out_dir=OUT_DIR
    )

print(f"Saved: predictions_daily_next30_with_ci.csv")
print(f"Saved: predictions_monthly_issue_day{ISSUE_DAY_OF_MONTH}_next30_with_ci.csv")
print("Saved: metrics_issue_cadence_next30.csv")

# --- Plot RandomForest Actual vs Predicted (daily only) ---
if RF_AVAILABLE:

    # Daily next-30d test (if RF produced daily preds stored in preds_test) - line plot over daily test period
    if 'RF_DailyNext30' in preds_test and preds_test['RF_DailyNext30'] is not None:
        # build dataframe aligned to test_d dates
        df_rf_daily = test_d[['date_day']].copy().reset_index(drop=True)
        df_rf_daily['y_true_next30'] = yd_t_raw.values
        df_rf_daily['y_pred_next30'] = preds_test['RF_DailyNext30']
        df_rf_daily = df_rf_daily.sort_values('date_day').reset_index(drop=True)

        plt.figure(figsize=(12,5))
        plt.plot(df_rf_daily['date_day'], df_rf_daily['y_true_next30'], label='Actual Next-30d', linewidth=1.8)
        plt.plot(df_rf_daily['date_day'], df_rf_daily['y_pred_next30'], label='RF Predicted Next-30d', linewidth=1.2)
        plt.xlabel('Date'); plt.ylabel('Next-30d Inflow'); plt.title('RandomForest: Actual vs Predicted (Daily Next-30 Test)')
        plt.grid(True); plt.legend(); plt.tight_layout()
        outfile2 = OUT_DIR / 'rf_line_actual_vs_pred_daily_next30.png'
        plt.savefig(outfile2, dpi=150); plt.close()
        print(f"Saved: {outfile2}")

# ---- Combined Actual vs Models (Daily Next-30 Test): Linear, RandomForest, XGBoost ----
df_daily_models = test_d[['date_day']].copy().reset_index(drop=True)
df_daily_models['y_true'] = yd_t_raw.values

# add model preds if present in preds_test
if 'Linear_DailyNext30' in preds_test and preds_test['Linear_DailyNext30'] is not None:
    df_daily_models['y_lin'] = preds_test['Linear_DailyNext30']
if 'RF_DailyNext30' in preds_test and preds_test['RF_DailyNext30'] is not None:
    df_daily_models['y_rf'] = preds_test['RF_DailyNext30']
if 'XGB_DailyNext30' in preds_test and preds_test['XGB_DailyNext30'] is not None:
    df_daily_models['y_xgb'] = preds_test['XGB_DailyNext30']

# Only proceed if at least one model column exists
model_cols = [c for c in ['y_lin','y_rf','y_xgb'] if c in df_daily_models.columns]
if len(model_cols) > 0:
    df_daily_models = df_daily_models.sort_values('date_day').reset_index(drop=True)
    plt.figure(figsize=(12,5))
    plt.plot(df_daily_models['date_day'], df_daily_models['y_true'], label='Actual Next-30d', linewidth=2.0, color='k')
    if 'y_lin' in df_daily_models:
        plt.plot(df_daily_models['date_day'], df_daily_models['y_lin'], label='Linear', color='tab:orange', linewidth=1.2)
    if 'y_rf' in df_daily_models:
        plt.plot(df_daily_models['date_day'], df_daily_models['y_rf'], label='RandomForest', color='tab:green', linewidth=1.2)
    if 'y_xgb' in df_daily_models:
        plt.plot(df_daily_models['date_day'], df_daily_models['y_xgb'], label='XGBoost', color='tab:blue', linewidth=1.2)
    plt.xlabel('Date'); plt.ylabel('Next-30d Inflow'); plt.title('Actual vs Model Predictions (Daily Next-30 Test)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    out_daily_combined = OUT_DIR / 'models_actual_daily_next30_test.png'
    plt.savefig(out_daily_combined, dpi=150); plt.close()
    print(f"Saved: {out_daily_combined}")
else:
    print("Skipping daily combined model plot: no daily model predictions found in preds_test.")
