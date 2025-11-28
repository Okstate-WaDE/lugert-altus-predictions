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
import calendar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math import sqrt
from scipy.stats import gamma

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

SOIL_PATH = Path(r"C:\Users\aalec\Desktop\Altus Project\Altus_Lugert_ERA5L_sm_layer1_hourly_v2.csv")
# Fixed split years
TRAIN_END = 2018   # inclusive
VAL_END   = 2021   # inclusive; test is > VAL_END

# Lags
N_LAGS = 6

# Exact CSV headers
COL_DAY         = 'day'
COL_POOL_ELEV   = 'elevations (ft)'
COL_STORAGE2400 = 'storage (2400hr)'
COL_REL_POWER   = 'releases (power)'
COL_REL_TOTAL   = 'releases (total)'
COL_EVAP        = 'evap inches'
COL_INFLOW      = 'inflow adj'
COL_RAIN_DAM    = 'rainfall inches (7A to Dam)'
COL_RAIN_BSN    = 'rainfall inches (7A to BSN)'
COL_MONTH       = 'month'
COL_YEAR        = 'year'

# ---------------- HELPERS ----------------
def extra_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def eval_model_log(name, model, X_i, y_true_raw):
    """Models trained on log1p(target); evaluate on original scale."""
    pred_log = model.predict(X_i)
    pred_raw = np.expm1(pred_log)
    res = extra_metrics(y_true_raw, pred_raw)
    res['name'] = name
    return res, pred_raw

def eval_model(name, model, X_i, y_true_raw):
    pred_raw = model.predict(X_i)
    res = extra_metrics(y_true_raw, pred_raw)
    res['name'] = name
    return res, pred_raw

def add_global_lags(frame, cols, n_lags=N_LAGS):
    for c in cols:
        for k in range(1, n_lags+1):
            frame[f'{c}_{k}'] = frame[c].shift(k)
    return frame

def plot_month_rain_pdf(df_monthly, month_int, out_dir=OUT_DIR):
    sub = df_monthly.loc[df_monthly[COL_MONTH]==month_int, 'rainfall'].dropna()
    sub = sub[sub > 0]
    if len(sub) < 8:
        print(f"Skipping {calendar.month_name[month_int]}: not enough samples ({len(sub)}).")
        return
    mu  = sub.mean()
    std = sub.std(ddof=1)
    shape, loc, scale = gamma.fit(sub, floc=0)
    xs = np.linspace(0, max(sub.max()*1.2, 1e-6), 300)
    yn = (1.0/(std*np.sqrt(2*np.pi))) * np.exp(-((xs-mu)**2)/(2*std**2))
    yg = gamma.pdf(xs, a=shape, loc=loc, scale=scale)

    plt.figure(figsize=(8,4))
    plt.hist(sub, bins=15, density=True, alpha=0.5, label='Empirical')
    plt.plot(xs, yn, label='Normal PDF')
    plt.plot(xs, yg, label='Gamma PDF')
    plt.title(f'Rainfall PDFs – {calendar.month_name[month_int]}')
    plt.xlabel('Rainfall (in)'); plt.ylabel('Density')
    plt.grid(True); plt.legend(); plt.tight_layout()
    outfile = out_dir / f"rain_pdf_{month_int:02d}_{calendar.month_abbr[month_int]}.png"
    plt.savefig(outfile, dpi=150); plt.close()
    print(f"Saved: {outfile}")

def plot_test_timeline(test_df, series_dict, out_dir=OUT_DIR, fname="test_timeline.png", title='Next-Month Inflow – Test Set'):
    p = test_df[['date']].copy()
    for k, v in series_dict.items():
        p[k] = v
    p = p.sort_values('date')
    plt.figure(figsize=(10,5))
    for k in series_dict.keys():
        lw = 2.5 if k == 'Actual' else 1.5
        plt.plot(p['date'], p[k], label=k, linewidth=lw)
    plt.title(f'{title} (original units)')
    plt.xlabel('Date'); plt.ylabel('Inflow')
    plt.grid(True); plt.legend(); plt.tight_layout()
    outfile = out_dir / fname
    plt.savefig(outfile, dpi=150); plt.close()
    print(f"Saved: {outfile}")

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

def resid(y, x, fit_intercept=True):
    """
    Return residuals of regressing y on x (y - pred), aligned to input index.
    y, x : array-like or pandas Series (can contain NaNs)
    Returns pandas Series with same index as y (NaN where insufficient data).
    """
    from sklearn.linear_model import LinearRegression
    y_s = pd.Series(y).astype(float)
    x_s = pd.Series(x).astype(float)
    mask = x_s.notna() & y_s.notna()
    res = pd.Series(np.nan, index=y_s.index)
    if mask.sum() < 3:
        return res
    lr = LinearRegression(fit_intercept=fit_intercept)
    X = x_s[mask].values.reshape(-1, 1)
    lr.fit(X, y_s[mask].values)
    pred = lr.predict(X)
    res.loc[mask] = y_s[mask].values - pred
    return res

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

# ---------------- LOAD & CLEAN ----------------
df = pd.read_csv(DATA_PATH)

# Map month strings to ints if needed
month_map = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
             'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
if df[COL_MONTH].dtype == object:
    df[COL_MONTH] = df[COL_MONTH].map(month_map).astype(int)

need = [COL_YEAR, COL_MONTH, COL_DAY, COL_POOL_ELEV, COL_STORAGE2400,
        COL_REL_POWER, COL_REL_TOTAL, COL_EVAP, COL_INFLOW, COL_RAIN_DAM, COL_RAIN_BSN]
df = df[need].copy().sort_values([COL_YEAR, COL_MONTH, COL_DAY])

# --- Monthly aggregation rules ---
agg_spec = {
    COL_POOL_ELEV:   'mean',
    COL_STORAGE2400: 'last',
    COL_REL_POWER:   'sum',
    COL_REL_TOTAL:   'sum',
    COL_EVAP:        'sum',
    COL_INFLOW:      'sum',
    COL_RAIN_DAM:    'sum',
    COL_RAIN_BSN:    'sum',
}
monthly_df = df.groupby([COL_YEAR, COL_MONTH], as_index=False).agg(agg_spec)

# Core engineered totals
monthly_df['rainfall']     = monthly_df[COL_RAIN_BSN].astype(float).fillna(0.0)
monthly_df['rainfall_total'] = monthly_df[COL_RAIN_BSN] + monthly_df[COL_RAIN_DAM]
monthly_df['releases_sum'] = monthly_df[COL_REL_TOTAL]
monthly_df = monthly_df.sort_values([COL_YEAR, COL_MONTH]).reset_index(drop=True)
monthly_df['date'] = pd.to_datetime(dict(year=monthly_df[COL_YEAR], month=monthly_df[COL_MONTH], day=1))


df_daily = df.copy()
df_daily['date_day'] = pd.to_datetime(dict(year=df_daily[COL_YEAR],
                                           month=df_daily[COL_MONTH],
                                           day=df_daily[COL_DAY]),
                                      errors='coerce')
df_daily = df_daily.dropna(subset=['date_day']).sort_values('date_day')

# Use only Rain 7A to BSN (basin) as the rain feature
df_daily['rain_total'] = df_daily[COL_RAIN_BSN].astype(float).fillna(0.0)
df_daily['evap_day']   = df_daily[COL_EVAP].astype(float).fillna(0.0)
df_daily['storage_day']= df_daily[COL_STORAGE2400].astype(float)

ROLL_7  = 7
ROLL_14 = 14
ROLL_30 = 30

df_daily['rain_7d']   = df_daily['rain_total'].rolling(ROLL_7,  min_periods=1).sum()
df_daily['rain_14d']  = df_daily['rain_total'].rolling(ROLL_14, min_periods=1).sum()
df_daily['rain_30d']  = df_daily['rain_total'].rolling(ROLL_30, min_periods=1).sum()
df_daily['rain_max1d_30d'] = df_daily['rain_total'].rolling(ROLL_30, min_periods=1).max()

thr_map_90, global_thr_90 = None, None
try:
    thr_map_90, global_thr_90 = (lambda df_d: (build_quantile_maps_raw(df_d['rain_total'], df_d['rain_total'], df_d['date_day'].dt.month)[0:2]))(df_daily)
except Exception:
    pass

df_daily['month'] = df_daily['date_day'].dt.month
# fallback heavy rain logic using fixed threshold if fit failed
df_daily['hr_thresh_p90'] = df_daily['month'].apply(lambda m: 1.5)
df_daily['is_heavy_p90'] = (df_daily['rain_total'] >= df_daily['hr_thresh_p90']).astype(float)
df_daily['heavy_rain_days_30d'] = df_daily['is_heavy_p90'].rolling(ROLL_30, min_periods=1).sum()

daily_snap = (
    df_daily.set_index('date_day')
            .resample('ME')   # changed from 'M' -> 'ME' per FutureWarning
            .last()[[
                'rain_7d','rain_14d','rain_30d','rain_max1d_30d','heavy_rain_days_30d'
            ]]
            .reset_index()
)
daily_snap['year']  = daily_snap['date_day'].dt.year
daily_snap['month'] = daily_snap['date_day'].dt.month
daily_snap = daily_snap.drop(columns=['date_day'])

monthly_df = monthly_df.merge(daily_snap, on=[COL_YEAR, COL_MONTH], how='left')

# ---------------- ERA5 Soil Moisture (0–7 cm) ----------------
# Load hourly ERA5 soil moisture from GEE export and aggregate to monthly means

soil_df = pd.read_csv(SOIL_PATH)

# GEE export has 'time' and 'volumetric_soil_water_layer_1'
soil_df['time'] = pd.to_datetime(soil_df['time'])

if 'volumetric_soil_water_layer_1' not in soil_df.columns:
    raise ValueError(
        "Expected 'volumetric_soil_water_layer_1' in ERA5 CSV. "
        f"Columns found: {soil_df.columns.tolist()}"
    )

# Rename to something nicer
soil_df = soil_df.rename(columns={'volumetric_soil_water_layer_1': 'sm_top_0_7cm'})

# Daily average, then monthly average over the basin
soil_df['date_day'] = soil_df['time'].dt.normalize()
soil_df['year']     = soil_df['date_day'].dt.year
soil_df['month']    = soil_df['date_day'].dt.month

soil_monthly = (
    soil_df
    .groupby(['year', 'month'], as_index=False)['sm_top_0_7cm']
    .mean()
    .rename(columns={'year': COL_YEAR, 'month': COL_MONTH})
)

# Merge monthly soil moisture into the main monthly_df
monthly_df = monthly_df.merge(soil_monthly, on=[COL_YEAR, COL_MONTH], how='left')

# Simple soil-moisture lags / smoothing (prior-only)
monthly_df['sm_top_0_7cm_lag1'] = monthly_df['sm_top_0_7cm'].shift(1)
monthly_df['sm_top_0_7cm_lag3'] = monthly_df['sm_top_0_7cm'].shift(3)
monthly_df['sm_top_0_7cm_roll3'] = (
    monthly_df['sm_top_0_7cm']
    .shift(1)                              # only use prior months
    .rolling(window=3, min_periods=1)
    .mean()
)

# Rolling features (prior-only via shift(1))
monthly_df['rainfall_roll3'] = monthly_df['rainfall'].shift(1).rolling(window=3, min_periods=1).sum()
monthly_df['rainfall_roll6'] = monthly_df['rainfall'].shift(1).rolling(window=6, min_periods=1).sum()
monthly_df['inflow_same_month_last_year'] = monthly_df[COL_INFLOW].shift(12)
monthly_df['inflow_lag1']  = monthly_df[COL_INFLOW].shift(1)
monthly_df['inflow_lag3']  = monthly_df[COL_INFLOW].shift(3)
monthly_df['inflow_lag6']  = monthly_df[COL_INFLOW].shift(6)
monthly_df = add_global_lags(
    monthly_df,
    cols=['rainfall', COL_EVAP, COL_INFLOW, 'releases_sum', COL_POOL_ELEV, COL_STORAGE2400],
    n_lags=N_LAGS
)

monthly_df['mon_sin'] = np.sin(2*np.pi*monthly_df[COL_MONTH]/12.0)
monthly_df['mon_cos'] = np.cos(2*np.pi*monthly_df[COL_MONTH]/12.0)

if COL_RAIN_DAM in monthly_df.columns and COL_RAIN_BSN in monthly_df.columns:
    monthly_df['rain_dam_resid'] = resid(monthly_df[COL_RAIN_DAM], monthly_df[COL_RAIN_BSN])
   

monthly_df['y_next_inflow_raw'] = monthly_df[COL_INFLOW].shift(-1)
monthly_df['y_next_inflow_log'] = np.log1p(monthly_df['y_next_inflow_raw'])

# create smoothed basin + dam-residual features (prior-only)
monthly_df['rain_bsn_roll3'] = monthly_df['rainfall'].shift(1).rolling(window=3, min_periods=1).sum()
monthly_df['rain_bsn_roll12'] = monthly_df['rainfall'].shift(1).rolling(window=12, min_periods=1).sum()

# ORACLE FEATURES: leak next month's rain totals
#monthly_df['rain_bsn_next'] = monthly_df[COL_RAIN_BSN].shift(-1)
#monthly_df['rain_dam_next'] = monthly_df[COL_RAIN_DAM].shift(-1)
#monthly_df['rainfall_next'] = monthly_df['rainfall_total'].shift(-1)


# 1. Build climatology dictionaries from ALL years
bsn_clim = monthly_df.groupby(COL_MONTH)[COL_RAIN_BSN].mean().to_dict()
dam_clim = monthly_df.groupby(COL_MONTH)[COL_RAIN_DAM].mean().to_dict()
tot_clim = monthly_df.groupby(COL_MONTH)['rainfall_total'].mean().to_dict()

def next_month(m: int) -> int:
    m = int(m)
    return 1 if m == 12 else m + 1

# 2. For each row, use climatological mean of *next* calendar month
monthly_df['rain_bsn_next'] = monthly_df[COL_MONTH].apply(
    lambda m: bsn_clim[next_month(m)]
)
monthly_df['rain_dam_next'] = monthly_df[COL_MONTH].apply(
    lambda m: dam_clim[next_month(m)]
)
monthly_df['rainfall_next'] = monthly_df[COL_MONTH].apply(
    lambda m: tot_clim[next_month(m)]
)

# compact engineered features + final FEATURES set
engineered_feats = [
    'rain_bsn_roll3', 'rain_bsn_roll12',
    'rain_14d', 'rain_30d',
    'rain_max1d_30d', 
    'inflow_same_month_last_year',
    'inflow_lag6',
    'rain_dam_resid',
    'rain_bsn_next','rain_dam_next'
    ]

# NEW: soil moisture features
soil_feats = [
    'sm_top_0_7cm',
    'sm_top_0_7cm_lag1',
    'sm_top_0_7cm_lag3',
    'sm_top_0_7cm_roll3'
]

# keep only single recent inflow lag for baseline/autoregression
lag_feats = [f'{COL_INFLOW}_1']

FEATURES =  ['mon_sin', 'mon_cos'] + lag_feats + engineered_feats + soil_feats

TARGET_RAW  = 'y_next_inflow_raw'
TARGET_LOG  = 'y_next_inflow_log'

data = monthly_df[['date', COL_YEAR, COL_MONTH] + FEATURES + [TARGET_RAW, TARGET_LOG]].dropna().reset_index(drop=True)

# ---------------- Fixed TIME SPLIT ----------------
# optionally exclude specific year(s) from the training set
# set to [] to include all years, or e.g. [2010] to remove 2010 from training
EXCLUDE_YEARS = []

train_mask = (data[COL_YEAR] <= TRAIN_END) & (~data[COL_YEAR].isin(EXCLUDE_YEARS))
train = data[train_mask]
val   = data[(data[COL_YEAR] > TRAIN_END) & (data[COL_YEAR] <= VAL_END)]
test  = data[data[COL_YEAR] > VAL_END]

Xtr, Xv, Xt = train[FEATURES], val[FEATURES], test[FEATURES]
ytr_raw, yv_raw, yt_raw = train[TARGET_RAW], val[TARGET_RAW], test[TARGET_RAW]
ytr_log, yv_log, yt_log = train[TARGET_LOG], val[TARGET_LOG], test[TARGET_LOG]

imp = SimpleImputer(strategy='mean')
Xtr_i = imp.fit_transform(Xtr)
Xv_i  = imp.transform(Xv)
Xt_i  = imp.transform(Xt)

# ---------------- MODELS ----------------
lin = LinearRegression()

lin.fit(Xtr_i, ytr_log)

metrics = []

m_tr, pred_tr = eval_model_log('Linear', lin, Xtr_i, ytr_raw)
m_v,  pred_v  = eval_model_log('Linear', lin, Xv_i,  yv_raw)
m_t,  pred_t  = eval_model_log('Linear', lin, Xt_i,  yt_raw)
metrics += [dict(split='train', **m_tr), dict(split='val', **m_v), dict(split='test', **m_t)]

# ---------------- RIDGE REGRESSION ----------------
ridge = Ridge(alpha=1.0, random_state=0)
ridge.fit(Xtr_i, ytr_log)

m_tr_ridge, pred_tr_ridge = eval_model_log('Ridge', ridge, Xtr_i, ytr_raw)
m_v_ridge,  pred_v_ridge  = eval_model_log('Ridge', ridge,  Xv_i, yv_raw)
m_t_ridge,  pred_t_ridge  = eval_model_log('Ridge', ridge,  Xt_i, yt_raw)

metrics += [
    dict(split='train', **m_tr_ridge),
    dict(split='val',   **m_v_ridge),
    dict(split='test',  **m_t_ridge)
]

# ---------------- LASSO (CV) ----------------
lasso = LassoCV(cv=5, random_state=0)
lasso.fit(Xtr_i, ytr_log)

m_tr_lasso, pred_tr_lasso = eval_model_log('Lasso', lasso, Xtr_i, ytr_raw)
m_v_lasso,  pred_v_lasso  = eval_model_log('Lasso', lasso,  Xv_i, yv_raw)
m_t_lasso,  pred_t_lasso  = eval_model_log('Lasso', lasso,  Xt_i, yt_raw)

metrics += [
    dict(split='train', **m_tr_lasso),
    dict(split='val',   **m_v_lasso),
    dict(split='test',  **m_t_lasso)
]

# ---------------- RANDOM FOREST ----------------
if RF_AVAILABLE:
    rf = RandomForestRegressor(n_estimators=1200, max_depth=8, min_samples_leaf=3, random_state=0, n_jobs=-1)
    rf.fit(Xtr_i, ytr_log)
    m_tr_rf, pred_tr_rf = eval_model_log('RandomForest', rf, Xtr_i, ytr_raw)
    m_v_rf,  pred_v_rf  = eval_model_log('RandomForest', rf,  Xv_i, yv_raw)
    m_t_rf,  pred_t_rf  = eval_model_log('RandomForest', rf,  Xt_i, yt_raw)
    metrics += [dict(split='train', **m_tr_rf),dict(split='val', **m_v_rf), dict(split='test', **m_t_rf)]

# ---------------- XGBOOST ----------------
if XGB_AVAILABLE:
    xgb_model = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.05, max_depth=8,
                                subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', random_state=0)
    xgb_model.fit(Xtr_i, ytr_log)
    m_tr_xgb, pred_tr_xgb = eval_model_log('XGBoost', xgb_model, Xtr_i, ytr_raw)
    m_v_xgb, pred_v_xgb = eval_model_log('XGBoost', xgb_model, Xv_i, yv_raw)
    m_t_xgb, pred_t_xgb = eval_model_log('XGBoost', xgb_model, Xt_i, yt_raw)
    metrics += [dict(split='train', **m_tr_xgb),dict(split='val', **m_v_xgb), dict(split='test', **m_t_xgb)]

# ---------------- BASELINE_Last ----------------
def get_inflow_lag1(split_df):
    lag1_col = f'{COL_INFLOW}_1'
    if lag1_col not in split_df.columns:
        raise KeyError(f"Missing lag column {lag1_col}.")
    return split_df[lag1_col].values

val_last  = get_inflow_lag1(val)
test_last = get_inflow_lag1(test)
b_v_last = extra_metrics(yv_raw, val_last);  b_v_last.update({"name":"Baseline_Last","split":"val"})
b_t_last = extra_metrics(yt_raw, test_last); b_t_last.update({"name":"Baseline_Last","split":"test"})
metrics += [b_v_last, b_t_last]

metrics_df = pd.DataFrame(metrics).sort_values(['split','RMSE','name'])
print("\n=== Metrics (fixed split; lower is better) ===")
print(metrics_df)
metrics_df.to_csv(OUT_DIR / "metrics_fixed_split.csv", index=False)

# ---------------- BUILD TEST PREDICTIONS ----------------
test_out = test[['date', COL_YEAR, COL_MONTH]].copy()
test_out['y_true']  = yt_raw.values
test_out['y_lin']   = pred_t
test_out['y_ridge'] = pred_t_ridge
test_out['y_lasso'] = pred_t_lasso

test_out['y_last']  = test_last
if RF_AVAILABLE:
    test_out['y_rf'] = pred_t_rf
if XGB_AVAILABLE:
    test_out['y_xgb'] = pred_t_xgb

# ---- Build CI bands (original-scale residual quantiles) ----
val_preds  = {'Linear': pred_v}
test_preds = {'Linear': pred_t}
if RF_AVAILABLE:
    val_preds['RandomForest']  = pred_v_rf
    test_preds['RandomForest'] = pred_t_rf
if XGB_AVAILABLE:
    val_preds['XGBoost']  = pred_v_xgb
    test_preds['XGBoost'] = pred_t_xgb

month_val  = val[COL_MONTH]
month_test = test[COL_MONTH]
name_to_col = {
    'Linear':'y_lin',
    'Ridge':'y_ridge',
    'Lasso':'y_lasso',
    'RandomForest':'y_rf',
    'XGBoost':'y_xgb'
}


for name, pv in val_preds.items():
    if pv is None:
        continue
    qlo90_map, qhi90_map, glo90, ghi90, qlo50_map, qhi50_map, glo50, ghi50 = build_quantile_maps_raw(
        y_true_raw=yv_raw.values,
        y_pred_raw=pv,
        months=month_val,
        min_n_month=10,
        min_n_season=20
    )
    yhat_raw = test_preds.get(name, None)
    if yhat_raw is None:
        continue

    qlo90 = month_test.apply(lambda m: q_for_month(m, qlo90_map, glo90)).to_numpy()
    qhi90 = month_test.apply(lambda m: q_for_month(m, qhi90_map, ghi90)).to_numpy()
    qlo50 = month_test.apply(lambda m: q_for_month(m, qlo50_map, glo50)).to_numpy()
    qhi50 = month_test.apply(lambda m: q_for_month(m, qhi50_map, ghi50)).to_numpy()

    lo90 = yhat_raw + qlo90
    hi90 = yhat_raw + qhi90
    lo50 = yhat_raw + qlo50
    hi50 = yhat_raw + qhi50

    col = name_to_col[name]
    test_out[f'{col}_lo90'] = lo90
    test_out[f'{col}_hi90'] = hi90
    test_out[f'{col}_lo50'] = lo50
    test_out[f'{col}_hi50'] = hi50

# ---- Plot Linear CI bands explicitly (if computed) ----
if 'y_lin_lo90' in test_out.columns and 'y_lin_hi90' in test_out.columns:
    plot_with_ci_band(test_out, actual_col='y_true', pred_col='y_lin',
                      lo_col='y_lin_lo90', hi_col='y_lin_hi90',
                      title='Linear Prediction with 90% CI',
                      fname='test_timeline_linear_ci_90.png', out_dir=OUT_DIR)
if 'y_lin_lo50' in test_out.columns and 'y_lin_hi50' in test_out.columns:
    plot_with_ci_band(test_out, actual_col='y_true', pred_col='y_lin',
                      lo_col='y_lin_lo50', hi_col='y_lin_hi50',
                      title='Linear Prediction with 50% CI',
                      fname='test_timeline_linear_ci_50.png', out_dir=OUT_DIR)

# ---------------- RAINFALL PDFS & PLOTS ----------------
for m in range(1, 13):
    plot_month_rain_pdf(monthly_df, m, OUT_DIR)

# If RF available, still create RF CI plots
if RF_AVAILABLE and 'y_rf' in test_out.columns:
    plot_with_ci_band(test_out, actual_col='y_true', pred_col='y_rf',
                      lo_col='y_rf_lo90', hi_col='y_rf_hi90',
                      title='RandomForest Prediction with 90% CI',
                      fname='test_timeline_rf_ci_90.png', out_dir=OUT_DIR)
    plot_with_ci_band(test_out, actual_col='y_true', pred_col='y_rf',
                      lo_col='y_rf_lo50', hi_col='y_rf_hi50',
                      title='RandomForest Prediction with 50% CI',
                      fname='test_timeline_rf_ci_50.png', out_dir=OUT_DIR)

# Build series for timeline: always include Actual and Baseline; include any model preds present
series = {'Actual': test_out['y_true'].values, 'Baseline: Last': test_out['y_last'].values}
series['Ridge'] = test_out['y_ridge'].values
series['Lasso'] = test_out['y_lasso'].values

for name, col in name_to_col.items():
    if col in test_out.columns:
        # label by model name (e.g., 'XGBoost', 'Ridge', 'Linear', 'RandomForest')
        series[name] = test_out[col].values


# Final timeline plot with whatever series we have (ensures XGBoost appears if it was best/was run)
plot_test_timeline(test_out, series, OUT_DIR, fname="test_timeline_fixed.png", title="Test (fixed split)")

# Save test predictions
test_out = test_out.sort_values('date')
test_out.to_csv(OUT_DIR / "predictions_test_fixed.csv", index=False)
print(f"\nOutputs written to: {OUT_DIR}")
print(" - metrics_fixed_split.csv")
print(" - predictions_test_fixed.csv")
print(" - rain_pdf_MM_MON.png (12 files)")
print(" - test_timeline_fixed.png")




# DAILY → NEXT-30-DAY INFLOW MODEL (rolling-window features)

print("\n===== DAILY NEXT-30-DAY INFLOW MODEL =====")

# 1. Build / clean daily dataframe (df_daily already exists)
daily = df_daily.copy().sort_values('date_day').reset_index(drop=True)

# Make sure inflow is float
daily[COL_INFLOW] = daily[COL_INFLOW].astype(float)

# 2. Merge in ERA5 daily soil moisture (top 0–7 cm) from GEE export
soil_df = pd.read_csv(SOIL_PATH)
soil_df['time'] = pd.to_datetime(soil_df['time'])

if 'volumetric_soil_water_layer_1' not in soil_df.columns:
    raise ValueError(
        "Expected 'volumetric_soil_water_layer_1' in ERA5 CSV. "
        f"Columns found: {soil_df.columns.tolist()}"
    )

soil_df = soil_df.rename(columns={'volumetric_soil_water_layer_1': 'sm_top_0_7cm'})
soil_df['date_day'] = soil_df['time'].dt.floor('D')

soil_daily = (
    soil_df.groupby('date_day', as_index=False)['sm_top_0_7cm']
           .mean()
)

# Merge into daily
daily = daily.merge(soil_daily, on='date_day', how='left')

# 3. Rolling-window features (rain, inflow, soil moisture, storage)

# Base series
rain = daily['rain_total'].astype(float).fillna(0.0)
qin  = daily[COL_INFLOW].astype(float).fillna(0.0)
sm   = daily['sm_top_0_7cm'].astype(float)


# Rainfall rolling windows (using data up to and including today)
daily['rain_1d']   = rain
daily['rain_3d']   = rain.rolling(3,  min_periods=1).sum()
daily['rain_7d']   = rain.rolling(7,  min_periods=1).sum()
daily['rain_14d']  = rain.rolling(14, min_periods=1).sum()
daily['rain_30d']  = rain.rolling(30, min_periods=1).sum()

# Inflow rolling windows (use only past inflow: shift by 1 day)
qin_shift = qin.shift(1)
daily['qin_lag1']   = qin_shift
daily['qin_3d']     = qin_shift.rolling(3,  min_periods=1).mean()
daily['qin_7d']     = qin_shift.rolling(7,  min_periods=1).mean()
daily['qin_14d']    = qin_shift.rolling(14, min_periods=1).mean()
daily['qin_30d']    = qin_shift.rolling(30, min_periods=1).mean()

# Soil moisture rolling windows (including today)
daily['sm_1d']      = sm
daily['sm_7d_mean'] = sm.rolling(7,  min_periods=1).mean()
daily['sm_14d_mean']= sm.rolling(14, min_periods=1).mean()
daily['sm_30d_mean']= sm.rolling(30, min_periods=1).mean()


# Seasonality features (day of year)
daily['doy']     = daily['date_day'].dt.dayofyear
daily['doy_sin'] = np.sin(2*np.pi*daily['doy']/365.25)
daily['doy_cos'] = np.cos(2*np.pi*daily['doy']/365.25)

# 4. Target: sum of inflow over the NEXT 30 days (rolling window)
#
# inflow_next30(t) = sum of inflow from t+1 to t+30
daily['inflow_next30'] = (
    qin.shift(-1)
       .rolling(30, min_periods=30)
       .sum()
)

daily['inflow_next30_log'] = np.log1p(daily['inflow_next30'])

# 5. Build modeling dataframe

daily['year'] = daily['date_day'].dt.year

daily_features = [
    'rain_1d','rain_3d','rain_7d','rain_14d','rain_30d',
    'qin_lag1','qin_3d','qin_7d','qin_14d','qin_30d',
    'sm_1d','sm_7d_mean',#'sm_14d_mean','sm_30d_mean',
    'doy_sin','doy_cos'
]

daily_model_df = daily[['date_day','year'] + daily_features + ['inflow_next30','inflow_next30_log']].dropna().reset_index(drop=True)

# 6. Time-based split (same TRAIN_END / VAL_END logic)

train_d = daily_model_df[daily_model_df['year'] <= TRAIN_END]
val_d   = daily_model_df[(daily_model_df['year'] > TRAIN_END) & (daily_model_df['year'] <= VAL_END)]
test_d  = daily_model_df[daily_model_df['year'] > VAL_END]

Xd_tr, Xd_v, Xd_t = train_d[daily_features], val_d[daily_features], test_d[daily_features]
yd_tr_raw, yd_v_raw, yd_t_raw = train_d['inflow_next30'], val_d['inflow_next30'], test_d['inflow_next30']
yd_tr_log, yd_v_log, yd_t_log = train_d['inflow_next30_log'], val_d['inflow_next30_log'], test_d['inflow_next30_log']

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

# 9. Choose best model on test (lowest MAE)
test_metrics = metrics_daily_df[metrics_daily_df['split'] == 'test'].sort_values('MAE')
best_name = test_metrics.iloc[0]['name']
print(f"\nBest daily next-30d model on test: {best_name}")


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
    title=f'{best_name} Next 30-Day Inflow with 90% CI (Test)',
    fname='test_timeline_daily_next30_ci90.png',
    out_dir=OUT_DIR
)

plot_with_ci_band(
    df=df_plot_90,
    actual_col='y_true',
    pred_col='y_pred',
    lo_col='y_lo50',
    hi_col='y_hi50',
    title=f'{best_name} Next 30-Day Inflow with 50% CI (Test)',
    fname='test_timeline_daily_next30_ci50.png',
    out_dir=OUT_DIR
)

print(f"Saved: predictions_daily_next30_with_ci.csv")