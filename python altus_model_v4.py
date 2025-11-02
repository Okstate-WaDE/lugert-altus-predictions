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
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
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
    # robust MAPE: ignore tiny denominators to avoid huge values
    y_true_arr = np.asarray(y_true, dtype=float)
    denom = np.where(np.abs(y_true_arr) >= 1e-3, np.abs(y_true_arr), np.nan)
    mape_vals = np.abs((y_true_arr - np.asarray(y_pred, dtype=float)) / denom)
    mape = float(np.nanmean(mape_vals) * 100.0) if np.isfinite(np.nanmean(mape_vals)) else np.nan
    acc  = float(100.0 * (1.0 - mae / (np.nanmean(y_true_arr) + 1e-6)))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape, "Accuracy_%": acc}

def eval_model_log(name, model, X_i, y_true_raw):
    """Models trained on log1p(target); evaluate on original scale."""
    pred_log = model.predict(X_i)
    pred_raw = np.expm1(pred_log)
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

# residualize dam vs basin so dam only contributes unique information
mask = monthly_df[[ 'rainfall', COL_RAIN_DAM ]].dropna().index
if len(mask) > 10:
    lr = LinearRegression()
    X = monthly_df.loc[mask, ['rainfall']].values.reshape(-1,1)
    y = monthly_df.loc[mask, COL_RAIN_DAM].values
    lr.fit(X, y)
    monthly_df['rain_dam_resid'] = monthly_df[COL_RAIN_DAM].astype(float).fillna(0.0) - lr.predict(monthly_df[['rainfall']].fillna(0.0).values.reshape(-1,1))
else:
    monthly_df['rain_dam_resid'] = monthly_df[COL_RAIN_DAM].astype(float).fillna(0.0)
# ---------------- DAILY-BASED FEATURES ----------------
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
ROLL_30 = 45

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

df_daily['storage_diff']     = df_daily['storage_day'].diff()
df_daily['storage_chg_7d']   = df_daily['storage_diff'].rolling(ROLL_7,  min_periods=1).sum()
df_daily['storage_chg_30d']  = df_daily['storage_diff'].rolling(ROLL_30, min_periods=1).sum()

stor_max = float(np.nanmax(df_daily['storage_day']))
df_daily['storage_pct_max'] = np.where(stor_max > 0, df_daily['storage_day'] / stor_max, np.nan)

df_daily['evap_30d'] = df_daily['evap_day'].rolling(ROLL_30, min_periods=1).sum()
df_daily['rain_minus_evap_30d'] = df_daily['rain_30d'] - df_daily['evap_30d']

daily_snap = (
    df_daily.set_index('date_day')
            .resample('ME')   # changed from 'M' -> 'ME' per FutureWarning
            .last()[[
                'rain_7d','rain_14d','rain_30d','rain_max1d_30d','heavy_rain_days_30d',
                'storage_pct_max','storage_chg_7d','storage_chg_30d',
                'evap_30d','rain_minus_evap_30d'
            ]]
            .reset_index()
)
daily_snap['year']  = daily_snap['date_day'].dt.year
daily_snap['month'] = daily_snap['date_day'].dt.month
daily_snap = daily_snap.drop(columns=['date_day'])

monthly_df = monthly_df.merge(daily_snap, on=[COL_YEAR, COL_MONTH], how='left')

# Rolling features (prior-only via shift(1))
monthly_df['rainfall_roll3'] = monthly_df['rainfall'].shift(1).rolling(window=3, min_periods=1).sum()
monthly_df['rainfall_roll6'] = monthly_df['rainfall'].shift(1).rolling(window=6, min_periods=1).sum()
monthly_df['inflow_same_month_last_year'] = monthly_df[COL_INFLOW].shift(12)

monthly_df = add_global_lags(
    monthly_df,
    cols=['rainfall', COL_EVAP, COL_INFLOW, 'releases_sum', COL_POOL_ELEV, COL_STORAGE2400],
    n_lags=N_LAGS
)

monthly_df['mon_sin'] = np.sin(2*np.pi*monthly_df[COL_MONTH]/12.0)
monthly_df['mon_cos'] = np.cos(2*np.pi*monthly_df[COL_MONTH]/12.0)

monthly_df['y_next_inflow_raw'] = monthly_df[COL_INFLOW].shift(-1)
monthly_df['y_next_inflow_log'] = np.log1p(monthly_df['y_next_inflow_raw'])

# create smoothed basin + dam-residual features (prior-only)
monthly_df['rain_bsn_roll3'] = monthly_df['rainfall_total'].shift(1).rolling(window=3, min_periods=1).sum()
monthly_df['rain_bsn_roll12'] = monthly_df['rainfall_total'].shift(1).rolling(window=12, min_periods=1).sum()

# winsorize dam residual then smooth (prior-only)
monthly_df['rain_dam_resid'] = np.asarray(monthly_df['rain_dam_resid'], dtype=float)
lo_hi = np.nanquantile(monthly_df['rain_dam_resid'].dropna(), [0.01, 0.99])
monthly_df['rain_dam_resid_w'] = np.clip(monthly_df['rain_dam_resid'], lo_hi[0], lo_hi[1])
monthly_df['rain_dam_resid_roll3'] = monthly_df['rain_dam_resid_w'].shift(1).rolling(window=3, min_periods=1).mean()

# ORACLE FEATURES: leak next month's rain totals
monthly_df['rain_bsn_next'] = monthly_df[COL_RAIN_BSN].shift(-1)
monthly_df['rain_dam_next'] = monthly_df[COL_RAIN_DAM].shift(-1)
monthly_df['rainfall_next'] = monthly_df['rainfall_total'].shift(-1)
# compact engineered features + final FEATURES set
engineered_feats = [
    'rain_bsn_roll3', 'rain_bsn_roll12',
    'rain_14d', 'rain_30d',
    'rain_bsn_next','rain_dam_next'
]


# keep only single recent inflow lag for baseline/autoregression
lag_feats = [f'{COL_INFLOW}_1']

FEATURES = ['mon_sin', 'mon_cos'] + lag_feats + engineered_feats

TARGET_RAW  = 'y_next_inflow_raw'
TARGET_LOG  = 'y_next_inflow_log'

data = monthly_df[['date', COL_YEAR, COL_MONTH] + FEATURES + [TARGET_RAW, TARGET_LOG]].dropna().reset_index(drop=True)

# ---------------- Fixed TIME SPLIT ----------------
train = data[data[COL_YEAR] <= TRAIN_END]
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
lin   = LinearRegression()
ridge = Ridge(alpha=1.0, random_state=0)

lin.fit(Xtr_i, ytr_log)
ridge.fit(Xtr_i, ytr_log)

metrics = []

m_tr, pred_tr = eval_model_log('Linear', lin, Xtr_i, ytr_raw)
m_v,  pred_v  = eval_model_log('Linear', lin, Xv_i,  yv_raw)
m_t,  pred_t  = eval_model_log('Linear', lin, Xt_i,  yt_raw)
metrics += [dict(split='train', **m_tr), dict(split='val', **m_v), dict(split='test', **m_t)]

m_tr_r, pred_tr_r = eval_model_log('Ridge', ridge, Xtr_i, ytr_raw)
m_v_r,  pred_v_r  = eval_model_log('Ridge', ridge, Xv_i,  yv_raw)
m_t_r,  pred_t_r  = eval_model_log('Ridge', ridge, Xt_i,  yt_raw)
metrics += [dict(split='train', **m_tr_r), dict(split='val', **m_v_r), dict(split='test', **m_t_r)]

if RF_AVAILABLE:
    rf = RandomForestRegressor(n_estimators=1200, max_depth=8, min_samples_leaf=3, random_state=0, n_jobs=-1)
    rf.fit(Xtr_i, ytr_log)
    m_v_rf,  pred_v_rf  = eval_model_log('RandomForest', rf,  Xv_i, yv_raw)
    m_t_rf,  pred_t_rf  = eval_model_log('RandomForest', rf,  Xt_i, yt_raw)
    metrics += [dict(split='val', **m_v_rf), dict(split='test', **m_t_rf)]

if XGB_AVAILABLE:
    xgb_model = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.05, max_depth=8,
                                subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', random_state=0)
    xgb_model.fit(Xtr_i, ytr_log)
    m_v_xgb, pred_v_xgb = eval_model_log('XGBoost', xgb_model, Xv_i, yv_raw)
    m_t_xgb, pred_t_xgb = eval_model_log('XGBoost', xgb_model, Xt_i, yt_raw)
    metrics += [dict(split='val', **m_v_xgb), dict(split='test', **m_t_xgb)]

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
test_out['y_ridge'] = pred_t_r
test_out['y_last']  = test_last
if RF_AVAILABLE:
    test_out['y_rf'] = pred_t_rf
if XGB_AVAILABLE:
    test_out['y_xgb'] = pred_t_xgb

# ---- Build CI bands (original-scale residual quantiles) ----
val_preds  = {'Linear': pred_v, 'Ridge': pred_v_r}
test_preds = {'Linear': pred_t, 'Ridge': pred_t_r}
if RF_AVAILABLE:
    val_preds['RandomForest']  = pred_v_rf
    test_preds['RandomForest'] = pred_t_rf
if XGB_AVAILABLE:
    val_preds['XGBoost']  = pred_v_xgb
    test_preds['XGBoost'] = pred_t_xgb

month_val  = val[COL_MONTH]
month_test = test[COL_MONTH]
name_to_col = {'Linear':'y_lin', 'Ridge':'y_ridge', 'RandomForest':'y_rf', 'XGBoost':'y_xgb'}

hist_cap = float(data[TARGET_RAW].max() * 1.10)

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

    lo90 = np.clip(lo90, 0.0, None)
    hi90 = np.minimum(hi90, hist_cap) if np.isfinite(hist_cap) else hi90
    lo50 = np.clip(lo50, 0.0, None)
    hi50 = np.minimum(hi50, hist_cap) if np.isfinite(hist_cap) else hi50

    col = name_to_col[name]
    test_out[f'{col}_lo90'] = lo90
    test_out[f'{col}_hi90'] = hi90
    test_out[f'{col}_lo50'] = lo50
    test_out[f'{col}_hi50'] = hi50

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
for name, col in name_to_col.items():
    if col in test_out.columns:
        # label by model name (e.g., 'XGBoost', 'Ridge', 'Linear', 'RandomForest')
        series[name] = test_out[col].values

# Highlight best test model (lowest MAE) and plot its CI if available
best_model_name = None
try:
    best_row = metrics_df[metrics_df['split'] == 'test'].sort_values('MAE').iloc[0]
    best_model_name = best_row['name']
except Exception:
    best_model_name = None

if best_model_name is not None:
    best_col = name_to_col.get(best_model_name)
    if best_col in test_out.columns:
        # add explicit "Best: <name>" series for emphasis
        series[f'Best: {best_model_name}'] = test_out[best_col].values
        # plot CI bands for best model if CI columns exist
        lo90, hi90 = f'{best_col}_lo90', f'{best_col}_hi90'
        lo50, hi50 = f'{best_col}_lo50', f'{best_col}_hi50'
        if lo90 in test_out.columns and hi90 in test_out.columns:
            plot_with_ci_band(test_out, actual_col='y_true', pred_col=best_col,
                              lo_col=lo90, hi_col=hi90,
                              title=f'{best_model_name} Prediction with 90% CI',
                              fname=f'test_timeline_{best_model_name.lower()}_ci_90.png', out_dir=OUT_DIR)
        if lo50 in test_out.columns and hi50 in test_out.columns:
            plot_with_ci_band(test_out, actual_col='y_true', pred_col=best_col,
                              lo_col=lo50, hi_col=hi50,
                              title=f'{best_model_name} Prediction with 50% CI',
                              fname=f'test_timeline_{best_model_name.lower()}_ci_50.png', out_dir=OUT_DIR)

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

# Check for high multicollinearity
corr = train[FEATURES].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.90)]
print("Highly correlated -> consider dropping:", to_drop)

# Feature importance
rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
rf.fit(Xtr_i, ytr_raw)                    # or ytr_log if you prefer
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp.head(30))

perm = permutation_importance(rf, Xv_i, yv_raw, n_repeats=10, random_state=0, n_jobs=-1)
perm_series = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
print(perm_series.head(30))

lasso = LassoCV(cv=5, random_state=0).fit(Xtr_i, ytr_log)
coef = pd.Series(lasso.coef_, index=FEATURES).sort_values()
print("Non-zero Lasso features:", coef[coef!=0].index.tolist())

# RFECV feature selection
selector = RFECV(Ridge(alpha=1.0), step=1, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
selector.fit(Xtr_i, ytr_log)
selected = [f for keep,f in zip(selector.support_, FEATURES) if keep]
print("RFECV selected:", selected)

# residualize dam vs basin so dam only contributes unique information
mask = monthly_df[[ 'rainfall', COL_RAIN_DAM ]].dropna().index
if len(mask) > 10:
    lr = LinearRegression()
    X = monthly_df.loc[mask, ['rainfall']].values.reshape(-1,1)
    y = monthly_df.loc[mask, COL_RAIN_DAM].values
    lr.fit(X, y)
    monthly_df['rain_dam_resid'] = monthly_df[COL_RAIN_DAM].astype(float).fillna(0.0) - lr.predict(monthly_df[['rainfall']].fillna(0.0).values.reshape(-1,1))
else:
    monthly_df['rain_dam_resid'] = monthly_df[COL_RAIN_DAM].astype(float).fillna(0.0)

# --- safe VIF and ablation test ---
# drop zero-variance and duplicate columns for VIF calculation
Xtr_df = pd.DataFrame(Xtr_i, columns=FEATURES).copy()
# drop constant cols
var = Xtr_df.var(axis=0)
const_cols = var[var <= 1e-8].index.tolist()
if const_cols:
    print("Dropping constant cols for VIF:", const_cols)
    Xtr_df = Xtr_df.drop(columns=const_cols)
# drop duplicate columns (perfect duplicates)
dupes = Xtr_df.T.duplicated()
dup_cols = Xtr_df.columns[dupes].tolist()
if dup_cols:
    print("Dropping duplicate cols for VIF:", dup_cols)
    Xtr_df = Xtr_df.drop(columns=dup_cols)

# compute VIF on cleaned frame
try:
    vif = pd.Series([variance_inflation_factor(Xtr_df.values, i) for i in range(Xtr_df.shape[1])],
                    index=Xtr_df.columns).sort_values(ascending=False)
    print("VIF (top 10):\n", vif.head(10))
except Exception as e:
    print("VIF calculation failed:", e)

# Ablation: compare Ridge val MAE with and without rain_dam_resid
if 'rain_dam_resid' in FEATURES:
    feats_with = FEATURES.copy()
    feats_without = [f for f in FEATURES if f != 'rain_dam_resid']

    # fit separate imputers for each feature subset (must match fit/transform dims)
    imp_w = SimpleImputer(strategy='mean')
    Xtr_w = imp_w.fit_transform(train[feats_with])
    Xv_w  = imp_w.transform(val[feats_with])

    imp_wo = SimpleImputer(strategy='mean')
    Xtr_wo = imp_wo.fit_transform(train[feats_without])
    Xv_wo  = imp_wo.transform(val[feats_without])

    # train small Ridge models on log-target (same as main workflow)
    r_with = Ridge(alpha=1.0, random_state=0).fit(Xtr_w, ytr_log)
    pred_v_w = np.expm1(r_with.predict(Xv_w))
    mae_w = mean_absolute_error(yv_raw, pred_v_w)

    r_wo = Ridge(alpha=1.0, random_state=0).fit(Xtr_wo, ytr_log)
    pred_v_wo = np.expm1(r_wo.predict(Xv_wo))
    mae_wo = mean_absolute_error(yv_raw, pred_v_wo)

    print(f"Ablation (val MAE) with rain_dam_resid: {mae_w:.1f}, without: {mae_wo:.1f}")