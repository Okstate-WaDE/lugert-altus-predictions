# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:13:02 2025

@author: ej_st
"""
#------------------------------------Linear regression data----------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

#-----------------------------------------Bring data in--------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from datetime import datetime
from scipy import stats
import calendar
import math
import sys
current_month = 1
df = pd.read_csv('ALTU_ALL.csv')

#----------------------------------------Format data and time--------------------------------------------

monthly_df = df.groupby(['year', 'month'], sort=False).sum().reset_index()

month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
month=1
if monthly_df['month'].dtype == object:
    monthly_df['month'] = monthly_df['month'].map(month_map)

monthly_df['date'] = pd.to_datetime(monthly_df[['year', 'month']].assign(day=1))

columns_keep = ['storage (2400hr)', 'rainfall inches (7A to Dam)', 'rainfall inches (7A to BSN)',
                'evap inches', 'releases (total)']
        
#-------------------------------------------Data Calculations-------------------------------------------------

monthly_df['rainfall'] = monthly_df['rainfall inches (7A to Dam)'] + monthly_df['rainfall inches (7A to BSN)']
monthly_df['rainfall__1'] = monthly_df['rainfall'].shift(-1)
monthly_df['inflow__1'] = monthly_df['inflow adj'].shift(-1)
for i in range(1, 13):
    monthly_df[f'rainfall_{i}'] = monthly_df['rainfall'].shift(i)
for i in range(1, 13):
    monthly_df[f'evap_{i}'] = monthly_df['evap inches'].shift(i)
monthly_df['inflow_1'] = monthly_df['inflow adj'].shift(1)
monthly_df['inflow_2'] = monthly_df['inflow adj'].shift(2)
monthly_df['inflow_3'] = monthly_df['inflow adj'].shift(3)
monthly_df['inflow_4'] = monthly_df['inflow adj'].shift(4)

#----------------------------------------Test for Normal Distribution-----------------------------------------
def test_normal_distribution(column):
    for i in range(1,13):
        ND_test_data = monthly_df.loc[monthly_df[column]==i,'rainfall']
        statistic, p_value = stats.shapiro(ND_test_data)
        print(f"Shapiro-Wilk Test Statistic for {i}: {statistic}")
        print(f"P-value for {i}: {p_value}")
print('---Normal Distribution Test Results---')
test_normal_distribution('month')
print(' ')

#-------------------------------------------Bell Curve Generation---------------------------------------------

def make_bell_curve(column):
    for i in range(1, 13):
        data = monthly_df.loc[monthly_df[column] == i, 'rainfall'].dropna()
        
        if data.empty:
            print(f"No data for month {i}")
            continue

        mu = data.mean()
        sigma = data.std()

        # Histogram
        counts, bins = np.histogram(data, bins=15, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # Assign colors based on distance from mean
        colors = []
        for x in bin_centers:
            z = abs(x - mu) / sigma
            if z < 1:
                colors.append('green')       # Within 1σ
            elif z < 2:
                colors.append('orange')      # 1σ to 2σ
            else:
                colors.append('red')         # > 2σ

        # Plot histogram with colored bars
       # plt.figure(figsize=(8, 4))
        for j in range(len(bin_centers)):
            plt.bar(bin_centers[j], counts[j], width=(bins[1] - bins[0]), color=colors[j], align='center', alpha=0.6)

        # Plot normal distribution curve
        x_vals = np.linspace(data.min(), data.max(), 100)
        y_vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x_vals - mu) ** 2) / (2 * sigma ** 2))
        plt.plot(x_vals, y_vals, label='Normal PDF', color='blue')

        # Title and labels
        month_label = calendar.month_abbr[i]
        plt.title(f'Rainfall Distribution for {month_label}')
        plt.xlabel('Rainfall (inches)')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        plt.show()
make_bell_curve('month')

#--------------------------------------Define Inflow Linear Regression----------------------------------------

def simulate_future_inflow_predictions(monthly_df, month,
         feature_cols=['rainfall_1', 'rainfall', 'rainfall__1','evap inches'],
         target_col='inflow__1'):
   # ['rainfall_1', 'rainfall_2', 'rainfall_3', 'rainfall', 'rainfall__1','evap inches','evap_1','evap_2'],
    """
    Train a linear regression model once and use historical rainfall data to simulate predictions for next month's inflow.
    Parameters:
        monthly_df (pd.DataFrame): DataFrame with lag features precomputed.
        feature_cols (list): Feature columns used in training and prediction.
        target_col (str): Target column for prediction (usually 'inflow__1').
        month (int): Optional. If None, use the latest available month.
    Returns:
        pd.DataFrame: Predictions for each year based on simulated next-month rainfall.
    """
    
    df = monthly_df.copy()

    # Default to latest available month in data if not provided
    if month is None:
        month = df['month'].max()

    # 1. Train the model once
    model_data = df[feature_cols + [target_col]].dropna()
    X = model_data[feature_cols]
    y = model_data[target_col]
    model_data = df[feature_cols + [target_col]]
    model_data = model_data[model_data[target_col].notna()]
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=feature_cols)
    model = LinearRegression(positive=True)
    model.fit(X, y)
    coefficients = model.coef_
    intercept = model.intercept_
    
    # 2. Simulate rainfall from past years for the "next month"
    predictions = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_eval = LinearRegression(positive=True)
    model_eval.fit(X_train, y_train)
    y_pred = model_eval.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"R² Score: {r2:.3f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print( )
    print("\n--- Linear Regression Formula ---")
    formula = f"inflow__1 = {intercept:.3f}"
    for coef, feature in zip(coefficients, feature_cols):
        formula += f" + ({coef:.3f} * {feature})"
    print(formula)
    print()
    for year in df['year'].unique():
        # Look up this year's row for the current month
        row = df[(df['year'] == year) & (df['month'] == current_month)]
        if row.empty:
            continue

        # Extract the feature values
        try:
            rainfall_pred = row['rainfall'].values[0]
            rainfall_pred = max(rainfall_pred, 0)  # Clip to zero if negative
            
            input_data = {
                'evap inches': row['evap inches'].values[0],
                'evap_1': row['evap_1'].values[0],
                'evap_2': row['evap_2'].values[0],
                'rainfall_1': row['rainfall_1'].values[0],
                'rainfall_2': row['rainfall_2'].values[0],
                'rainfall_3': row['rainfall_3'].values[0],
                'rainfall': row['rainfall'].values[0],
                'rainfall__1': rainfall_pred}
        except IndexError:
            continue

        # Impute any missing value with column means
        for col in feature_cols:
            if pd.isna(input_data[col]):
                input_data[col] = X[col].mean()

        input_array = np.array([input_data[col] for col in feature_cols])
        predicted_inflow = np.dot(coefficients, input_array) + intercept

        predictions.append({
            'Year (used)': year,
            'Month used for rainfall': current_month,
            'Simulated rainfall__1': input_data['rainfall__1'],
            'Predicted inflow__1': round(max(predicted_inflow, 0), 2)})

    return pd.DataFrame(predictions), model, feature_cols

historical_ave = monthly_df.loc[monthly_df['month'] == current_month, 'inflow adj'].mean()
simulated_results, trained_model, used_features = simulate_future_inflow_predictions(monthly_df, 1)
print(simulated_results)
print(f"Minimum predicted: {simulated_results['Predicted inflow__1'].min()}")
print(f"Maximum predicted: {simulated_results['Predicted inflow__1'].max()}")
print(f"Mean predicted: {simulated_results['Predicted inflow__1'].mean():.2f}")
print(f' Historical Average: {historical_ave}')

# Step 1: Create a DataFrame of predictions grouped by month
simulated_results['month'] = current_month  # all predictions are for same month
simulated_results_grouped = simulated_results.groupby('month').agg(
    Mean_Predicted=('Predicted inflow__1', 'mean'),
    Min_Predicted=('Predicted inflow__1', 'min'),
    Max_Predicted=('Predicted inflow__1', 'max')).reset_index()

# Step 2: Get the historical actual inflow for the next month
historical_actual = (monthly_df[monthly_df['month'] == month]
    .groupby('month')
    .agg(Historical_Mean_Actual=('inflow adj', 'mean'))
    .reset_index())

# ---------------------- Visualization of Last 12 Months + Prediction using max,min,ave.--------------------------

# Step 1: Get last 12 months of actual inflow data
last_12_months = monthly_df.sort_values('date').dropna(subset=['inflow adj']).tail(12)

# Step 2: Prepare data
dates = last_12_months['date']
inflows = last_12_months['inflow adj']

# Step 3: Calculate predicted inflow stats
pred_mean = simulated_results['Predicted inflow__1'].mean()
pred_min = simulated_results['Predicted inflow__1'].min()
pred_max = simulated_results['Predicted inflow__1'].max()

# Step 4: Create the "next month" date (1 month after the last date)
next_month_date = dates.max() + pd.DateOffset(months=1)

# Step 5: Build full x-axis dates and labels
all_dates = dates.tolist() + [next_month_date]
month_labels = [d.strftime('%b') for d in all_dates]  

# Step 6: Plot
plt.figure(figsize=(12, 6))
plt.plot(dates, inflows, marker='o', label='Observed Inflow (Past 12 Months)', color='blue')

# Add predicted point with error bars
plt.errorbar(
    next_month_date,
    pred_mean,
    yerr=[[pred_mean - pred_min], [pred_max - pred_mean]],
    fmt='o',
    color='red',
    ecolor='orange',
    capsize=5,
    label='Predicted Inflow (Next Month)')
plt.text(next_month_date, pred_min - 5, f"{pred_min:.1f}", ha='center', va='top', fontsize=9, color='black')
plt.text(next_month_date, pred_max + 5, f"{pred_max:.1f}", ha='center', va='top', fontsize=9, color='black')
plt.text(next_month_date, pred_mean, f"{pred_mean:.1f}", ha='center', va='top', fontsize=9, color='black')
# Customize x-axis with month names
plt.xticks(all_dates, month_labels, rotation=45)

# Formatting
plt.axvline(next_month_date, linestyle='--', color='gray', alpha=0.5)
plt.title('Observed Inflow (Last 12 Months) + Predicted Next Month using max,min, and ave.')
plt.xlabel('Month')
plt.ylabel('Inflow (adj)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#------------------------------Test Gamma Distribution------------------------------------------

from scipy.stats import gamma, kstest

def test_gamma_distribution(column):
    print('--- Gamma Distribution Test Results ---')
    for i in range(1, 13):
        data = monthly_df.loc[monthly_df[column] == i, 'rainfall'].dropna()
        
        if data.empty:
            print(f"Month {i}: No data available.")
            continue

        # Fit gamma distribution
        params = gamma.fit(data)
        shape, loc, scale = params

        # Perform K-S test
        D, p_value = kstest(data, 'gamma', args=params)

        print(f"Month {i} (Shape={shape:.2f}, Loc={loc:.2f}, Scale={scale:.2f})")
        print(f"  K-S Test D-statistic: {D:.4f}")
        print(f"  P-value: {p_value:.4f}")

        
test_gamma_distribution('month')

def make_gamma_curve(column):
    gamma_std_dev_bounds = {}
    for i in range(1, 13):
        data = monthly_df.loc[monthly_df[column] == i, 'rainfall'].dropna()

        if len(data) < 5:
            print(f"Skipping month {i} due to insufficient data ({len(data)} samples).")
            continue

        # Fit gamma distribution
        params = gamma.fit(data)
        shape, loc, scale = params
        confidence_level = 0.95
        alpha = 1 - confidence_level
        lower_bound = gamma.ppf(alpha / 2, params, scale)
        upper_bound = gamma.ppf(1 - alpha / 2, params, scale)
        # Dynamic bin count
        
        gamma_mean = gamma.mean(shape, loc=loc, scale=scale)
       

        
        gamma_std_dev_bounds[i] = (lower_bound, upper_bound)
        
        bin_count = min(15, max(12, int(np.sqrt(len(data)))))
        counts, bins = np.histogram(data, bins=bin_count, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        '''colors = []
        for x in bin_centers:
            z = abs(x - gamma_mean) / gamma_std
            if z < 1:
                colors.append('green')
            elif z < 2:
                colors.append('orange')
            else:
                colors.append('red')'''
                
        plt.figure(figsize=(8, 4))
        for j in range(len(bin_centers)):
            plt.bar(bin_centers[j], counts[j], width=(bins[1] - bins[0]), 
                    color='green', align='center', alpha=0.6)
        
        if counts.sum() == 0:
            print(f"Month {i} has non-zero data but histogram is flat.")
        plt.ylim(bottom=0)
        
        # Gamma PDF
        x_vals = np.linspace(data.min(), data.max(), 100)
        y_vals = gamma.pdf(x_vals, a=shape, loc=loc, scale=scale)
        plt.plot(x_vals, y_vals, 'r-', lw=2, label='Gamma PDF')

        # Labels
        month_label = calendar.month_abbr[i]
        plt.title(f'Rainfall Distribution for {month_label} (Gamma Fit)')
        plt.xlabel('Rainfall (inches)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()     
        return gamma_std_dev_bounds
make_gamma_curve('month')

# ---------------------- Visualization of Last 12 Months + Prediction using gamma dist. --------------------------

def make_gamma_curve(column):
    gamma_std_dev_bounds = {}
    for i in range(1, 13):
        data = monthly_df.loc[monthly_df[column] == i, 'rainfall'].dropna()

        if len(data) < 5:
            print(f"Skipping month {i} due to insufficient data ({len(data)} samples).")
            continue

        # Fit gamma distribution
        shape, loc, scale = gamma.fit(data)

        # Calculate 95% confidence interval bounds
        alpha = 0.05
        lower_bound = gamma.ppf(alpha / 2, a=shape, loc=loc, scale=scale)
        upper_bound = gamma.ppf(1 - alpha / 2, a=shape, loc=loc, scale=scale)
        gamma_std_dev_bounds[i] = (lower_bound, upper_bound)

        # Histogram setup
        bin_count = min(15, max(12, int(np.sqrt(len(data)))))
        counts, bins = np.histogram(data, bins=bin_count, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # Color bars based on confidence interval
        colors = []
        for x in bin_centers:
            if lower_bound <= x <= upper_bound:
                colors.append('green')  # Inside 95% CI
            else:
                colors.append('red')    # Outside 95% CI

        # Plotting
        plt.figure(figsize=(8, 4))
        for j in range(len(bin_centers)):
            plt.bar(bin_centers[j], counts[j], width=(bins[1] - bins[0]),
                    color=colors[j], align='center', alpha=0.6)

        # Gamma PDF line
        x_vals = np.linspace(data.min(), data.max(), 100)
        y_vals = gamma.pdf(x_vals, a=shape, loc=loc, scale=scale)
        plt.plot(x_vals, y_vals, 'k-', lw=2, label='Gamma PDF')

        # Labels
        month_label = calendar.month_abbr[i]
        plt.title(f'Rainfall Distribution for {month_label} (Gamma Fit)')
        plt.xlabel('Rainfall (inches)')
        plt.ylabel('Density')
        plt.axvline(lower_bound, color='blue', linestyle='--', linewidth=1, label='95% CI')
        plt.axvline(upper_bound, color='blue', linestyle='--', linewidth=1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return gamma_std_dev_bounds
simulated_results, trained_model, used_features = simulate_future_inflow_predictions(monthly_df, 1)
gamma_bounds = make_gamma_curve('month')  # returns ±2σ bounds

scenario_preds = make_gamma_curve(
    trained_model, used_features, gamma_bounds, current_month, monthly_df)

print(scenario_preds)

# Existing 12-month inflow + prediction plot
plt.figure(figsize=(12, 6))
plt.plot(dates, inflows, marker='o', label='Observed Inflow (Past 12 Months)', color='blue')

# Plot each scenario prediction
for _, row in scenario_preds.iterrows():
    plt.scatter(next_month_date, row['Predicted Inflow'], label=row['Scenario'], s=80)
    plt.text(next_month_date, row['Predicted Inflow'] + 10, f"{row['Predicted Inflow']:.1f}",
             ha='center', fontsize=9)

# Axis and formatting
plt.xticks(all_dates, month_labels, rotation=45)
plt.axvline(next_month_date, linestyle='--', color='gray', alpha=0.5)
plt.title('Observed Inflow (Last 12 Months) + Predicted Next Month using gamma dist.')
plt.xlabel('Month')
plt.ylabel('Inflow (adj)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#---------------------------Predictions using Normal Distribution-------------------------------
def calculate_normal_std_dev_bounds(monthly_df, column='month'):
    normal_bounds = {}
    for i in range(1, 13):
        data = monthly_df.loc[monthly_df[column] == i, 'rainfall'].dropna()
        if len(data) < 5:
            continue
        mean = data.mean()
        std_dev = data.std()
        lower_bound = mean - 2 * std_dev
        upper_bound = mean + 2 * std_dev
        normal_bounds[i] = (lower_bound, upper_bound)
    return normal_bounds

normal_bounds = calculate_normal_std_dev_bounds(monthly_df)

scenario_preds_normal = predict_inflow_for_rainfall_scenarios(
    trained_model, used_features, normal_bounds, current_month, monthly_df)

plt.figure(figsize=(12, 6))
plt.plot(dates, inflows, marker='o', label='Observed Inflow (Past 12 Months)', color='blue')

# Plot normal distribution-based scenarios
for _, row in scenario_preds_normal.iterrows():
    plt.scatter(next_month_date, row['Predicted Inflow'], label=row['Scenario'], s=80)
    plt.text(next_month_date, row['Predicted Inflow'] + 10, f"{row['Predicted Inflow']:.1f}",
             ha='center', fontsize=9)

# Format the plot
plt.xticks(all_dates, month_labels, rotation=45)
plt.axvline(next_month_date, linestyle='--', color='gray', alpha=0.5)
plt.title('Observed Inflow (Last 12 Months) + Predicted Next Month using normal dist.')
plt.xlabel('Month')
plt.ylabel('Inflow (adj)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()