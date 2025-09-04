# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:35:25 2025

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

df = pd.read_csv('ALTU_ALL.csv')

#----------------------------------------Format data and time--------------------------------------------

monthly_df = df.groupby(['year', 'month'], sort=False).mean().reset_index()

month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

if monthly_df['month'].dtype == object:
    monthly_df['month'] = monthly_df['month'].map(month_map)

monthly_df['date'] = pd.to_datetime(monthly_df[['year', 'month']].assign(day=1))

columns_keep = ['storage (2400hr)', 'rainfall inches (7A to Dam)', 'rainfall inches (7A to BSN)',
                'evap inches', 'releases (total)']

#-------------------------------------------Data Calculations-------------------------------------------------

monthly_df['rainfall'] = monthly_df['rainfall inches (7A to Dam)'] + monthly_df['rainfall inches (7A to BSN)']
#monthly_df['rainfall 3month sum'] = monthly_df['rainfall'].rolling(window=4, min_periods=1).sum()
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

#-----------------------------------User Input for Linear Regression------------------------------------
'''
desired_x_cols = ['rainfall','rainfall_1','rainfall_2','rainfall_3','rainfall_4']
desired_y_col = 'inflow adj'

#------------------------------------------Input processing--------------------------------------------

feature_cols = columns_keep + desired_x_cols
X = monthly_df[feature_cols]
y = monthly_df[desired_y_col]

# Replace NaNs in X with the mean of each column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=feature_cols)
'''
dates = monthly_df['date']
'''
#-------------------------------------------Train/Test Split---------------------------------------------------

X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, dates, test_size=0.2, random_state=42)

#------------------------------------------Linear Regression--------------------------------------------------

model = LinearRegression(positive=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#------------------------------------R^2 and Error Value Calculation----------------------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

#------------------------------------Scatter Plot: Actual vs Predicted-----------------------------------

residuals = y_test - y_pred
threshold = 2 * np.std(residuals)
outlier_mask = np.abs(residuals) > threshold

plt.figure(figsize=(8, 6))

plt.scatter(y_test[~outlier_mask], y_pred[~outlier_mask], color='blue', alpha=0.6, label='Inliers')

plt.scatter(y_test[outlier_mask], y_pred[outlier_mask], color='red', alpha=0.8, label='Outliers')

slope, intercept = np.polyfit(y_test, y_pred, 1)
best_fit_line = slope * y_test + intercept
ideal_line = 1 * y_test + 0

plt.plot(y_test, best_fit_line, color='green', lw=2, label='Trend Line')
plt.plot(y_test, ideal_line, color='orange', lw=2, label='1:1 Line')

plt.text(x=0.7 * 1250,y=0.5 * 0,s=f"$R^2$ = {r2:.3f}",fontsize=12,verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))
plt.text(x=0.7 * 1250,y=0.5 * 200,s=f"Y = {slope:.3f}x + {intercept:.3f}",fontsize=12,verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))

coefs = model.coef_
intercept = model.intercept_

filtered_terms = [f"{coefs[X.columns.get_loc(col)]:.2f}{col}"
    for col in desired_x_cols if col in X.columns
]
formula_str = " + ".join(filtered_terms)
full_formula = f"y = {intercept:.2f} + " + formula_str

plt.text(x=0.05 * max(y_test), y=1400, s=full_formula,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.4", edgecolor='gray', facecolor='white'))

plt.xlabel(f'Actual {desired_y_col}')
plt.ylabel(f'Predicted {desired_y_col}')
plt.title(f'Actual vs Predicted {desired_y_col}')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

#------------------------------------Time Line Plot: Actual vs Predicted-----------------------------------

results_df = pd.DataFrame({'Date': dates_test,f'Actual {desired_y_col}': y_test,
    f'Predicted {desired_y_col}': y_pred}).sort_values('Date')

plt.figure(figsize=(10, 6))

plt.plot(results_df['Date'], results_df[f'Actual {desired_y_col}'],
         label=f'Actual {desired_y_col}', marker='o')

plt.plot(results_df['Date'], results_df[f'Predicted {desired_y_col}'],
         label=f'Predicted {desired_y_col}', marker='x', linestyle='--')

plt.title(f'Predicted vs Actual {desired_y_col} Over Time')
plt.xlabel('Date')
plt.ylabel(desired_y_col)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''
#--------------------Using linear regression results and past data, creating predictions---------------------
#------------------------------------------------------------------------------------------------------------

# Set up features and target
feature_cols = ['rainfall','rainfall_1','rainfall_2','rainfall_3','evap inches','evap_1','evap_2']
X = monthly_df[feature_cols]
y = monthly_df['rainfall__1']
dates = monthly_df['date']  # Replace with actual column name

# Combine to drop NaNs consistently
data = pd.concat([X, y], axis=1).dropna()

# Filter X, y, and dates to match the same indices
X = data[feature_cols]
y = data['rainfall__1']
dates = dates.loc[data.index]  # Filter dates accordingly

# Impute remaining NaNs in X
from sklearn.impute import SimpleImputer
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=feature_cols)

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, dates, test_size=0.2, random_state=42)
# Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

latest_input = X.tail(1)  
next_month_rainfall = model.predict(latest_input)[0]
print(f"Predicted Rainfall for Next Month: {next_month_rainfall:.2f} inches")

# Optional: Plot actual vs predicted rainfall
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Next-Month Rainfall')
plt.ylabel('Predicted Next-Month Rainfall')
plt.title('Predicted vs Actual Rainfall (Next Month)')
plt.grid(True)
plt.show()


results_df = pd.DataFrame({'Date': dates_test,'Actual rainfall': y_test,
    'Predicted rainfall': y_pred}).sort_values('Date')

plt.figure(figsize=(10, 6))

plt.plot(results_df['Date'], results_df['Actual rainfall'],
         label='Actual rainfall', marker='o')

plt.plot(results_df['Date'], results_df['Predicted rainfall'],
         label='Predicted rainfall', marker='x', linestyle='--')

plt.title('Predicted vs Actual rainfall Over Time')
plt.xlabel('Date')
plt.ylabel('rainfall')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
