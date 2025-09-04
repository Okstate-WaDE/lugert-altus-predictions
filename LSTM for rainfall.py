# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:53:45 2025

@author: ej_st
"""


#------------------------------------Linear regression data----------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

#-----------------------------------------Bring data in--------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
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
#monthly_df['rainfall__1'].dropna(inplace=True)
monthly_df['rainfall_1'] = monthly_df['rainfall'].shift(1)
monthly_df['rainfall_2'] = monthly_df['rainfall'].shift(2)
monthly_df['rainfall_3'] = monthly_df['rainfall'].shift(3)
monthly_df['rainfall_4'] = monthly_df['rainfall'].shift(4)

monthly_df['water change'] = monthly_df['inflow adj'].shift(1) + monthly_df['rainfall'] - monthly_df['evap inches'] - monthly_df['releases (total)']
monthly_df['water carryover'] = monthly_df['storage (2400hr)'] + monthly_df['water change']
monthly_df['water carryover from last month'] = monthly_df['water carryover'].shift(1)
monthly_df['storage of next month'] = monthly_df['storage (2400hr)'].shift(-1)
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

dates = monthly_df['date']

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


# Step 1: Extract and scale rainfall data
rainfall_series = monthly_df[['date', 'rainfall']].dropna().sort_values('date')
rainfall_values = rainfall_series['rainfall'].values.reshape(-1, 1)

# Scale data to [0,1]
scaler = MinMaxScaler()
rainfall_scaled = scaler.fit_transform(rainfall_values)

# Step 2: Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 12  # use past 12 months
X, y = create_sequences(rainfall_scaled, seq_length)

# Step 3: Train/test split
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the model
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8,
                    validation_data=(X_test, y_test), verbose=1)

# Predict and inverse scale
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test_inv)), y_test_inv, label='Actual')
plt.plot(range(len(y_pred_inv)), y_pred_inv, label='Predicted')
plt.title('LSTM Forecast of Rainfall')
plt.xlabel('Time Step')
plt.ylabel('Rainfall (inches)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Use the last 12 months to predict the next
last_sequence = rainfall_scaled[-seq_length:].reshape(1, seq_length, 1)
next_month_scaled = model.predict(last_sequence)
next_month_rainfall = scaler.inverse_transform(next_month_scaled)[0, 0]

print(f"Predicted Rainfall for Next Month: {next_month_rainfall:.2f} inches")