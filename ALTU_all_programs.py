# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 17:43:47 2025

@author: ej_st
"""

# Set desired test month and year
current_month = 1
current_year = 2024
month=current_month

#-------------------------------------------import libraries-------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from scipy.stats import gamma
import matplotlib.pyplot as plt
import calendar
import datetime


#----------------------df creation and formatting--------------------------------------------
today = datetime.date.today()
c_year = today.year
months =['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'] 
years = list(range(c_year-30,c_year))
num_rowsskip=10

def read_one_txt_file_header(txt_file, mnth, yr):
    num_rows_skip = num_rowsskip
    num_foot_skip = 12
    df = pd.read_csv(txt_file, skipfooter=num_foot_skip, header=None, 
                     skiprows=num_rows_skip,sep='\\s+', engine='python')
    cln_df = df.dropna(axis=0,how='any',)
    #cln_df = df.dropna(axis='TOTAL',how='any',)
    col_names = ['day','pool','elevations (ft)','storage (2400hr)','releases (power)',
                 'releases (total)','evap inches','inflow adj',
                 'rainfall inches (7A to Dam)','rainfall inches (7A to BSN)'] 
    cln_df.columns = col_names
    cln_df['month'] = mnth
    cln_df['year'] = yr
    return cln_df

df_list = []

for month in months:
    for year in years:
        txt_file = f"ALTU_{month}_{year}.txt"
        if year < 2018:   
            num_rowsskip=10
        elif year ==2018:
            if month == "JAN":
                num_rowsskip=11
            else:
                num_rowsskip=11
    df_mnth_yr = read_one_txt_file_header(txt_file, month, year)
    df_list.append(df_mnth_yr)
combined_df = pd.concat(df_list)
combined_df = combined_df.fillna(0)    
combined_df.to_csv("ALTU_ALL.csv")


        
#-------------------------------------------set up format for dates-------------------------------------------------

# Load data
df = pd.read_csv('ALTU_ALL.csv')

# Format month to text
month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
             'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
if df['month'].dtype == object:
    df['month'] = df['month'].map(month_map)

# Aggregate data monthly by sum of data
monthly_df = df.groupby(['year', 'month'], sort=False).sum().reset_index()

#-------------------------------------------Data Calculations-------------------------------------------------

# Calculate rainfall sum of basin and reservoir total
monthly_df['rainfall'] = monthly_df['rainfall inches (7A to Dam)'] + monthly_df['rainfall inches (7A to BSN)']

#Create future data fields for rainfall and inflow
monthly_df['rainfall__1'] = monthly_df['rainfall'].shift(-1)
monthly_df['inflow__1'] = monthly_df['inflow adj'].shift(-1)

#create data points for past 6 month of rainfall, evaporation, and inflow
for i in range(1, 6):
    monthly_df[f'rainfall_{i}'] = monthly_df['rainfall'].shift(i)
for i in range(1, 6):
    monthly_df[f'evap_{i}'] = monthly_df['evap inches'].shift(i)
for i in range(1, 6):
    monthly_df[f'inflow_{i}'] = monthly_df['inflow adj'].shift(i)

#----------------------------------------------Create bell curves for normal distribution---------------------------------------------

def plot_normal_bell_curve_by_month(df):
    #create for loop to make bell curve for each of 12 months with rainfall as input
    for i in range(1, 13):
        data = df.loc[df['month'] == i, 'rainfall'].dropna()
        
        # set mu as data mean and sigma as standard deviation
        mu, sigma = data.mean(), data.std()
        
        #create x values for plot
        x_vals = np.linspace(data.min(), data.max(), 100)
        
        #create y values for plot using normal distribution formula
        y_vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x_vals - mu)**2) / (2 * sigma**2))

        #set bin(column) count and size
        counts, bins = np.histogram(data, bins=15, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
        #set colors to a gradient based on standard deviations(sigma)
        colors = []
        for x in bin_centers:
            z = abs((x - mu) / sigma)
            if z <= 1:
                colors.append('green')
            elif z <= 2:
                colors.append('orange')
            else:
                colors.append('red')
        
        #create plot
        plt.figure(figsize=(8, 4))
        
        #create bars
        for j in range(len(bin_centers)):
            plt.bar(bin_centers[j], counts[j], width=(bins[1] - bins[0]),
                    color=colors[j], align='center', alpha=0.6)

        #format and label plot
        plt.plot(x_vals, y_vals, label='Normal PDF', color='blue')
        plt.title(f'Rainfall Distribution (Normal) - {calendar.month_abbr[i]}')
        plt.xlabel('Rainfall (inches)')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
  
#----------------------------Create bell curve for gamma distribution-----------------------------------------    
        
def plot_gamma_curve_by_month(df):
    
    #make bounds dictionary
    gamma_bounds = {}
    
    #create for loop to make a curve for each month
    for i in range(1, 13):
        data = df.loc[df['month'] == i, 'rainfall'].dropna()
        
        #set parameters using gamma.fit
        shape, loc, scale = gamma.fit(data)

        #set bounds value(alpha) and create bounds
        alpha = 0.05
        lower = gamma.ppf(alpha / 2, a=shape, loc=loc, scale=scale)
        upper = gamma.ppf(1 - alpha / 2, a=shape, loc=loc, scale=scale)
        gamma_bounds[i] = (lower, upper)
        
        #set x values
        x_min = max(data.min() - data.std(), 0)
        x_max = data.max() + data.std()
        
        #set y values using gamma.fit
        x_vals = np.linspace(x_min, x_max, 200)
        y_vals = gamma.pdf(x_vals, a=shape, loc=loc, scale=scale)

        #set bins(columns) and bin size
        bin_count = min(15, max(12, int(np.sqrt(len(data)))))
        counts, bins = np.histogram(data, bins=bin_count, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
        #set colors as gradient based on gamma bounds
        colors = ['green' if lower <= x <= upper else 'red' for x in bin_centers]

        #create plot
        plt.figure(figsize=(8, 4))
        
        #add bins
        for j in range(len(bin_centers)):
            plt.bar(bin_centers[j], counts[j], width=(bins[1] - bins[0]),
                    color=colors[j], align='center', alpha=0.6)
        #plot data
        plt.plot(x_vals, y_vals, 'k-', lw=2, label='Gamma PDF')
        
        #format plot
        plt.title(f'Rainfall Distribution (Gamma) - {calendar.month_abbr[i]}')
        plt.xlabel('Rainfall (inches)')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    #return bounds for later prediction
    return gamma_bounds

#------------------------determine distribution for normal and gamma distirbution---------------------------

def get_distribution_stats(df, month):
    #filter data to only contain values associated with desired month
    data = df.loc[df['month'] == month, 'rainfall'].dropna()
    data = data[data > 0]

    # Normal distribution calculations
    normal_mean = data.mean()
    normal_std = data.std()

    # Gamma distribution calulations
    
    shape, loc, scale = gamma.fit(data, floc=0)
    gamma_mean = shape * scale
    gamma_ci_50 = gamma.interval(0.50, a=shape, loc=loc, scale=scale)
    gamma_ci_95 = gamma.interval(0.95, a=shape, loc=loc, scale=scale)


    return {
        'normal_mean': normal_mean,
        'normal_std': normal_std,
        'gamma_mean': gamma_mean,
        'gamma_ci_50': gamma_ci_50,
        'gamma_ci_95': gamma_ci_95
    }
def get_historical_inflow_mean(df, month):
    inflow_data = df.loc[df['month'] == month, 'inflow__1'].dropna()
    if len(inflow_data) == 0:
        return None
    return inflow_data.mean()


stats = get_distribution_stats(monthly_df, month)
normal_mean = stats['normal_mean']
gamma_mean = stats['gamma_mean']
hist_mean = monthly_df.loc[(monthly_df['month'] == month) & (monthly_df['rainfall'] > 0), 'rainfall'].mean()
print(f"Average rainfall for {calendar.month_name[month]}: {hist_mean:.2f} inches")
print(f"Normal Mean for {calendar.month_name[month]}: {normal_mean:.2f} inches")
print(f"Gamma Mean for {calendar.month_name[month]}: {gamma_mean:.2f} inches")

#-------------------------Call normal distribution bell curve function---------------------------------------

plot_normal_bell_curve_by_month(monthly_df)

#-------------------------Call gamma distribution bell curve function---------------------------------------

gamma_bounds = plot_gamma_curve_by_month(monthly_df)

#-------------------------Create linear prediction fuction---------------------------------------

def predict_linear_inflow(df, feature_cols, target_col, current_month):
    model_data = df[feature_cols + [target_col]].dropna()
    X = model_data[feature_cols]
    y = model_data[target_col]

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    model = LinearRegression(positive=True)
    model.fit(X_imputed, y)

    # Predict inflow for current_month in all past years
    month_data = df[df['month'] == current_month].copy()
    month_data = month_data.dropna(subset=feature_cols)
    month_features = imputer.transform(month_data[feature_cols])
    month_data['predicted_inflow'] = model.predict(month_features)

    return month_data[['year', 'predicted_inflow']]

#---------------------Create annotation function for gamma prediction---------------------------------

def annotate_point(ax, x, y, label):
    ax.annotate(f"{label:.2f}", xy=(x, y), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    
#-----------------Create plot and prediction using normal distribution----------------------------------    
    
def plot_normal_prediction(linear_preds, normal_stats, month_name, current_year):
    recent = linear_preds.sort_values(by='year', ascending=False).head(12).sort_values(by='year')
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot regression predictions (last 12 months)
    ax.plot(recent['year'], recent['predicted_inflow'], 'bo-', label='Linear Regression')
    for x, y in zip(recent['year'], recent['predicted_inflow']):
        annotate_point(ax, x, y, y)

    # Plot normal prediction
    if normal_stats:
        mean = normal_stats['normal_mean']
        std = normal_stats['normal_std']

        # ±1σ
    ax.errorbar([current_year + 0.1], [mean], yerr=[[std], [std]],
                fmt='o', color='orange', label='Normal Prediction ±1σ')
    annotate_point(ax, current_year + 0.1, mean, mean)
    annotate_point(ax, current_year + 0.09, mean + std, mean + std)
    annotate_point(ax, current_year + 0.11, mean - std, mean - std)
    
    # ±2σ
    ax.errorbar([current_year + 0.2], [mean], yerr=[[2 * std], [2 * std]],
                fmt='o', color='darkorange', label='Normal Prediction ±2σ')
    annotate_point(ax, current_year + 0.2, mean, mean)
    annotate_point(ax, current_year + 0.19, mean + 2 * std, mean + 2 * std)
    annotate_point(ax, current_year + 0.21, mean - 2 * std, mean - 2 * std)

    ax.set_title(f"Linear Regression + Normal Prediction – {month_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Inflow (units)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
#-----------------------------------Create plot and prediction using gamma distribution--------------------------------------
    
def plot_gamma_prediction(linear_preds, gamma_stats, month_name, current_year):
    recent = linear_preds.sort_values(by='year', ascending=False).head(12).sort_values(by='year')
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot regression predictions
    ax.plot(recent['year'], recent['predicted_inflow'], 'bo-', label='Linear Regression')
    for x, y in zip(recent['year'], recent['predicted_inflow']):
        annotate_point(ax, x, y, y)

    # Gamma prediction
    if gamma_stats:
        mean = gamma_stats['gamma_mean']
        ci_50 = gamma_stats['gamma_ci_50']
        ci_95 = gamma_stats['gamma_ci_95']

        low_50 = mean - ci_50[0]
        up_50 = ci_50[1] - mean
        low_95 = mean - ci_95[0]
        up_95 = ci_95[1] - mean

        # 50% CI
    ax.errorbar([current_year + 0.1], [mean], yerr=[[low_50], [up_50]],
                fmt='o', color='green', label='Gamma Prediction 50% CI')
    annotate_point(ax, current_year + 0.1, mean, mean)
    annotate_point(ax, current_year + 0.09, mean + up_50, mean + up_50)
    annotate_point(ax, current_year + 0.11, mean - low_50, mean - low_50)
    
    # 95% CI
    ax.errorbar([current_year + 0.2], [mean], yerr=[[low_95], [up_95]],
                fmt='o', color='darkgreen', label='Gamma Prediction 95% CI')
    annotate_point(ax, current_year + 0.2, mean, mean)
    annotate_point(ax, current_year + 0.19, mean + up_95, mean + up_95)
    annotate_point(ax, current_year + 0.21, mean - low_95, mean - low_95)

    ax.set_title(f"Linear Regression + Gamma Prediction – {month_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Inflow (units)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
month = current_month
month_name = calendar.month_name[month]

# Gather data
stats = get_distribution_stats(monthly_df, month)
# Extract normal and gamma stats if available
normal_stats = {'normal_mean': stats.get('normal_mean'),
    'normal_std': stats.get('normal_std')}

gamma_stats = {    'gamma_mean': stats.get('gamma_mean'),
    'gamma_ci_50': stats.get('gamma_ci_50'),
    'gamma_ci_95': stats.get('gamma_ci_95')}
linear_preds = predict_linear_inflow(monthly_df,
    feature_cols=['rainfall_1', 'rainfall', 'rainfall__1', 'evap inches'],
    target_col='inflow__1', current_month=month)

#----------------------------------------------Call defined functins---------------------------------------------

# Plot combined graphs
plot_normal_prediction(linear_preds, stats, current_month, current_year)
plot_gamma_prediction(linear_preds, stats, current_month, current_year)
        