# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 18:50:09 2025

@author: ej_st
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ALTU_ALL.csv')

monthly_df=df.groupby(['year','month'], sort=False).mean()
#group by year and then month
columns_keep = ['storage (2400hr)','inflow adj','rainfall inches (7A to Dam)', \
                'rainfall inches (7A to BSN)','evap inches','releases (total)']
monthly_df=monthly_df[columns_keep]
#Drop all unneccesary data for now
monthly_df['storage change'] = monthly_df['storage (2400hr)'].diff()
#determine rain sum and storage change

monthly_df['inflow_sum_shift_1']=monthly_df['inflow adj'].shift(1)
monthly_df['inflow_sum_shift_2']=monthly_df['inflow adj'].shift(2)
monthly_df['inflow_sum_shift_3']=monthly_df['inflow adj'].shift(3)
monthly_df['inflow_sum_shift_4']=monthly_df['inflow adj'].shift(4)
#shift copies down to prepare for sum

monthly_df['4 month inflow'] = monthly_df['inflow adj']+monthly_df['inflow_sum_shift_1']+monthly_df \
    ['inflow_sum_shift_2']+monthly_df['inflow_sum_shift_3']+monthly_df['inflow_sum_shift_4']
#sum last four months worth of inflow

monthly_df['4 month inflow change'] = monthly_df['4 month inflow'].diff()
#determine difference between monthly inflow sums

monthly_df['rainfall_sum'] = monthly_df['rainfall inches (7A to Dam)']+ monthly_df['rainfall inches (7A to BSN)']
#sum the two different rain values

monthly_df['Rainfall_sum_shift_1']=monthly_df['rainfall_sum'].shift(1)
monthly_df['Rainfall_sum_shift_2']=monthly_df['rainfall_sum'].shift(2)
monthly_df['Rainfall_sum_shift_3']=monthly_df['rainfall_sum'].shift(3)
#shift copies down to prepare for sum

monthly_df['4 month rain'] = monthly_df['rainfall_sum']+monthly_df['Rainfall_sum_shift_1']+monthly_df \
    ['Rainfall_sum_shift_2']+monthly_df['Rainfall_sum_shift_3']
#sum rain from past four months

monthly_df['4 month rain change'] = monthly_df['4 month rain'].diff()
#determine difference between rain sums

monthly_df['total_input_change (ref)']=monthly_df['4 month rain change']+monthly_df['4 month inflow change'] \
    -monthly_df['evap inches']
    #make column to reference to for correlation in terms of best correlation to date
    
monthly_df['total_input_change (test)']=monthly_df['4 month rain change']+monthly_df['4 month inflow change'] \
    -monthly_df['evap inches']-monthly_df['releases (total)']
    #make column to test to for correlation
correlation_new = monthly_df['storage change'].corr(monthly_df['total_input_change (test)'])
correlation_ref = monthly_df['storage change'].corr(monthly_df['total_input_change (ref)'])
#determine correlation values

monthly_df.drop(columns=['inflow_sum_shift_1','inflow_sum_shift_2','inflow_sum_shift_3','inflow_sum_shift_4'], inplace=True)
monthly_df.drop(columns=['Rainfall_sum_shift_1','inflow adj','Rainfall_sum_shift_2','Rainfall_sum_shift_3', \
                         'rainfall inches (7A to Dam)','rainfall inches (7A to BSN)','evap inches','releases (total)'], \
                inplace=True)
    #drop all unneccesary columns

fig, ax1 = plt.subplots()
plt.xticks(rotation=45)
sns.scatterplot(data=monthly_df, x='month', y='storage change',
                ax=ax1, color='blue')

ax2 = ax1.twinx()
sns.scatterplot(data=monthly_df, x='month', y='4 month rain change'
                ,ax=ax2, color='green')

fig, ax1 = plt.subplots()
plt.xticks(rotation=45)
sns.scatterplot(data=monthly_df, x='month', y='storage change',
                ax=ax1, color='blue')
"""
ax2 = ax1.twinx()
sns.scatterplot(data=monthly_df, x='month', y='total_input_change (ref)'
                ,ax=ax2, color='green')

#monthly_df.to_csv("ALTU_ANALYSIS_w_FOUR_MONTH_INFLOW_SUMS.csv")
"""