# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:23:56 2025

@author: ej_st
"""

import pandas as pd
import seaborn as sns

df = pd.read_csv('ALTU_ALL.csv')

monthly_df=df.groupby('month', sort=False).mean()

columns_keep = ['storage (2400hr)','rainfall inches (7A to Dam)','rainfall inches (7A to BSN)']
monthly_df_rain=monthly_df[columns_keep]

new_rows={'storage (2400hr)':['NaN','NaN','NaN'],'rainfall inches (7A to Dam)':['NaN','NaN','NaN'],
          'rainfall inches (7A to BSN)':['NaN','NaN','NaN']}
new_rows=pd.DataFrame(new_rows)

monthly_rain_extra_df=pd.concat([monthly_df_rain,new_rows],ignore_index=True)
monthly_rain_analysis=monthly_rain_extra_df
monthly_rain_analysis['rainfall_sum'] = monthly_rain_analysis['rainfall inches (7A to Dam)']+ monthly_rain_analysis['rainfall inches (7A to BSN)']

monthly_rain_analysis['Rainfall_sum_shift_1']=monthly_rain_analysis['rainfall_sum'].shift(1)
monthly_rain_analysis['Rainfall_sum_shift_2']=monthly_rain_analysis['rainfall_sum'].shift(2)
monthly_rain_analysis['Rainfall_sum_shift_3']=monthly_rain_analysis['rainfall_sum'].shift(3)


month_names = {0: 'January', 1: 'February', 2: 'March', 3: 'April', 4: 'May', 5: 'June',
    6: 'July', 7: 'August', 8: 'September', 9: 'October', 10: 'November', 11: 'December',12: 'Extra 1',13: 'Extra 2',14: 'Extra 3'}
monthly_rain_analysis.index = monthly_rain_analysis.index.map(month_names)

#monthly_rain_analysis.to_csv("ALTU_ANALYSIS.csv")
