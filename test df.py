# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 18:00:57 2025

@author: ej_st
"""

import pandas as pd


#data = [[1, 2, 2, 4, 5] for _ in range(5)]
data = [
    [1, 1, 1, 1, 1],
    [1, 1, 2, 2, 1],
    [1, 1, 3, 3, 3],
    [1, 1, 3, 4, 4],
    [1, 1, 3, 4, 5]]
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
df['diff']=df['E'].diff()
correlation_rain_4MIC_to_RS2 = df['A'].corr()
#df_test_corr=df['A'].corr(df['B'])