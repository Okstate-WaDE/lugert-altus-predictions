current_month = 1
current_year = 2024
month=current_month

import pandas as pd
import datetime
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
    
#combined_df.to_csv("ALTU_ALL.csv")
