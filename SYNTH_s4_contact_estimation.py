import glob

import pandas as pd
import numpy as np
import os
import datetime as dt
from multiprocessing import Pool, cpu_count
from Modules import config
from Modules import analysis

#folder where the output of the analysis is stored
FOLD_save = config.FOLD_SYNTH
PARALLEL = True

path_stays = f'{FOLD_save}03_stop_tables/'
path_contacts = f'{FOLD_save}04_contacts/'

#[0] import stops of the complete users
files = glob.glob(f'{path_stays}*.csv')
df_stops = pd.concat([pd.read_csv(f,
                                  parse_dates = ['start_time', 'end_time'],
                                  index_col =0).assign(user_id=f.split('/')[-1].replace('.csv', '')) for f in files])

#[2] estimate contacts
#import the hourly record indicator to run the stop detection only on the complete trajectories
df_hri = pd.read_csv(f'{FOLD_save}02_df_hourly_record_indicator.csv')
df_hri = df_hri.set_index(['user_id','weekstep_index']).fillna(0)
Dates = np.unique(pd.to_datetime(df_hri.columns).date)

def process_day(day):
    day_dt = pd.Timestamp(day)
    df_contacts = analysis.estimate_contacts(df_stops,
                                             ghr = 8,
                                             time_step = '1hour',
                                             Date_range = [day_dt, day_dt + dt.timedelta(days = 1)])
    df_contacts.to_csv(f'{path_contacts}{day_dt.strftime("%Y-%m-%d")}.csv')

if PARALLEL:
    with Pool(cpu_count()) as pool:
        pool.map(process_day, Dates)
else:
    for day in Dates:
        process_day(day)
