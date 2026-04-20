import pandas as pd
import numpy as np
import os
import datetime as dt
from Modules import config
from Modules import analysis

#folder where the output of the analysis is stored
FOLD_output = config.FOLD_DTU

#[1] GET COMPLETE USERS from the hourly record indicator
print('Importing sequence indicator...')
path_seq = f'{FOLD_output}02_df_hourly_record_indicator.csv'
df_seq = pd.read_csv(path_seq).set_index(['user_id', 'weekstep_index'])
print(f'  sequence indicator loaded: {df_seq.shape}')

USERS_select = analysis.get_complete_users(df_seq)
print(f'  complete users selected: {len(USERS_select)}')

#[2] IMPORT TRAJECTORIES FOR COMPLETE USERS
#date range accounts for the study period plus a 2-day margin on each side
#to avoid undetected stops due to border effects in stop detection
Date_range = [config.Study_period[0] - dt.timedelta(days=2),
              config.Study_period[1] + dt.timedelta(days=2)]

print(f'Importing trajectories (date range: {Date_range[0].date()} -> {Date_range[1].date()})...')
path = f'{FOLD_output}01_trajectories_preprocessed/'
traj_complete = analysis.read_folder_files(path,
                                           Cols_select       = ['user_id', 'datetime', 'lat', 'lon'],
                                           parse_dates_list  = ['datetime'],
                                           FILES_select      = [f'{u}.csv' for u in USERS_select],
                                           Date_range        = Date_range,
                                           n_workers         = min(90, os.cpu_count() or 1))

traj_complete['datetime'] = pd.to_datetime(traj_complete['datetime'],
                                           utc=True).dt.tz_localize(None)
print(f'  trajectories loaded: {traj_complete.shape}')

#[3] SAVE
path_out = f'{FOLD_output}02a_traj_complete.pkl'
traj_complete.to_pickle(path_out)
print(f'  saved to {path_out}')
