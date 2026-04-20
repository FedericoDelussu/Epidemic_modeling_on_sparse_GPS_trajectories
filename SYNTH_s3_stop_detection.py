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

DICT_dtypes = {'user_id':'int32',
               'datetime':'string',
               'lat': 'float32',
               'lon': 'float32'}

#[1] GET THE COMPLETE USERS

#import the hourly record indicator to run the stop detection only on the complete trajectories
df_hri = pd.read_csv(f'{FOLD_save}02_df_hourly_record_indicator.csv')
df_hri = df_hri.set_index(['user_id','weekstep_index'])

#consider the time-range (weekstep_index ==0)
df_hri_tr = df_hri[df_hri.index.get_level_values('weekstep_index') == 0]

#condition for completeness over the time-range
bool_complete = df_hri_tr.sum(axis=1)/df_hri_tr.shape[1] >=0.95
USERS_complete = df_hri_tr.index.get_level_values('user_id')[bool_complete]
USERS_complete = np.sort(list(USERS_complete))

#[2] ESTIMATE THE STOPS FROM THE COMPLETE USERS

#path of preprocessed trajectories
path_trajs = f'{FOLD_save}01_trajectories_preprocessed/'

#path of stop tables
path_stays = f'{FOLD_save}03_stop_tables/'

def process_user(u):
    print(u)
    traj_u = pd.read_csv(path_trajs + f'{u}.csv', usecols = ['user_id','datetime','lat','lon'], dtype = DICT_dtypes)
    stays_u = analysis.lachesis(traj_u)
    stays_u.to_csv(path_stays + f'{u}.csv')

if PARALLEL:
    with Pool(cpu_count()) as pool:
        pool.map(process_user, USERS_complete)
else:
    for u in USERS_complete:
        process_user(u)
