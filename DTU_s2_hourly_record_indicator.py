import pandas as pd
import numpy as np
import os
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from Modules import config
from Modules import analysis

#folder where the output of the analysis is stored
FOLD_save = config.FOLD_DTU

path = f'{FOLD_save}01_trajectories_preprocessed/'
FILES_select = os.listdir(path)

#Study-period for the selection of the complete trajectories for doing epidemic modeling
#from 2014 Feb 2 to March 7 (included)
Study_period = [dt.datetime(2014,2,8), dt.datetime(2014,3,7)]
#Time-range for the selection of the sparse trajectories for sparsification
Time_range = [dt.datetime(2014,2,1), dt.datetime(2015,2,1)]

#[0] import location trajectories of all the users in parallel chunks
DICT_dtypes = {'user_id':'int32',
               'datetime':'string'}

CHUNK_SIZE = 100
chunks = [FILES_select[i:i+CHUNK_SIZE] for i in range(0, len(FILES_select), CHUNK_SIZE)]

def process_chunk(chunk):

    df = analysis.read_folder_files(path,
                                    Cols_select = ['user_id', 'datetime'],
                                    dtype = DICT_dtypes,
                                    FILES_select = chunk)

    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)

    return analysis.collect_sequences(df,
                                      Study_period,
                                      Time_range)

sequences_parts = []
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_chunk, c): i for i, c in enumerate(chunks)}
    for i, future in enumerate(as_completed(futures), 1):
        print(f'[{i}/{len(chunks)}] chunk done')
        sequences_parts.append(future.result())

df_sequences = pd.concat(sequences_parts, axis=0)
df_sequences.to_csv(f'{FOLD_save}02_df_hourly_record_indicator.csv')



