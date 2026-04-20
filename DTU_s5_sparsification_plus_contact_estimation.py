import glob
import pandas as pd
import numpy as np
import os
import datetime as dt
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from Modules import config
from Modules import analysis

#folder where the output of the analysis is stored
FOLD_output = config.FOLD_DTU

#SIMULATION SPECS
#levels of sparsity
Levels = config.Levels
#sparsity approaches
Sparsity_approaches = ['Random_shuffling', 'Random_uniform']

#Import sequence indicator
print('Importing sequence indicator...')
path_seq = f'{FOLD_output}02_df_hourly_record_indicator.csv'
df_seq = pd.read_csv(path_seq).set_index(['user_id','weekstep_index'])
print(f'  sequence indicator loaded: {df_seq.shape}')

USERS_select = analysis.get_complete_users(df_seq)
print(f'  complete users selected: {len(USERS_select)}')

#Import the pre-filtered trajectory data saved by DTU_s2a
print('Importing trajectories...')
path_traj = f'{FOLD_output}02a_traj_complete.pkl'
traj_complete = pd.read_pickle(path_traj)
print(f'  trajectories loaded: {traj_complete.shape}')

FOLD_save = f'{FOLD_output}05_sparsified_pipeline_outputs/'


def run_iter(n_iter):
    print(f'[START] iter={n_iter}')

    print(f'  [{n_iter}] generating sparsification masks...')
    DICT_masks = analysis.gen_sparsification_masks(df_seq, Levels)
    print(f'  [{n_iter}] masks ready')

    def run_level(level):
        for sparsity_approach in Sparsity_approaches:
            print(f'  [{n_iter}] {sparsity_approach} | level={level}')
            file_prefix = f'Iter{n_iter}_{sparsity_approach}_{level}'
            analysis.sparsification_pipeline(traj_complete,
                                             DICT_masks,
                                             sparsity_approach,
                                             level,
                                             FOLD_iter=FOLD_save,
                                             file_prefix=file_prefix)
            print(f'  [{n_iter}] {sparsity_approach} | level={level} done')

    #[2] repeat the pipeline (sparsification, stop-detection, contact estimation) for each level in parallel
    with ThreadPool(len(Levels)) as tp:
        tp.map(run_level, Levels)

    print(f'[DONE]  iter={n_iter}')

if __name__ == '__main__':
    N_CORES = min(50 // len(Levels), os.cpu_count() or 1)
    print(f'Launching 50 tasks on {N_CORES} cores ({N_CORES} processes x {len(Levels)} threads = {N_CORES*len(Levels)} total workers)')
    with Pool(processes=N_CORES) as pool:
        pool.map(run_iter, range(1, 51))
    print('All done')
