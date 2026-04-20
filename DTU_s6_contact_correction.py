import glob
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from Modules import config
from Modules import analysis

#folder where the output of the analysis is stored
FOLD_output = config.FOLD_DTU
#folder containining the pipeline outputs after sparsification
FOLD_sparsified = f'{FOLD_output}05_sparsified_pipeline_outputs/'
#folder containing the corrected contacts
FOLD_contact_corrected = f'{FOLD_output}06_contacts_data-driven_ipw-weight/'

Iterations = range(1, 51)
Levels = config.Levels
Sparsity_approaches = ['Random_shuffling', 'Random_uniform']

#Import sequence indicator 
path_seq = f'{FOLD_output}02_df_hourly_record_indicator.csv'
#collect the sequences for the complete users
df_seq = pd.read_csv(path_seq).set_index(['user_id','weekstep_index'])
df_seq_w0 = df_seq[df_seq.index.get_level_values("weekstep_index") == 0].droplevel('weekstep_index')
df_seq_complete = df_seq_w0[df_seq_w0.mean(axis=1) >= 0.95]

#Import groundtruth contacts
files = glob.glob(f'{FOLD_output}04_contacts/*.csv')
df_contacts_gt = pd.concat([pd.read_csv(f, index_col = 0) for f in files])
df_contacts_gt = df_contacts_gt.rename(columns = {'n_minutes': 'n_minutes_gt'})

def run_iter(n_iter):
    print(f'[START] iter={n_iter}')

    for sparsity_approach in Sparsity_approaches:
        for level in Levels:

            file_prefix = f'{FOLD_sparsified}Iter{n_iter}_{sparsity_approach}_{level}'

            #import contacts
            df_contacts = pd.read_csv(f'{file_prefix}_df_contacts.csv')

            #import mask (hourly record indicator of the sparsified trajectories)
            df_mask = pd.read_csv(f'{file_prefix}_mask.csv', index_col=0)
            df_seq_complete_level = df_seq_complete.loc[df_mask.index]

            #then multiply the mask by the hourly record indicator of the complete users
            df_hri_sparsified = df_mask*df_seq_complete_level
            df_contacts, coefs = analysis.compute_contact_correction_weights(df_contacts,
                                                                             df_hri_sparsified)

            #join ground truth n_minutes for reference
            #entries in gt not covered by df_contacts are included with n_minutes = 0
            df_contacts = df_contacts.merge(
                df_contacts_gt[['u1', 'u2', 'date_hour', 'n_minutes_gt']],
                on=['u1', 'u2', 'date_hour'],
                how='right')

            df_contacts['n_minutes'] = df_contacts['n_minutes'].fillna(0)

            #save corrected contacts
            df_contacts.to_csv(f'{FOLD_contact_corrected}Iter{n_iter}_{sparsity_approach}_{level}', index=False)

    print(f'[DONE] iter={n_iter}')


if __name__ == '__main__':
    N_CORES = min(90, os.cpu_count() or 1)
    print(f'Launching {len(list(Iterations))} tasks on {N_CORES} cores')
    with Pool(processes=N_CORES) as pool:
        pool.map(run_iter, Iterations)
    print('All done')
