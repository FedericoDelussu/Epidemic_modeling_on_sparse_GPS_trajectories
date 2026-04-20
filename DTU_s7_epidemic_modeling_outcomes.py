import glob
import pickle

import pandas as pd
import numpy as np
import os
import datetime as dt
from multiprocessing import Pool
from Modules import config
from Modules import analysis

#folder where the output of the analysis is stored
FOLD_output = config.FOLD_DTU
#folder containing the corrected contacts
FOLD_contact_corrected = f'{FOLD_output}06_contacts_data-driven_ipw-weight/'
#folder where saving the epidemic modeling outcomes
FOLD_emo = f'{FOLD_output}07_epidemic_modeling_outcomes/'
os.makedirs(FOLD_emo, exist_ok=True)


#RUN FLAGS
RUN_GROUNDTRUTH                      = False
RUN_ORACLE_BIASED                    = True
RUN_COMPARE_ORACLE_CALIBRATION_CORRECTION = False

#IMPORTS
#[1] sequence indicator
path_seq = f'{FOLD_output}02_df_hourly_record_indicator.csv'
df_seq = pd.read_csv(path_seq).set_index(['user_id','weekstep_index'])
#get the list of complete users
USERS_select = analysis.get_complete_users(df_seq)

#[2] ground truth contacts
files = glob.glob(f'{FOLD_output}04_contacts/*.csv')
df_contacts_gt = pd.concat([pd.read_csv(f, index_col=0) for f in files])

#CONFIGURATION OF THE EPIDEMIC SIMULATION

#[1] study-period
_em_start = dt.datetime(2014, 2, 10)
_em_end   = dt.datetime(2014, 3, 7)
Study_period_em = [(_em_start + dt.timedelta(days=i)).date()
                   for i in range((_em_end - _em_start).days + 1)]

#[2] epidemic modeling parameters: (beta; gamma; seed size)
groundtruth_pars = config.groundtruth_pars

Iterations          = range(1, 51)
Levels              = config.Levels
Sparsity_approaches = ['Data_driven', 'Random_shuffling', 'Random_uniform']
N_iter   = 100


def _save_outputs(epid_curves, epid_metrics, prefix, suffix):
    with open(f'{FOLD_emo}{prefix}_{suffix}_simulations.pkl', 'wb') as f:
        pickle.dump(epid_curves, f)
    df_epid_stats = analysis.compute_epid_stats(epid_metrics).reset_index()
    df_epid_stats.to_csv(f'{FOLD_emo}{prefix}_{suffix}_df_epid_stats.csv')

def _save_calibration_info(best_params, best_value, prefix, contacts_type):
    pd.DataFrame([{**best_params, 'best_value': best_value}]).to_csv(
        f'{FOLD_emo}{prefix}_{contacts_type}_calibration_info.csv', index=False)


###########################################################
##### BLOCK 1 - GROUNDTRUTH EPIDEMIC SIMULATION ###########
###########################################################

if RUN_GROUNDTRUTH:
    epid_curves, epid_metrics = analysis.epidemic_modeling(df_contacts_gt,
                                                           USERS_select,
                                                           groundtruth_pars,
                                                           Study_period_em,
                                                           5000)
    with open(f'{FOLD_emo}groundtruth_simulations.pkl', 'wb') as f:
        pickle.dump(epid_curves, f)
    df_epid_stats = analysis.compute_epid_stats(epid_metrics).reset_index()
    df_epid_stats.to_csv(f'{FOLD_emo}groundtruth_df_epid_stats.csv')


#############################################################################################
##### BLOCK 2 - ORACLE ON SPARSE CONTACTS / DIFFERENT SPARSITY SCENARIOS ####################
#############################################################################################

def run_iter_oracle_biased(n_iter):
    print(f'[ORACLE_BIASED][START] iter={n_iter}')

    for sparsity_approach in Sparsity_approaches:
        for level in Levels:
            prefix = f'Iter{n_iter}_{sparsity_approach}_{level}'
            f_path = f'{FOLD_contact_corrected}Iter{n_iter}_{sparsity_approach}_{level}'
            df_contacts = pd.read_csv(f_path)

            if not os.path.exists(f'{FOLD_emo}simulations_{prefix}_sparse_contacts_oracle.pkl'):
                epid_curves, epid_metrics = analysis.epid_modeling(
                    df_contacts[['u1', 'u2', 'date_hour', 'n_minutes']],
                    USERS_select,
                    Study_period_em,
                    N_iter,
                    groundtruth_pars,
                    modeling_type='Oracle')
                _save_outputs(epid_curves, epid_metrics, prefix, 'sparse_contacts_oracle')

    print(f'[ORACLE_BIASED][DONE]  iter={n_iter}')


###################################################################################################
##### BLOCK 3 - ORACLE/CALIBRATION ON SPARSE/CORRECTED CONTACTS FOR DATA-DRIVEN SPARSIFICATION ####
###################################################################################################

#load Curve_ref (median infected from groundtruth ensemble) needed by calibration
with open(f'{FOLD_emo}groundtruth_simulations.pkl', 'rb') as f:
    _epid_curves_gt = pickle.load(f)
Curve_ref = pd.Series(np.median(_epid_curves_gt[:, :, 1], axis=0))

#parameter search bounds for calibration
GRID_stats = config.GRID_stats

#trials of the calibration optimization problem
n_trials = 100

def run_iter_compare(n_iter):
    print(f'[COMPARE][START] iter={n_iter}')

    for level in Levels:
        prefix = f'Iter{n_iter}_Data_driven_{level}'
        f_path = f'{FOLD_contact_corrected}Iter{n_iter}_Data_driven_{level}'
        df_contacts = pd.read_csv(f_path)

        #--- Sparse contacts, Oracle ---
        if not os.path.exists(f'{FOLD_emo}simulations_{prefix}_sparse_contacts_oracle.pkl'):
            epid_curves, epid_metrics = analysis.epid_modeling(
                df_contacts[['u1', 'u2', 'date_hour', 'n_minutes']],
                USERS_select,
                Study_period_em,
                N_iter,
                groundtruth_pars,
                modeling_type='Oracle')
            _save_outputs(epid_curves, epid_metrics, prefix, 'sparse_contacts_oracle')

        #--- Sparse contacts, Calibration ---
        epid_curves, epid_metrics, best_params, best_value = analysis.epid_modeling(
            df_contacts[['u1', 'u2', 'date_hour', 'n_minutes']],
            USERS_select,
            Study_period_em,
            N_iter,
            groundtruth_pars,
            modeling_type='Calibration',
            Curve_ref=Curve_ref,
            GRID_stats=GRID_stats,
            n_trials=n_trials)
        _save_outputs(epid_curves, epid_metrics, prefix, 'sparse_contacts_calibration')
        _save_calibration_info(best_params, best_value, prefix, 'sparse_contacts')

        #reassign n_minutes as n_minutes * weight for corrected contacts
        df_corrected = df_contacts.copy()
        df_corrected['n_minutes'] = df_corrected['n_minutes'] * df_corrected['weight']

        #--- Corrected contacts, Oracle ---
        epid_curves, epid_metrics = analysis.epid_modeling(
            df_corrected[['u1', 'u2', 'date_hour', 'n_minutes']],
            USERS_select,
            Study_period_em,
            N_iter,
            groundtruth_pars,
            modeling_type='Oracle')
        _save_outputs(epid_curves, epid_metrics, prefix, 'corrected_contacts_oracle')

        #--- Corrected contacts, Calibration ---
        epid_curves, epid_metrics, best_params, best_value = analysis.epid_modeling(
            df_corrected[['u1', 'u2', 'date_hour', 'n_minutes']],
            USERS_select,
            Study_period_em,
            N_iter,
            groundtruth_pars,
            modeling_type='Calibration',
            Curve_ref=Curve_ref,
            GRID_stats=GRID_stats,
            n_trials=n_trials)
        _save_outputs(epid_curves, epid_metrics, prefix, 'corrected_contacts_calibration')
        _save_calibration_info(best_params, best_value, prefix, 'corrected_contacts')

    print(f'[COMPARE][DONE]  iter={n_iter}')


if __name__ == '__main__':
    N_CORES = min(50, os.cpu_count() or 1)

    if RUN_ORACLE_BIASED:
        print(f'[BLOCK 2] Launching {len(list(Iterations))} tasks on {N_CORES} cores')
        with Pool(processes=N_CORES) as pool:
            pool.map(run_iter_oracle_biased, Iterations)
        print('[BLOCK 2] Done')

    if RUN_COMPARE_ORACLE_CALIBRATION_CORRECTION:
        print(f'[BLOCK 3] Launching {len(list(Iterations))} tasks on {N_CORES} cores')
        with Pool(processes=N_CORES) as pool:
            pool.map(run_iter_compare, Iterations)
        print('[BLOCK 3] Done')
