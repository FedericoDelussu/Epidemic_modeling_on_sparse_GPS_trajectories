import os
import pickle
import itertools
import datetime as dt
import pandas as pd
import numpy as np
from . import config

#Folder where the results of the pipeline are saved
FOLD = config.FOLD_DTU
FOLD_data = f'{FOLD}08_data_for_figures'

class KIT_conversion_dates:
    """Utilities to handle date-range arrays"""

    #hourofday-weekperiod coverage
    def get_hourofday_weekperiod(v):
        '''
        v : date-range series (will be converted to pandas datetime)
        gen hourofday_weekperiod column
        '''
        v = pd.to_datetime(v)
        _wdays = [0,1,2,3,4,5]
        _wperiod = np.array(['weekend','weekday'])
        
        return [f"{t}_{_wperiod[w*1]}" for t,w in zip(v.hour, v.weekday.isin(_wdays))] 

    def split_df_hourofday_weekperiod(df, 
                                      column = 'hourofday_weekperiod'):
        '''
        split the column in 2 cols
        '''
        #split into two columns
        df[['hourofday', 'weekperiod']] = df['hourofday_weekperiod'].str.split('_', expand=True)
        #cast hourofday to integer
        df['hourofday'] = df['hourofday'].astype(int)
        
    def convert_daterange_from_weekday(Date_range):
        '''
        given data-range let the first entry start from a weekday
        '''
        idx = 0
        while idx < len(Date_range) and Date_range[idx].weekday() >= 5:  # Skip weekends
            idx += 1
        return Date_range[(idx):]

##############################################
##### COUNT OF ACTIVE USERS AND CONTACTS #####
##############################################

def load_sparse_contacts(iter, sparsity, level, fold=FOLD):
    path = f'{fold}05_sparsified_pipeline_outputs/Iter{iter}_{sparsity}_{level}_df_contacts.csv'
    return pd.read_csv(path)

def load_sparse_mask(iter, sparsity, level, fold=FOLD):
    path = f'{fold}05_sparsified_pipeline_outputs/Iter{iter}_{sparsity}_{level}_mask.csv'
    return pd.read_csv(path, index_col=0)

def compute_hourly_counts(iter, sparsity, level):
    '''
    compute hourly counts of active users and contacts for a given (iter, sparsity, level)
    '''
    df_contacts = load_sparse_contacts(iter, sparsity, level)
    df_mask     = load_sparse_mask(iter, sparsity, level)

    hourly_contact_count = df_contacts[['u1','u2','date_hour']].drop_duplicates().groupby('date_hour').size()
    hourly_active_users  = df_mask.sum(axis=0)

    return hourly_active_users, hourly_contact_count

def merge_hourly_counts(df_active_users,
                                           df_contact_counts):
    '''merge active users and number of contacts into a single long-format dataframe'''

    df_active_users = (
        df_active_users.reset_index()
                       .melt(
                           id_vars=["iter", "sparsity", "level"],
                           var_name="datehour",
                           value_name="count_users"
                       )
    )

    df_contact_counts = (
        df_contact_counts.reset_index()
                         .melt(
                             id_vars=["iter", "sparsity", "level"],
                             var_name="datehour",
                             value_name="count_contacts"
                         )
    )

    df_merged = (
        df_active_users.merge(
            df_contact_counts,
            on=["iter", "sparsity", "level", "datehour"],
            how="inner"
        )
    )

    return df_merged

def save_hourly_counts(filename, Iters = range(50)):
    list_active_users  = []
    list_contact_counts = []
    for s in config.List_sparsity:
        print(s)
        for l in config.Levels:
            for ni in Iters:
                df_au, df_nc = compute_hourly_counts(ni, s, l)

                for c, cname in zip([ni, s, l], ['iter', 'sparsity', 'level']):
                    df_au[cname] = c
                    df_nc[cname] = c

                list_active_users.append(df_au)
                list_contact_counts.append(df_nc)

    df_active_users   = pd.concat(list_active_users,   axis=1).T
    df_contact_counts = pd.concat(list_contact_counts, axis=1).T

    df_active_users   = df_active_users.set_index(['iter', 'sparsity', 'level'])
    df_contact_counts = df_contact_counts.set_index(['iter', 'sparsity', 'level'])

    df_merged = merge_hourly_counts(df_active_users, df_contact_counts)

    df_merged['datehour'] = pd.to_datetime(df_merged['datehour'])
    #accounting for 1hour transition
    df_merged['datehour'] += dt.timedelta(hours=1)
    df_merged['hourofday'] = df_merged['datehour'].dt.hour

    df_merged['hourofweek'] = pd.to_datetime(df_merged['datehour']).dt.weekday*24 + df_merged['hourofday']
    df_merged['hourofday_weekperiod'] = KIT_conversion_dates.get_hourofday_weekperiod(df_merged['datehour'].values)
    KIT_conversion_dates.split_df_hourofday_weekperiod(df_merged)

    df_merged_save = df_merged[pd.to_datetime(df_merged['datehour']) >= dt.datetime(2014, 2, 10)]

    df_merged_save = df_merged_save.rename(columns={'level': 'sparsity_level',
                                                    'iter':  'iter_sparsity'})

    df_merged_save['sparsity'] = df_merged_save['sparsity'].replace(config.DICT_rename_ss)
    df_merged_save.to_csv(f'{FOLD_data}/{filename}')

##################################
##### EMO METRICS AND CURVES #####
##################################

def load_groundtruth_emo():
    FOLD_emo = f'{FOLD}07_epidemic_modeling_outcomes/'
    path_epid_stats  = f'{FOLD_emo}groundtruth_df_epid_stats.csv'
    path_simulations = f'{FOLD_emo}groundtruth_simulations.pkl'

    df_epid_stats = pd.read_csv(path_epid_stats, index_col=0)
    with open(path_simulations, 'rb') as f:
        COLLECT_simulations = pickle.load(f)

    return df_epid_stats, COLLECT_simulations

def load_sparse_emo(scenario,
                   emv_name,
                   result_import='epid_stats'):
    '''
    scenario : (iter, sparsity, level)
    emv_name : {'corrected','sparse'}_contacts_{'oracle','calibration'}
    possible result_import:
        'epid_stats'
        'simulations'
        'info'
    '''
    FOLD_emo = f'{FOLD}07_epidemic_modeling_outcomes/'
    iter, sparsity, level = scenario
    prefix = f'Iter{iter}_{sparsity}_{level}'

    _dict_paths = {'epid_stats'  : f'{FOLD_emo}{prefix}_{emv_name}_df_epid_stats.csv',
                   'simulations' : f'{FOLD_emo}{prefix}_{emv_name}_simulations.pkl',
                   'info'        : f'{FOLD_emo}{prefix}_{emv_name}_calibration_info.csv'}

    file_path = _dict_paths[result_import]

    if '.csv' in file_path:
        return pd.read_csv(file_path, index_col=0)

    if '.pkl' in file_path:
        with open(file_path, 'rb') as f:
            return pickle.load(f)

def compute_epid_metrics(ts_SI):
    '''
    epidemic metric computation on a single experimental realization
    return ['peak_day', 'peak_size', 'final_size', 'epid_duration', 'day_final_case']
    '''
    #time-series of susceptibles
    S = ts_SI[:,0]
    #time-series of infected
    I = ts_SI[:,1]
    
    #infected peak
    peak_size = np.max(I)
    #first time in which infected time-series reached its peak
    peak_day = np.where(I==peak_size)[0][0] + 1
    
    #duration of the epidemic - computed as the day in which the I became constantly 0
    epid_duration = len(I)
    cumsum_condition = np.cumsum(I[::-1]) > 0
    indices = np.argwhere(cumsum_condition) 

    #time of final case
    #corresponding to the day since when the susceptible curve became a plateau
    #np.argmin returns the first index of occurring minimum in case of multiple occurrences
    day_final_case = np.argmin(S) + 1 
    
    if indices.size > 0:  # Check if indices array is not empty
        epid_duration -= (indices[0, 0] + 1)
    
    #final size of the epidemic - number of individuals which became infected
    final_size = I[0] + S[0] - S[-1]
    
    Epid_metrics = [peak_day, peak_size, final_size, epid_duration, day_final_case]

    return Epid_metrics

def collect_epid_metrics(DICT_EMO_metrics,
                   _SS,
                   Levels,
                   N_users,
                   Emo_metrics_names,
                   List_N_si=['ALL'],
                   normalize_size=True,
                   EMVs=None):
    '''
    collect all simulation metrics into a dataframe
    '''
    if EMVs is None:
        EMVs = list({k[3] for k in DICT_EMO_metrics if k != 'groundtruth'})

    df_metrics = []

    for s in _SS:
        for l in Levels:
            for ni in List_N_si:
                for emv in EMVs:
                    _df_sl = pd.DataFrame(DICT_EMO_metrics[(ni, s, l, emv)],
                                          columns=Emo_metrics_names)
                    _df_sl['iter']     = ni
                    _df_sl['sparsity'] = s
                    _df_sl['level']    = str(l)
                    _df_sl['emv']      = emv
                    df_metrics.append(_df_sl)

    df_metric_complete = pd.DataFrame(DICT_EMO_metrics['groundtruth'],
                                      columns=Emo_metrics_names)

    df_metric_complete['iter']     = 'groundtruth'
    df_metric_complete['sparsity'] = 'groundtruth'
    df_metric_complete['level']    = 'groundtruth'
    df_metric_complete['emv']      = 'groundtruth'

    df_metrics.append(df_metric_complete)
    df_metrics = pd.concat(df_metrics, axis=0)

    if normalize_size:
        df_metrics['peak_size']  /= N_users
        df_metrics['final_size'] /= N_users

    return df_metrics


def load_all_emos(List_Sparse_Scenarios, EMVs, import_complete=True):

    DICT_EMO = {}
    for emv in EMVs:
        for scenario in List_Sparse_Scenarios:
            epid_stats = load_sparse_emo(scenario, emv, result_import='epid_stats')
            epid_sims  = load_sparse_emo(scenario, emv, result_import='simulations')
            DICT_EMO[(*scenario, emv)] = (epid_stats, epid_sims)

    if import_complete:
        DICT_EMO['groundtruth'] = load_groundtruth_emo()

    return DICT_EMO

def build_emo_dict(List_Sparse_Scenarios, EMVs, import_complete=True):

    DICT_EMO = load_all_emos(List_Sparse_Scenarios,
                           EMVs,
                           import_complete=import_complete)

    DICT_EMO_metrics = {k: np.array([compute_epid_metrics(e) for e in EMO[1]]) for k, EMO in DICT_EMO.items()}

    #aggregate sample of metrics over all sparsification iterations
    df_ss = pd.DataFrame(List_Sparse_Scenarios, columns=['iter', 'sparsity', 'level'])
    _iters    = df_ss['iter'].unique()
    _sparsity = df_ss['sparsity'].unique()
    _levels   = df_ss['level'].unique()

    DICT_EMO_metrics.update({(s, l, 'ALL', emv): np.concatenate([DICT_EMO_metrics[(ni, s, l, emv)] for ni in _iters])
                             for s in _sparsity
                             for l in _levels
                             for emv in EMVs})

    return DICT_EMO, DICT_EMO_metrics

def curves_to_dataframe(k_curves):
    '''
    converts array of 100 epidemic curves to a df 
    column structure: [S,I,iter_SIR]
    '''
    
    df_k = []
    
    for i, df in enumerate(k_curves):
        df = pd.DataFrame(df, columns = ['S','I'])
        df['iter_SIR'] = i
        df_k.append(df)
        
    df_k = pd.concat(df_k, 
                     axis=0)
    
    return df_k

def scenario_curves_to_dataframe(DICT_EMO, k):
    '''
    select a sparsity scenario and generates a df of epidemic curves
    '''
    k_curves = DICT_EMO[k][1]
    df_k = curves_to_dataframe(k_curves)

    if k == 'groundtruth':
        df_k['iter_sparsity']  = 'groundtruth'
        df_k['sparsity']       = 'groundtruth'
        df_k['sparsity_level'] = 'groundtruth'
        df_k['modeling_type']  = 'groundtruth'
    else:
        iter, sparsity, level, emv = k
        df_k['iter_sparsity']  = iter
        df_k['sparsity']       = sparsity
        df_k['sparsity_level'] = str(level)
        df_k['modeling_type']  = emv

    return df_k

def save_emo_sparsification(filename_curves, filename_metrics, Iters=range(50)):

    LSS = list(itertools.product(Iters, config.List_sparsity, config.Levels))

    DICT_EMO, DICT_EMO_metrics = build_emo_dict(LSS,
                                                EMVs=['sparse_contacts_oracle'],
                                                import_complete=True)

    #[1] collect the epidemic curves
    df_EMO = pd.concat([scenario_curves_to_dataframe(DICT_EMO, k) for k in DICT_EMO], axis=0)
    df_EMO.to_csv(f'{FOLD_data}/{filename_curves}')

    #get the number of users for doing normalization
    df_complete = scenario_curves_to_dataframe(DICT_EMO, 'groundtruth')
    N_users = int(df_complete[df_complete['iter_SIR'] == 0].iloc[0][['S', 'I']].sum())

    #[2] collect the metrics and compute aggregated statistics
    df_metrics = collect_epid_metrics(DICT_EMO_metrics,
                                      config.List_sparsity,
                                      config.Levels,
                                      N_users,
                                      config.Emo_metrics_names,
                                      List_N_si=Iters,
                                      normalize_size=False)

    df_metrics.loc[df_metrics['emv'] == 'groundtruth', 'sparsity'] = 'groundtruth'
    df_metrics.loc[df_metrics['emv'] == 'groundtruth', 'level']    = 'groundtruth'
    df_metrics.loc[df_metrics['emv'] == 'groundtruth', 'iter']     = 'groundtruth'

    df_metrics['sparsity'] = df_metrics['sparsity'].replace(config.DICT_rename_ss)

    df_metrics.to_csv(f'{FOLD_data}/{filename_metrics}')

def save_emo_debiasing(filename_curves, filename_metrics, Iters=range(50)):

    LSS = list(itertools.product(Iters, ['Data_driven'], config.Levels))

    EMVs = ['sparse_contacts_calibration',
            'sparse_contacts_oracle',
            'corrected_contacts_calibration',
            'corrected_contacts_oracle']

    DICT_EMO, DICT_EMO_metrics = build_emo_dict(LSS,
                                                EMVs=EMVs,
                                                import_complete=True)

    #[1] collect the epidemic curves
    df_EMO = pd.concat([scenario_curves_to_dataframe(DICT_EMO, k) for k in DICT_EMO], axis=0)
    df_EMO.to_csv(f'{FOLD_data}/{filename_curves}')

    df_complete = scenario_curves_to_dataframe(DICT_EMO, 'groundtruth')
    N_users = int(df_complete[df_complete['iter_SIR'] == 0].iloc[0][['S', 'I']].sum())

    #[2] collect the metrics and compute aggregated statistics
    df_metrics = collect_epid_metrics(DICT_EMO_metrics,
                                      ['Data_driven'],
                                      config.Levels,
                                      N_users,
                                      config.Emo_metrics_names,
                                      List_N_si=Iters,
                                      normalize_size=False)

    df_metrics.loc[df_metrics['emv'] == 'groundtruth', 'sparsity'] = 'groundtruth'
    df_metrics.loc[df_metrics['emv'] == 'groundtruth', 'level']    = 'groundtruth'
    df_metrics.loc[df_metrics['emv'] == 'groundtruth', 'iter']     = 'groundtruth'

    df_metrics['sparsity'] = df_metrics['sparsity'].replace(config.DICT_rename_ss)

    df_metrics.to_csv(f'{FOLD_data}/{filename_metrics}')


def load_calibration_info(Iters = range(1,51), 
                          fold=FOLD):
    '''
    Read all calibration_info.csv files in the 07_epidemic_modeling_outcomes folder.
    Iterates over config.List_sparsity, config.Levels, Iters, and contact types.
    Adds columns: iter, sparsity, level, contacts_type.
    Returns a concatenated DataFrame.
    '''
    folder = f'{fold}07_epidemic_modeling_outcomes/'
    contact_types = ['sparse', 'corrected']
    records = []
    for sparsity in ['Data_driven']:
        for level in config.Levels:
            for ni in Iters:
                for contacts_type in contact_types:
                    fname = f'Iter{ni}_{sparsity}_{level}_{contacts_type}_contacts_calibration_info.csv'
                    df = pd.read_csv(os.path.join(folder, fname))
                    df['iter']          = ni
                    df['sparsity']      = sparsity
                    df['level']         = str(level)
                    df['contacts_type'] = contacts_type
                    records.append(df)

    df_calibration_outcomes = pd.concat(records, axis=0, ignore_index=True)
    df_calibration_outcomes['beta']*=1e3
    
    return df_calibration_outcomes







