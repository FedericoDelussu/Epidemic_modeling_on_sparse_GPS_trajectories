import pandas as pd 
import sys
from Modules.utils import * 
from Modules.analysis import *

iter_sparse = sys.argv[1]
print(iter_sparse)

#Folder in which all the sparsification iterations are realized
FOLD_sparse = DICT_paths['TMP'] + 'f01_013_D1_iter_sparsified_mask_loc_stops_contacts/'
#Sparse_scenarios = ['Data_driven', 'Random_uniform', 'Random_keepdurations']
str_1wgcu = '1wgcu'
Sparse_scenarios = ['Data_driven','Random_uniform', 'Random_keepdurations']
Sparse_scenarios = [f'{s}_{str_1wgcu}' for s in Sparse_scenarios]
print(Sparse_scenarios)

#parameters for lachesis stop-detection
par_lach = (10,360,50)
#geohash resolution for contact estimation
ghr = 8

#epidemic-window search parameters - days of the window
SW_width_days = 28
#tolerance on percentage of missing hours 
epsilon = 0.05

df_contacts, USERS_select, Date_range = import_complete_contacts_users_drange() 
#extend date range by 2days for each edge to avoid border effects in stop detection
Date_range = pd.date_range(start = Date_range.min() - pd.Timedelta(days=2), 
                           end = Date_range.max() + pd.Timedelta(days=2), 
                           freq='D')

#[0.3] Define the sparsity levels
Levels = gen_sparsity_ranges() 

#Sequences for DATA-DRIVEN Sparsification
Cols_info = ['id','sparse', 'weekstep_index']

FILTER_SEQUENCES = True

df_seq = import_gap_seq()
df_seq_complete = get_complete_sequences(df_seq, USERS_select)
metric = 'perc-missing-hours'
Seq_sparsity_complete = compute_sparse_metric(df_seq_complete, 
                                              metric, 
                                              SW_width_days)
df_seq_complete = df_seq_complete.loc[Seq_sparsity_complete.index]

if FILTER_SEQUENCES: 
    df_seq = filter_sequences(df_seq)
    df_seq = df_seq.reset_index()


metric = 'perc-missing-hours'
Seq_sparsity_complete = compute_sparse_metric(df_seq_complete, 
                                              metric, 
                                              SW_width_days)

df_seq_complete = df_seq_complete.loc[Seq_sparsity_complete.index]

df_seq = df_seq.drop(Cols_info, axis = 1)
Seq_series = compute_sparse_metric(df_seq, 
                                   metric,
                                   SW_width_days)

#Sequences for RANDOM-UNIFORM Sparsification
df_seq_uniform = df_seq.copy()*0 + 1
def set_random_zeros(row):
    p = np.random.uniform(0,0.6)
    n_zeros = int(len(row) * p)
    zero_indices = np.random.choice(row.index, size=n_zeros, replace=False)
    row[zero_indices] = 0
    return row

    
df_seq_uniform = df_seq_uniform.apply(lambda row: set_random_zeros(row), axis=1)
Seq_sparsity_uniform = compute_sparse_metric(df_seq_uniform, 
                                             metric,
                                             SW_width_days)

#Location data for the complete users
df_loc_complete = import_loc_1min_data(USERS_select, 
                                       Date_range = Date_range)


#for iter_sparse in range(N_iter_sparse):  

print(f'Sparsification Iter {iter_sparse}')

FOLD_iter = f'{FOLD_sparse}Iter_sparse_{iter_sparse}/'
gen_fold(FOLD_iter)

#Data-driven sparsification masks
DICT_mask = {level: gen_mask(level, 
                             Seq_series, 
                             df_seq, 
                             df_seq_complete, 
                             metric, 
                             SW_width_days) for level in Levels}

#Random-uniform sparsification masks   
DICT_mask_uniform = {level: gen_mask(level, 
                                     Seq_sparsity_uniform, 
                                     df_seq_uniform, 
                                     df_seq_complete, 
                                     metric, 
                                     SW_width_days) for level in Levels}

DICT_mask_uniform_keep_durations = {l : m.apply(lambda row: pd.Series(shuffle_gaps_keep_durations(row.values)), axis=1) 
                                    for l,m in DICT_mask.items()}
for l in Levels:
    DICT_mask_uniform_keep_durations[l].columns = DICT_mask[l].columns

DICT_scenario_masks = {f'Data_driven_{str_1wgcu}': DICT_mask, 
                       f'Random_uniform_{str_1wgcu}': DICT_mask_uniform, 
                       f'Random_keepdurations_{str_1wgcu}': DICT_mask_uniform_keep_durations}


def process_scenario(Sparse_scenario):
    FOLD_iter_ss = f'{FOLD_iter}{Sparse_scenario}/'
    gen_fold(FOLD_iter_ss)

    for level in Levels:
        
        print(f'\t\t{Sparse_scenario} level {level}')
        FOLD_iter_ss_l = f'{FOLD_iter_ss}{level}/'
        gen_fold(FOLD_iter_ss_l)

        mask = DICT_scenario_masks[Sparse_scenario][level]
        mask.to_csv(FOLD_iter_ss_l + 'df_mask.csv')

        df_record_select = from_mask_to_record_indicator(mask, 'date_hour')
        df_lcf = pd.merge(df_record_select, df_loc_complete, on=['id', 'datetime'], how='left').dropna(subset=['lat', 'lng'])
        df_lcf = df_lcf.groupby('id').filter(lambda x: len(x) > 1)

        df_lcf_stops = detect_stops(df_lcf, par_lach)
        df_lcf_stops.to_csv(FOLD_iter_ss_l + 'df_stops.csv')

        df_cwithin_hour, df_cmargin_hour = contact_estimate_daily_partition(df_lcf_stops, ghr)
        df_cmargin_hour.to_csv(FOLD_iter_ss_l + 'df_contact_marginal.csv')
        df_cwithin_hour.to_csv(FOLD_iter_ss_l + 'df_contact_within.csv')

from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    executor.map(process_scenario, Sparse_scenarios)
