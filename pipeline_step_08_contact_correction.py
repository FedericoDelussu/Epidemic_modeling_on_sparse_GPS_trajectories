'''
launch oracle-biased modeling on 2 random missingness configurations: 
- Random_uniform_1wgcu
- Random_keepdurations_1wgcu
'''

import pandas as pd 
import numpy as np
import sys
from Modules.utils import * 
from Modules.analysis import *

INSTR = "provide as" 
#INSTR += "\n 1st input: either 'sparse' or 'corrected' contacts"
#INSTR += "\n 2nd input: either 'none' or 'ipw_global'"
INSTR += '\n 1st input: Sparsification Iteration from 0 to 50'
INSTR += '\n 2nd input: Temporal bucket (date, hourofday_weekday)'

print(INSTR)

#INPUT_CONTACT = sys.argv[1]
#INPUT_CONTACT_CORRECTION = sys.argv[2]
INPUT_SPARSE_ITER = sys.argv[1]
bucket = sys.argv[2]

#list of missingness-models 
List_ss = ['Data_driven_1wgcu']
#list of sparsification levels
Levels = gen_sparsity_ranges()
#number of sparsification iterations
Ranges_iter = [INPUT_SPARSE_ITER]
#list of all sparse scenarios combinations
List_Sparse_Scenarios =  list(itertools.product(List_ss, Levels, Ranges_iter))

#Get users and date-range
df_contacts, USERS_select, Date_range = import_complete_contacts_users_drange()

for k in List_Sparse_Scenarios:
    
    print(k)

    #[1] Compute the weights for the contacts at different resolutions
    C_k, R_k = import_contacts_coverage(k, Date_range)
    Contact_corrections = ['ipw_global', 'ipw_daily', 'ipw_hourofday_weekday']
    Contact_corrections += [f'{c}_aligned' for c in Contact_corrections]
    _ = compute_coefs(C_k, R_k, Contact_corrections) 
    FOLD_save = '/home/fedde/work/Project_Penn/TMP/f01_015_D1_contacts_with_weights/'
    C_k.to_csv(FOLD_save + '_'.join([str(i) for i in k]) + '.csv')

    #[2] Define the targets for groundtruth contact recovery 
    Contacts_complete = import_contacts_complete(return_contact_dataframe = True)
    Contacts_complete = Contacts_complete.rename(columns = {'n_minutes': 'n_minutes_gt'})
    C_k = pd.merge(Contacts_complete, C_k, 
                   on = ['u1','u2', 'date_hour'], 
                   how = 'left')
    C_k['hour'] = pd.to_datetime(C_k['date_hour']).dt.hour
    C_k['date'] = pd.to_datetime(C_k['date_hour']).dt.date
    C_k['hourofday_weekday'] = gen_hw_col(C_k['date_hour'].values)
    C_k["n_minutes"] = C_k["n_minutes"].fillna(0)
    
    #[1] bucket-pair level weights and targets of contact estimation
    BCT_det = compare_ipw_gt_weights(C_k, bucket, Date_range, k)
    PATH_save = '/home/fedde/work/Project_Penn/TMP/f01_016_D1_contacts_weights_analysis/'
    k_str = '_'.join([str(i) for i in k])
    BCT_det.to_csv(f'{PATH_save}/{bucket}/{k_str}.csv')
