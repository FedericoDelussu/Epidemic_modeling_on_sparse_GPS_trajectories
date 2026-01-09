'''
This script launches an SIR model on the contact network of the complete users
'''

import sys
import pandas as pd 
import numpy as np
from Modules.utils import * 
from Modules.analysis import *

#Folder collecting the epidemic modeling outcomes
FOLD_emo = sys.argv[0]

#[0] Import the ground truth estimated contacts for the complete users within a specific Date-range
df_contacts, USERS_select, Date_range = import_complete_contacts_users_drange() 
N_users = len(USERS_select)
W = gen_contact_daily(df_contacts, USERS_select)

#[1] Launch the epidemic simulation

#[1.1] Epidemic parameters
#epid_pars: (beta, gamma) tuple
#beta: probability of infection given 1-minute contact
#gamma : probability of recovery over 1 day
#n_init : seed size of the SIR model
#N_iter : number of iterations of the epidemic simulation
groundtruth_pars {'beta': 0.0008457834699125536, 
                  'gamma': 0.2683760024015006, 
                  'n_init': 3}

epid_pars = (groundtruth_pars['beta'], groundtruth_pars['gamma'])
n_init = groundtruth_pars['n_init']
N_iter = 5000

#[1.2] Collect the simulation outcomes; daily time-series of Susceptible and Infected 
#and associated epidemic metrics
COLLECT_simulations_complete, COLLECT_epid_metrics_complete = iter_epid_simulation(W, 
                                                                                   Date_range,
                                                                                   USERS_select, 
                                                                                   epid_pars, 
                                                                                   n_init = n_init, 
                                                                                   N_iter = N_iter,
                                                                                   from_weekday = True, 
                                                                                   gamma_daily = True)

df_epid_stats = compute_epid_stats(COLLECT_epid_metrics_complete).reset_index()
df_epid_stats.to_csv(f'{FOLD_emo}/df_epid_stats.csv')
with open(f'{FOLD_emo}simulations.pkl', 'wb') as f:
    pickle.dump(COLLECT_simulations_complete, f)

