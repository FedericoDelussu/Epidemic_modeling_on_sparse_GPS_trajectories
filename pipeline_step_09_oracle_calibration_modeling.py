import pandas as pd 
import numpy as np
import sys
from Modules.utils import * 
from Modules.analysis import *

INSTR = "provide as" 
INSTR += "\n 1st input: either 'sparse' or 'corrected' contacts"
INSTR += "\n 2nd input: either 'none' or 'ipw_global'"
INSTR += '\n 3rd input: Sparsification Iteration from 0 to 20'

print(INSTR)

INPUT_CONTACT = sys.argv[1]
INPUT_CONTACT_CORRECTION = sys.argv[2]
INPUT_SPARSE_ITER = sys.argv[3]

#experimental condition:
#start the simulation from the first weekday 
#because there are few contacts in the weekend and the seed does not propagate
FROM_weekday = True

#[1] MISSINGNESS SCENARIOS
#Current folder collecting experimental results for epidemic simulations
FOLD_exp = 'f01_014_D1_iter_sparsified_epid_simulations_from_weekday/'
#list of sparsification scenarios
List_ss = ['Data_driven_1wgcu']#, 'Random_uniform']
#list of sparsification levels
Levels = gen_sparsity_ranges()
#number of sparsification iterations
Ranges_iter = [INPUT_SPARSE_ITER]
#list of all sparse scenarios combinations
List_Sparse_Scenarios =  list(itertools.product(List_ss, Levels, Ranges_iter))

#[2] CALIBRATION SPECIFICS
#[2.1] REFERENCE CURVE
#EMO will be used only for the complete dataset
DICT_EMO = import_EMO_complete(FOLD_exp)
#set the reference curve for the optimization task
metric_ref = 'median_I'
Curve_ref = gen_reference_curve(DICT_EMO, metric_ref)
#set the metric for objective function computation
metric_obj = 'RMSE'

#[2.2] CALIBRATION SIMULATION PARAMETERS
#Parameter boundaries for the bayesian parameter search
Betas  = np.linspace(.5e-4, 1e-2, 20)
#gamma at second resolution
#recovery period of 1-day
gamma_second = 1/(24*3600) 
#compound probability conversion
lambda_gamma_minute = lambda gamma_second : 1 - (1 - gamma_second)**60
Gamma_min = lambda_gamma_minute(gamma_second/8)
Gamma_max = lambda_gamma_minute(gamma_second)
Gammas = np.linspace(Gamma_min, Gamma_max, 10)
N_inits = [1,3,5,10]#,15,20,30,50]
GRID = list(itertools.product(Betas, Gammas, N_inits))
GRID_stats = pd.DataFrame(GRID, columns = ['beta','gamma','seedsize']).describe().loc[['min','max']]

#[2.2.b] simulation parameters
#number of trials for optuna parameter search
n_trials = 100
#number of simulations 
N_epid_iter = 100

#Get users and date-range
df_contacts, USERS_select, Date_range = import_complete_contacts_users_drange()

def epid_modeling(DICT_contacts, 
                  List_Sparse_Scenarios, 
                  emv_name,
                  FOLD_exp,
                  ORACLE_MODELING = True, 
                  contact_correction = 'ipw_global', 
                  CALIBRATION_MODELING = True):
    '''
    DICT_contacts : collection of contacts
    List_Sparse_Scenarios : list of missingness configurations
    emv_name : type of epidemic modeling variation
    correction: type of contact correction
    FOLD_exp : fold where to save the experiments
    ORACLE_MODELING: use the ground-truth simulation parameters
        - NB: if ORACLE MODELING does not Calibrate
    '''

    #specifics of calibration experiments
    str_calib = f'{emv_name}'
    if contact_correction is not None: 
        str_calib += f'_{contact_correction}'
    str_calib += f'_{metric_ref}_{metric_obj}_{n_trials}_trials'
    
    for Sparse_scenario in List_Sparse_Scenarios:
    
        print(Sparse_scenario) 
        s, l, N_si = Sparse_scenario

        #[3.3] SAVING THE RESULTS
        FOLD_ss = get_path_sparse_scenario(FOLD_exp, Sparse_scenario)
        gen_fold(FOLD_ss)
        print(f'created {FOLD_ss}')
        

        if CALIBRATION_MODELING:
            
            #dictionary where calibration study and df of fitted parameters are saved 
            DICT_study = {}
            DICT_study_outcome = {}
            
        
            #[3.1] determine the optimal parmeters
            study = optuna_param_search(Curve_ref, 
                                        metric_obj,
                                        GRID_stats, 
                                        DICT_contacts, 
                                        Sparse_scenario, 
                                        metric_ref,
                                        USERS_select, 
                                        Date_range, 
                                        N_iter = N_epid_iter, 
                                        from_weekday = FROM_weekday, 
                                        n_trials = n_trials)
        
            #store the results of the study
            best_params = get_study_best_params(study)
            best_value  = study.best_value
    
            DICT_study[Sparse_scenario] = study
            DICT_study_outcome[Sparse_scenario] = (best_params, best_value)
        
            #[3.2] run the epidemic modeling on best fitting parameters
            COLLECT_simulations, COLLECT_epid_metrics = iter_epid_simulation(DICT_contacts[Sparse_scenario], 
                                                                             Date_range,
                                                                             USERS_select, 
                                                                             pars = (best_params[0], best_params[1]), 
                                                                             n_init = best_params[2], 
                                                                             N_iter = N_epid_iter,
                                                                             from_weekday = FROM_weekday)
    
            #[3.3] SAVING THE RESULTS
            #save epid statistics
            df_epid_stats = compute_epid_stats(COLLECT_epid_metrics).reset_index()            
            df_epid_stats.to_csv(f'{FOLD_ss}df_epid_stats_{l}_{str_calib}.csv')
            
            #save epid simulations
            with open(f'{FOLD_ss}simulations_{l}_{str_calib}.pkl', 'wb') as f:
                pickle.dump(COLLECT_simulations, f)  
            #save the fitted parameters 
            data = []
            for (label, params, idx), ((x1, x2, x3), score) in DICT_study_outcome.items():
                data.append([label, params, idx, x1, x2, x3, score])
            columns = ["s", "l", "N_si", 
                       "beta", "gamma", "n_init", 
                       "score"]#epide
            df_study = pd.DataFrame(data, columns=columns)  
            df_study.to_csv(f'{FOLD_ss}df_study_best_params_{l}_{str_calib}.csv')
            #save the study to get better information on the calibration procedure
            with open(f'{FOLD_ss}study_{l}_{str_calib}.pkl', 'wb') as f:
                pickle.dump(DICT_study, f)

        if ORACLE_MODELING:
            
            #simulate with the ground truth experimental parameters
            #epidemic experimental parameters: (beta, gamma) tuple, seed size, number of epidemic simulation iterations
            epid_pars, n_init, N_iter = gen_complete_epid_groundtruth()
            COLLECT_simulations, COLLECT_epid_metrics = iter_epid_simulation(DICT_contacts[Sparse_scenario], 
                                                                             Date_range,
                                                                             USERS_select, 
                                                                             epid_pars, 
                                                                             n_init = n_init, 
                                                                             N_iter = N_iter,
                                                                             from_weekday = FROM_weekday)
                        
            df_epid_stats = compute_epid_stats(COLLECT_epid_metrics).reset_index()
            if 'calibration_biased' in emv_name:
                df_epid_stats.to_csv(f'{FOLD_ss}df_epid_stats_{l}_oracle_biased.csv')
                with open(f'{FOLD_ss}simulations_{l}_oracle_biased.pkl', 'wb') as f:
                    pickle.dump(COLLECT_simulations, f)
            else: 
                df_epid_stats.to_csv(f'{FOLD_ss}df_epid_stats_{l}_oracle_{contact_correction}.csv')
                with open(f'{FOLD_ss}simulations_{l}_oracle_{contact_correction}.pkl', 'wb') as f:
                    pickle.dump(COLLECT_simulations, f)

                
if INPUT_CONTACT == 'sparse':

    #import Data_driven_1wgcu sparse contacts
    #import complete and sparse contacts (I will need them for the simulation once I find the grid minimizers)
    DICT_contacts_sparse = import_contacts_complete_sparse(List_ss, 
                                                           Levels, 
                                                           Ranges_iter, 
                                                           import_complete = False)
    
     
    #[1] calibration on biased contacts 
    emv_name = 'calibration_biased'

    #Oracle on biased contacts
    print('Oracle on biased contacts')
    epid_modeling(DICT_contacts_sparse, 
                  List_Sparse_Scenarios, 
                  emv_name,
                  FOLD_exp,
                  ORACLE_MODELING = True,
                  CALIBRATION_MODELING = False, 
                  contact_correction = None)
    
    #Calibration on biased contacts
    print('Calibration on biased contacts')
    epid_modeling(DICT_contacts_sparse, 
                  List_Sparse_Scenarios, 
                  emv_name,
                  FOLD_exp,
                  ORACLE_MODELING = False, 
                  CALIBRATION_MODELING = True, 
                  contact_correction = None)

if INPUT_CONTACT == 'corrected':
    
    print('Calibration and Oracle modeling on corrected contacts')
    
    #[2-3] calibration and oracle modeling on corrected contacts 
    #generate the corrected contacts
    #rescale the contacts with inverse probabilistic weighting
    contact_correction = INPUT_CONTACT_CORRECTION
    
    DICT_contacts_corrected = {k : ipw_rescale_contacts(k, 
                                                        USERS_select, 
                                                        Date_range,
                                                        rescaling = contact_correction)
                               for k in List_Sparse_Scenarios}
    
    emv_name = 'calibration_corrected'
    epid_modeling(DICT_contacts_corrected, 
                  List_Sparse_Scenarios, 
                  emv_name,
                  FOLD_exp,
                  ORACLE_MODELING = True, 
                  CALIBRATION_MODELING = True, 
                  contact_correction = contact_correction)















