import pandas as pd 
import numpy as np
from Modules.utils import * 
from Modules.analysis import *

#experimental condition 
#starts the simulation from the first weekday 
#(because the contacts are sparse in the weekend)
FROM_weekday = True
#epidemic experimental parameters: (beta, gamma) tuple, seed size, number of epidemic simulation iterations
epid_pars, n_init, N_iter = gen_complete_epid_groundtruth()

#EMO FOLDER
FOLD_emo = DICT_paths['TMP'] + 'f01_014_D1_iter_sparsified_epid_simulations_from_weekday/'
#subfolder for specific epidemic regime
DICT_epid_pars = gen_dict_epid_pars()
str_epid_gt = import_str_complete_epid_groundtruth_str()
FOLD_epid_scenario = FOLD_emo + str_epid_gt + '/'
gen_fold(FOLD_epid_scenario)

#[0] COMPLETE DATA SIMULATION
df_contacts, USERS_select, Date_range = import_complete_contacts_users_drange() 
N_users = len(USERS_select)
W = gen_contact_daily(df_contacts, USERS_select)

COLLECT_simulations_complete, COLLECT_epid_metrics_complete = iter_epid_simulation(W, 
                                                                                   Date_range,
                                                                                   USERS_select, 
                                                                                   epid_pars, 
                                                                                   n_init = n_init, 
                                                                                   N_iter = N_iter,
                                                                                   from_weekday = FROM_weekday)

df_epid_stats = compute_epid_stats(COLLECT_epid_metrics_complete).reset_index()
df_epid_stats.to_csv(f'{FOLD_epid_scenario}/df_epid_stats.csv')
with open(f'{FOLD_epid_scenario}simulations.pkl', 'wb') as f:
    pickle.dump(COLLECT_simulations_complete, f)

#[1] SIMULATION OF SPARSIFIED DATA FOR A SINGLE SPARSIFICATION ITERATION

#list of sparsification scenarios
str_1wgcu = '1wgcu'
Sparse_scenarios = ['Data_driven','Random_uniform', 'Random_keepdurations']
List_ss = [f'{s}_{str_1wgcu}' for s in Sparse_scenarios]

#list of sparsification level
Levels = gen_sparsity_ranges()

#[TO-CHANGE] list of sparsification iteration indexes
Range_sparse_iterations = np.arange(20)

#list of EM pipeline steps (with execution condition)
DICT_execute_emv = {'emv_basic': True, 
                    'emv_rescaled_contacts': False, 
                    'emv_rescaled_contacts_squared': False,
                    'emv_param_calibration': False}

#[i0] select a sparsification realization
for N_sparse_iter in Range_sparse_iterations:
    FOLD_iter = f'{FOLD_epid_scenario}Iter_sparse_{N_sparse_iter}/' 
    gen_fold(FOLD_iter)
    #[i1] select a sparsification scenario
    for ss in List_ss:
        
        #fold of sparsification scenario
        FOLD_ss = f'{FOLD_iter}{ss}/'
        gen_fold(FOLD_ss)
        
        #[i2] select sparsification level
        for level in Levels:
            
            print(f'{ss} level {level}')
            df_contacts_level = import_sparse_contacts(ss, level, N_sparse_iter)
            df_contacts_level['date_hour'] = pd.to_datetime(df_contacts_level['date_hour'], format = 'mixed')
            W_level = gen_contact_daily(df_contacts_level, USERS_select)
            #print(W_level.keys())
            #[EV0] Basic Epidemic Modeling
            emv_name = 'emv_basic'
            if DICT_execute_emv[emv_name]:
                
                print('\t' + emv_name)  
                COLLECT_simulations, COLLECT_epid_metrics = iter_epid_simulation(W_level, 
                                                                                 Date_range,
                                                                                 USERS_select, 
                                                                                 epid_pars, 
                                                                                 n_init = n_init, 
                                                                                 N_iter = N_iter,
                                                                                 from_weekday = FROM_weekday)
                
                df_epid_stats = compute_epid_stats(COLLECT_epid_metrics).reset_index()
                df_epid_stats.to_csv(f'{FOLD_ss}df_epid_stats_{level}.csv')
                with open(f'{FOLD_ss}simulations_{level}.pkl', 'wb') as f:
                    pickle.dump(COLLECT_simulations, f)
    
            #[EV1] Epidemic Modeling with Rescaled Contacts
            emv_name = 'emv_rescaled_contacts'
            if DICT_execute_emv[emv_name]:
                print(emv_name)            
                avg_sparsity = 0.5*(level[0] + level[1])
                #contact rescaling factor
                c_r = 1/(1 - avg_sparsity)
                #dictionary of rescaled daily contacts
                W_level_rescaled = {}
                for k,df_v_ in W_level.items():
                    df_v = df_v_.copy()
                    df_v *= c_r
                    W_level_rescaled[k] = df_v
                    
                COLLECT_simulations, COLLECT_epid_metrics = iter_epid_simulation(W_level_rescaled, 
                                                                                 Date_range,
                                                                                 USERS_select, 
                                                                                 epid_pars, 
                                                                                 n_init = n_init, 
                                                                                 N_iter = N_iter,
                                                                                 from_weekday = FROM_weekday)
                
                df_epid_stats = compute_epid_stats(COLLECT_epid_metrics).reset_index()
                df_epid_stats.to_csv(f'{FOLD_ss}df_epid_stats_{level}_{emv_name}.csv')
                with open(f'{FOLD_ss}simulations_{level}_{emv_name}.pkl', 'wb') as f:
                    pickle.dump(COLLECT_simulations, f)
    
            #[EV1] Epidemic modeling with rescaled contacts
            emv_name = 'emv_rescaled_contacts_squared'
            if DICT_execute_emv[emv_name]:   
                print(emv_name)            
                #get the sparse contacts
                df_contacts_level = get_sparse_contacts(Sparse_scenario, level)
                W_level = gen_contact_daily(df_contacts_level, USERS_select)            
                avg_sparsity = 0.5*(level[0] + level[1])
                #contact rescaling factor
                c_r = 1/(1 - avg_sparsity)**2
                #dictionary of rescaled daily contacts
                W_level_rescaled = {}
                for k,df_v_ in W_level.items():
                    df_v = df_v_.copy()
                    df_v['n_minutes'] *= c_r
                    W_level_rescaled[k] = df_v
                
                COLLECT_simulations, COLLECT_epid_metrics = iter_epid_simulation(W_level_rescaled, 
                                                                                 Date_range,
                                                                                 USERS_select, 
                                                                                 epid_pars, 
                                                                                 n_init = n_init, 
                                                                                 N_iter = N_iter,
                                                                                 from_weekday = FROM_weekday)
    
                df_epid_stats = compute_epid_stats(COLLECT_epid_metrics).reset_index()
                df_epid_stats.to_csv(f'{FOLD_ss}df_epid_stats_{level}_{emv_name}.csv')
                with open(f'{FOLD_ss}simulations_{level}_{emv_name}.pkl', 'wb') as f:
                    pickle.dump(COLLECT_simulations, f)
    
            #[EV2] Epidemic modeling with parameter calibration
            emv_name = 'emv_param_calibration'
            if DICT_execute_emv[emv_name]:
                print(emv_name)            
                FOLD_grid_search_info = f'{FOLD_ss}grid_search_info/'
                gen_fold(FOLD_grid_search_info)
                        
                #[0] get the groundtruth; cumulative number of cases, from the simulation on complete data
                #N_users = len(USERS_select)
                #CC_gt = np.mean(N_users - COLLECT_simulations_complete[:,:,0], axis = 0)
    
                #select the grid elements for searching the optimal epidemiological parameters
                DICT_grid = gen_epid_calibration_grid_dict()
                grid_version = 'v3'
                epid_pars_grid = DICT_grid[grid_version]
                
                def calibrate_epid_pars(epid_pars_grid, 
                                        W_level, 
                                        Date_range,
                                        #CC_gt, 
                                        N_grid_iter = 100):
                    '''
                    epid_pars_grid: list of epidemiological parameters
         0           CC_gt: cumulative cases groudntruth
                    W_level: collection of daily contacts
                    '''
                
                    #epid_pars_grid_rmse = []
                    
                    for EPG0 in epid_pars_grid:
                        
                        print(EPG0)
                        
                        #simulate for a specific grid element
                        grid_par, n_init = EPG0[:2], EPG0[2]
                        COLLECT_simulations, COLLECT_epid_metrics = iter_epid_simulation(W_level, 
                                                                                         Date_range, 
                                                                                         USERS_select, 
                                                                                         grid_par, 
                                                                                         n_init = n_init, 
                                                                                         N_iter = N_grid_iter, 
                                                                                         from_weekday = True)
                
                        _beta    = np.round(grid_par[0]*100, 6)
                        _gamma   = np.round(grid_par[1]*100, 6)
                        str_epid = f'_beta_{_beta}_gamma_{_gamma}_ninit_{n_init}_Niter_{N_grid_iter}' 
                        with open(f'{FOLD_grid_search_info}simulations_{level}_{str_epid}_grid_{grid_version}.pkl', 'wb') as f:
                            pickle.dump(COLLECT_simulations, f)
                            
                        #CC_grid_par = np.mean(N_users - COLLECT_simulations[:,:,0], axis = 0)
                        #rmse = np.sqrt( np.mean((CC_gt - CC_grid_par)**2) )
                        #epid_pars_grid_rmse.append(rmse)
                        
                #[2] Run the simulation for each element of the grid 
                calibrate_epid_pars(epid_pars_grid, 
                                    W_level, 
                                    Date_range)#,CC_gt)
                
                #THIS PART IS IN SCRIPT-009 (RMSE is computed both for susceptible and for infected)
                #Before it was computed for the cumulative number of cases (but it is equivalent to susceptibles)

                ##[2.1] save the parameters and rmse computation for the specific sparsity level
                #df_epg = pd.DataFrame(np.array([list(e) for e in epid_pars_grid]), 
                #                      columns = ['beta', 'gamma','n_init'])
                #    
                #df_epg['rmse'] = epid_pars_grid_rmse
                #df_epg.to_csv(f'{FOLD_grid_search_info}df_epg_rmse_{level}_grid_{grid_version}.csv') 
                #              #mode='a', 
                #              #header=False, 
                #              #index=False)
                #
                ##[3.1] find the calibrated parameters which minimize the RMSE
                #par_calibrated = epid_pars_grid[np.argmin(epid_pars_grid_rmse)]
                #epc, nc = par_calibrated[:2], par_calibrated[2]
                #
                ##[3.2] run the epidemic modeling and save the simulation
                #COLLECT_simulations, COLLECT_epid_metrics = iter_epid_simulation(W_level, 
                #                                                                 Date_range,
                #                                                                 USERS_select, 
                #                                                                 epc, 
                #                                                                 n_init = nc, 
                #                                                                 N_iter = N_iter,
                #                                                                 from_weekday = True)
                #
                #df_epid_stats = compute_epid_stats(COLLECT_epid_metrics).reset_index()
                #df_epid_stats.to_csv(f'{FOLD_ss}df_epid_stats_{level}_{emv_name}_grid_{grid_version}.csv')
                #with open(f'{FOLD_ss}simulations_{level}_{emv_name}_grid_{grid_version}.pkl', 'wb') as f:
                #    pickle.dump(COLLECT_simulations, f)
            
        

        

















