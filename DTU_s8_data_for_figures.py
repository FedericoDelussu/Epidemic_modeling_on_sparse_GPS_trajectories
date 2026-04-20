
import glob
import pickle

import pandas as pd
import numpy as np
import os
import datetime as dt
from multiprocessing import Pool

from Modules import config
from Modules import analysis
from Modules import data_figures

#folder where the output of the analysis is stored
FOLD_output = config.FOLD_DTU
#folder containing the corrected contacts
FOLD_contact_corrected = f'{FOLD_output}06_contacts_data-driven_ipw-weight/'
#folder where saving the epidemic modeling outcomes
FOLD_emo = f'{FOLD_output}07_epidemic_modeling_outcomes/'
os.makedirs(FOLD_emo, exist_ok=True)

#Collection of files to create for making the figures
DICT_files = {#DATA FOR FIGURES IN THE MAIN AND SUPPLEMENTARY
              'df_trajectory_record_indicator.csv'                                : False,
              'df_user_contact_count.csv'                                         : False,
              'df_emo_metrics_debiasing.csv'                                      : True,
              'df_emo_metrics_sparsification.csv'                                 : True,
              'df_calibration_biased.csv'                                         : True,
              'df_calibration_corrected.csv'                                      : True,
              'df_R0_grid_estimates.csv'                                          : True,
              #DATA USED FOR SUPPLEMENTARY FIGURES      
              'df_supp_contact_count_change-duration.csv'                         : True,
              'df_supp_contact_count_sparsification.csv'                          : True,
              'df_supp_contact_average_duration_sparsification.csv'               : True,
              'df_supp_contact_average_R0_sparsification.csv'                     : True,
              'df_supp_contact_average_duration_data-driven_biased-corrected.csv' : True,
              'df_supp_contact_average_R0_data-driven_biased-corrected.csv'       : True}


Iters = range(1,51)

file = 'df_user_contact_count.csv'
if DICT_files[file]:
    data_figures.save_hourly_counts(file, Iters=Iters)


file = 'df_emo_metrics_sparsification.csv'
#save also the epidemic curves
if DICT_files[file]:
    data_figures.save_emo_sparsification('df_emo_curves_sparsification.csv', file, Iters=Iters)

file = 'df_emo_metrics_debiasing.csv'
#save also the epidemic curves
if DICT_files[file]:
    data_figures.save_emo_debiasing('df_emo_curves_debiasing.csv', file, Iters=Iters)

#OUTCOMES FROM CALIBRATION
















