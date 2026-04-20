import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import itertools

#PATHS FOR EXPERIMENTS ON DTU AND SYNTHETIC DATASET 
#this folder collects a csv for each trajectory
FOLD_DTU_INPUT = '/home/fedde/work/Project_Penn/TMP/f_001_D1_samples_original/'
FOLD_SYNTH_INPUT = '' #tbd

FOLD_DTU = 'Pipeline_output_DTU/'
FOLD_SYNTH = 'Pipeline_output_SYNTH/'

###########################################
######### EXPERIMENTAL VARIABLES ##########
###########################################

#INDEX OF USERS WITH COMPLETE TRAJECTORIES OVER THE STUDY PERIOD
USERS_select = np.array([0,  10, 100, 101, 102, 104, 105, 106, 107, 110, 111, 113, 114,
                         119, 121, 123, 125, 128, 129,  13, 133, 135, 136, 137, 138, 140,
                         146, 147, 150, 151, 153, 154, 155, 156, 157, 158, 160, 163, 164,
                         165, 166, 169,  17, 170, 172, 173, 178, 179,  18, 180, 183, 185,
                         186, 187, 189,  19, 191, 192, 193, 195, 196, 197, 199,   2, 200,
                         202, 203, 204, 206, 208, 209,  21, 214, 215, 221, 222, 223, 224,
                         225, 226, 227, 230, 231, 232, 233, 237, 239, 240, 241, 242, 243,
                         244, 246, 247, 248, 252, 254, 255, 258, 259, 263,  27, 270, 271,
                         272, 274, 276, 279, 281, 284, 285, 286, 287, 288, 290, 293, 294,
                         295, 296, 298, 299,   3,  30, 300, 301, 303, 306, 307, 308,  31,
                         313, 314, 315, 317, 319,  32, 324, 325, 326, 328, 329,  33, 330,
                         334, 335, 336, 339, 340, 341, 342, 343, 344, 345, 347,  35, 352,
                         354, 355, 359,  36, 360, 361, 363, 364, 367, 368,  37, 370, 371,
                         374, 375, 376, 379,  38, 380, 382, 383, 386, 389,  39, 391, 394,
                         395, 396, 397, 399,  40, 404, 405, 406, 407, 408, 409, 414, 416,
                         418,  42, 422, 423, 425, 426, 427, 428, 430, 434, 435, 436, 437,
                         440, 441, 443, 444, 446, 448, 449,  45, 450, 455, 458, 459,  46,
                         460, 461, 463, 464, 465, 466, 467, 468, 470, 472, 473, 474, 476,
                          48, 480, 482, 483, 487,  49, 492, 493, 495, 496,   5,  50, 501,
                         505, 506, 508,  51, 512, 513, 514, 516, 517, 518, 519,  52, 520,
                         521, 524, 525, 526, 527, 529,  53, 530, 532, 533, 536, 537,  54,
                         540, 542, 543, 544, 545, 546, 547, 549, 554, 557, 558, 559, 560,
                         563, 568,  57, 571, 575, 576, 577, 579,  58, 582, 585, 586, 588,
                         589,  59, 590, 592, 593, 595, 596, 597,  60, 600, 601, 607, 609,
                          61, 611, 612, 613, 617, 619,  62, 621, 624,  63, 630, 635, 637,
                         639, 641, 643, 644, 645, 648, 651, 655, 666, 668, 673,  68, 680,
                         694, 695,   7,  70,  71,  72,  73,  74,  75,  76,  78,   8,  80,
                         81,  82,  83,  84,  85,  89,  91,  92,  93,  95,  96,  99])

N_users = len(USERS_select) 

#[3] EXPERIMENTAL VARIABLES
#time-range for the selection of the hourly record indicator sequences
Time_range = [dt.datetime(2014,2,1), dt.datetime(2015,2,1)]
#study period for the selection of the complete trajectories
Study_period = [dt.datetime(2014,2,8), dt.datetime(2014,3,7)]
#study period for epidemic modeling (starting on a monday)
Study_period_em = [dt.datetime(2014,2,10), dt.datetime(2014,3,7)]


#List of sparsity ranges
Levels = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)]
level_complete = (0,0.05)

#DAYS IN THE STUDY PERIOD FOR EPIDEMIC MODELING
Dates = pd.date_range(dt.date(2014,2,10), dt.date(2014,3,7)).date
#study period plus the following day (for plotting the epidemic curves which account an additional day)
Dates_plus1 = np.append(Dates, Dates[-1]+dt.timedelta(days=1))

#Epidemic parameters
groundtruth_pars = {'beta': 0.000845, #probability of infection given 1-minute contact
                    'gamma': 0.268,   #probability of recovery over 1 day
                    'seedsize': 3}    # seed size of the SIR model

#[2.2] CALIBRATION SIMULATION PARAMETERS
#[2.2.a] parameter boundaries for the bayesian parameter search
Betas  = np.linspace(.5e-4, 1e-2, 20)
#gamma at second resolution
#recovery period of 1-day
gamma_second = 1/(24*3600) 
#compound probability conversion
lambda_gamma_minute = lambda gamma_second : 1 - (1 - gamma_second)**60
Gamma_min = lambda_gamma_minute(gamma_second/8)
Gamma_max = lambda_gamma_minute(gamma_second)
Gammas = np.linspace(Gamma_min, Gamma_max, 10)
N_inits = [1,3,5,10]
GRID = list(itertools.product(Betas, Gammas, N_inits))
GRID_stats = pd.DataFrame(GRID, columns = ['beta','gamma','seedsize']).describe().loc[['min','max']]
#converting gamma at daily scale
GRID_stats['gamma'] = GRID_stats['gamma'].apply(lambda x: 1 - (1-x)**(60*24))


#[1] CHOICE OF THE COLORPALETTE 

#SIR modeling
COLOR_A = '#009E73'
COLOR_B = '#D55E00'
COLOR_R = '#EAD9B8'

#Gap colors
COLOR_GAPS = '#1A1A1A'   # Charcoal

#Epidemic modeling
COLOR_GT = '#3A86FF'
COLOR_BS = '#8338EC'  
COLOR_CALIB_BS = '#F781BF'
COLOR_CC       = '#104911' 
COLOR_CALIB_CC = '#548C2F'

#Random baselines
COLOR_RANDOM_SHUFFLING = '#E63946'
COLOR_RANDOM_UNIFORM   = '#8D99AE'

#Colors for different levels of sparsity from 10-20, 20-30, 30-40, 40-50 to 50-60
COLORS_LEVEL = [plt.cm.YlOrRd(i) for i in np.linspace(0.4, 1.0, 5)]

#[2] AX PARAMS
ax_title_size        = 18
ax_label_size        = 15
ax_text_size         = 12
ax_tick_size         = 12 
ax_legend_font_size  = 12
ax_legend_title_size = 15


Level_colors = [plt.cm.YlOrRd(i) for i in np.linspace(0.4, 1.0, 5)]
DICT_colors_level = dict(zip(Levels, Level_colors))

def convert_to_percent_range(input_str):
    # Remove parentheses and split the string
    numbers = input_str.strip('()').split(',')
    # Convert the numbers to percentage format
    lower = int(float(numbers[0]) * 100)
    upper = int(float(numbers[1]) * 100)
    # Return the formatted string
    return f'{lower}-{upper}' 
    
DICT_rename_levels = {l: convert_to_percent_range(str(l)) for l in Levels}
DICT_rename_levels.update({str(l):k for l,k in DICT_rename_levels.items()})

Levels_str = [convert_to_percent_range(str(l)) for l in Levels]

#Sparsification iterations
Ranges_iter = range(50)


#Sparsification approaches
List_ss = ['Data_driven_1wgcu',
           'Random_keepdurations_1wgcu',
           'Random_uniform_1gcu']

List_ss_rename = ['Data driven',
                  'Random shuffling',
                  'Random uniform']

#List of sparsity approaches
List_sparsity = ['Data_driven',
                 'Random_shuffling',
                 'Random_uniform']

DICT_rename_ss = dict(zip(List_ss, List_ss_rename))


DICT_colors_ss = {'Data_driven': COLOR_BS,
                  'Random_uniform': COLOR_RANDOM_UNIFORM,
                  'Random_keepdurations': COLOR_RANDOM_SHUFFLING}
DICT_colors_ss.update({'Data driven': COLOR_BS,
                       'Random uniform': COLOR_RANDOM_UNIFORM,
                       'Random shuffling': COLOR_RANDOM_SHUFFLING})


Contact_corrections  = ['ipw_hourofday_weekday']
DICT_rename_contact_corrections = {'ipw_hourofday_weekday' : '(ipw. hour-weekperiod)'}
EMVs = ['calibration_biased', 'oracle_biased'] 
DICT_rename_EMVs = {'oracle_biased': 'Oracle on biased contacts',
                    'calibration_biased' : 'Calibration on biased contacts'} 
for c in Contact_corrections:
    EMVs += [f'calibration_corrected_{c}', 
              f'oracle_{c}']
    c_rename = DICT_rename_contact_corrections[c]
    DICT_rename_EMVs[f'calibration_corrected_{c}'] = f'Calibration on corrected contacts {c_rename}'
    DICT_rename_EMVs[f'oracle_{c}'] = f'Oracle on corrected contacts {c_rename}'
DICT_colors_emv = {'oracle_biased' : COLOR_BS,
                   'calibration_biased' : COLOR_CALIB_BS,
                   'oracle_ipw_hourofday_weekday': COLOR_CC,       
                   'calibration_corrected_ipw_hourofday_weekday': COLOR_CALIB_CC} 
DICT_colors_emv.update({'oracle_biased_contacts' : COLOR_BS,
                        'calibration_biased_contacts' : COLOR_CALIB_BS,
                        'oracle_corrected_contacts': COLOR_CC,
                        'calibration_corrected_contacts': COLOR_CALIB_CC})

List_emvs =  ['oracle_biased',
              'calibration_biased',
              'oracle_rescaled',
               'calibration_rescaled']

COLOR_emvs = [COLOR_BS, 
              COLOR_CALIB_BS,
              COLOR_CC, 
              COLOR_CALIB_CC]

DICT_colors_emv.update(dict(zip(List_emvs, COLOR_emvs)))

    
#[5] Debiasing approaches
Emo_metrics_names = ['peak_day', 
                     'peak_size', 
                     'final_size', 
                     'epid_duration', 
                     'day_final_case'] 

DICT_rename_class_epid = {'I': 'Infected', 
                          'S': 'Susceptible'}

DICT_rename_emo_metrics = {'peak_size'  : 'Peak', 
                           'final_size' : 'Total',
                           'peak_day' : 'Peak day', 
                           'epid_duration'  : 'Day of last recovery',
                           'day_final_case' : 'Day of last case'}

DICT_rename_emo_metrics_norm = {'peak_size'  : 'Maximum of daily infected (%)', 
                                'final_size' : 'Total infected (%)',
                                'peak_day' : 'Peak day', 
                                'epid_duration'  : 'Day of last recovery',
                                'day_final_case' : 'Day of last case'}

DICT_rename_em_fraction = {'peak_size' : 'max I (%)', 
                           'final_size': 'tot I (%)'}











