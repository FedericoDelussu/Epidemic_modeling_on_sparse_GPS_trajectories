import pandas as pd
import numpy as np
import os
from Modules import config
from Modules import analysis

#folder collecting a csv file for each individual
FOLD_input = config.FOLD_DTU_INPUT
#folder where the output of the analysis is stored
FOLD_save = config.FOLD_DTU

tz = 'Europe/Copenhagen'

Files = np.sort(os.listdir(FOLD_input))
Files = [f for f in Files if '.csv' in f]

for fname in Files: 
    
    print(fname)
    
    traj = pd.read_csv(f'{FOLD_input}{fname}', index_col =0)
    traj.columns = ['id', 'time_utc', 'lat', 'lon', 'accuracy'] 
    
    traj_df = analysis.preprocess_GPS_mobility_trajectory(traj, 
                                                          tz = tz)
    
    traj_df.to_csv(f'{FOLD_save}01_trajectories_preprocessed/{fname}')
