import pandas as pd 
import numpy as np 
import sys
import os
import shutil
import datetime as dt
import math
import matplotlib.pyplot as plt
import skmob 
from skmob.preprocessing import filtering

#path containing a collection of .csv files of the individual trajectories
#each .csv should have columns ['id', 'time_utc', 'lat','lon']
path = sys.argv[0]

#Folder for saving the preprocessed trajectories 
FOLD_save = sys.argv[1]

for f in Files: 

    #read a trajectory dataset for the single individual
    df_f = pd.read_csv(path + f, index_col = 0)
    df_f.columns = ['id', 'time_utc', 'lat','lon']

    #convert the dataframe into a skmob trajectory dataframe - default CRS is 4326
    c_dict= {'lon':'lng', 
             'time_utc' : 'datetime'}

    tdf = skmob.TrajDataFrame(df_f.rename(columns = c_dict), 
                              latitude  = 'lat', 
                              longitude = 'lng', 
                              datetime  = 'datetime', 
                              timestamp = True)
    
    #filter out pings with speed exceeding 100km/h
    ftdf = filtering.filter(tdf, max_speed_kmh = 100)

    #down-sample to 1-minute resolution
    ftdf['datetime'] = ftdf['datetime'].dt.floor('T')
    ftdf_min = ftdf.drop_duplicates(subset=['datetime'])        
    ftdf_min.to_csv(FOLD_save + 'filtered_traj_1min_downsample/' + f)
