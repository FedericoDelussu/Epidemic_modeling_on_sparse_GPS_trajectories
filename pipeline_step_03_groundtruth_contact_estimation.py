import pandas as pd 
import gc
import sys
from Modules.utils import * 
from Modules.analysis import *

PATH_stops = sys.argv[0]
PATH_contacts = sys.argv[1]

#Study period for contact estimation (tuple of 2 dt.datetime objects)
Study_period = sys.argv[2]

#[0] IMPORT STOPS
#Lachesis parameters 
# dur_min    : minimum stop duration (minutes), 
# dt_max     : max consectuive ping time difference (minutes), 
# delta_roam : max diameter (meters)
S = (10,360,50)
dur_min, dt_max, delta_roam = S  
str_lach  = f'mintime_{dur_min}_maxdtime_{dt_max}_maxdiam_{delta_roam}'
df_stops = pd.read_csv(PATH_save + 'df_stops_' + str_lach + '.csv',
                       index_col = 0, 
                       parse_dates = ['start_time', 'end_time'])

ALL_users = df_stops['id'].unique()

#[1] CONTACT ESTIMATION

#geohash resolution for contact estimation
ghr = 8
#temporal aggregation level of contact estimation
t_step = '1hour'


#Partition of the study period into 1day intervals
#The contact estimation is time expensive; so we partition the study period in intervals of 1day 
#and perform the contact estimation for each day
SP_partition = pd.date_range(Study_period[0], Study_period[1], freq = 'D')
SP_partition = [(a1,a2) for a1,a2 in zip(SP_partition[:-1], SP_partition[1:]) ]

for SP0 in SP_partition:

    Cols_select = ['id', 
                   'start_time', 'end_time', 
                   'geohash9', 
                   'medoid_x', 'medoid_y']
    df_stops_SP = filter_stops(df_stops, 
                               ALL_users, 
                               Date_range = SP0, 
                               Cols_select = Cols_select, 
                               reset_ghr = ghr)

    #contact estimation
    #
    df_cwithin, df_cmargin = estimate_contacts(df_stops_SP, 
                                               ghr, 
                                               t_step)

    SP0_str = SP0[0].strftime('%Y-%m-%d')
    
    df_cwithin.to_csv(f'{PATH_contacts}df_cwithin_{SP0_str}.csv')
    df_cmargin.to_csv(f'{PATH_contacts}df_cmargin_{SP0_str}.csv')

    del df_stops_SP, df_cwithin, df_cmargin
    gc.collect()        