import pandas as pd 
from Modules.utils import * 
from Modules.analysis import *

#Condition for stop tolerance
BOOL_tolerate_stops = False
COND_START_DAY_01 = True

Study_period = pd.DatetimeIndex([dt.datetime(2014,2,1), dt.datetime(2015,2,1)])

#[0] import location interpolated dataset by Lachesis
#S = (10, 360, 50)
#dur_min, dt_max, delta_roam = S
#par_lach  = f'mintime_{dur_min}_maxdtime_{dt_max}_maxdiam_{delta_roam}'
#PATH_loc = DICT_paths['TMP'] + 'f_007_D1_lachesis_stops_AND_loc-1hdi/'
#df_LOC_1hdi = pd.read_csv(PATH_loc + 'df_LOC_1hdi_' + par_lach + '.csv', 
#                          index_col = 0, 
#                          parse_dates = ['date_hour'])
#
#df_LOC_1hdi = df_LOC_1hdi[df_LOC_1hdi['date_hour'].between(Study_period[0], Study_period[1])]
#
##Do not consider the stop interpolation for assessing completeness
#if not BOOL_tolerate_stops: 
#    #filter out hours interpolated by lachesis 
#    df_LOC_1hdi = df_LOC_1hdi[~df_LOC_1hdi.lach_inter]

#[0] IMPORT LOCATION DATASET AND ROUND IT UP TO 1HOUR RESOLUTION



#Saving the repository
FOLD_SAVE =  DICT_paths['TMP'] + 'f01_009_D1_gap_sequence_sample/'

#[1] Sliding-window hour record counting - window of the epidemic period

SW_width_days = 28 

print(SW_width_days)

N_tot_hours = 24*SW_width_days

df_swc = sw_count_hour_records(df_LOC_1hdi, 
                               SW_width_days = SW_width_days, 
                               SW_step_days  = 1)
print(df_swc.index)
#df_swc = df_swc[(df_swc.index >= Study_period[0]) & (df_swc.index <= Study_period[-1])]

#CONDITION FOR STARTING DAY OF SLIDING WINDOW - [MONDAY OR TUESDAY]
if COND_START_DAY_01:
    COND_filter = pd.to_datetime(df_swc.index, errors='coerce').weekday
    COND_filter = COND_filter.isin([0,1])
    df_swc = df_swc.loc[COND_filter]

#SAVING THE SLIDING WINDOW COUNTS
swc_title = f'df_swhourcount_epidwindowdays_{SW_width_days}' 
if COND_START_DAY_01:
    swc_title += '_startday01'
if BOOL_tolerate_stops:
    swc_title += '_withtolstops'
df_swc.to_csv(f'{FOLD_SAVE}{swc_title}.csv')

for epsilon in [0, 0.05, 0.1]:
    
    print(epsilon)
    
    #[2] Assessment of complete users with 5% tolerance on missing records
    N_tolerated_hours = int(epsilon*N_tot_hours)
    Th_rec = N_tot_hours - N_tolerated_hours
    Date_start = df_swc.index[np.argmax((df_swc>= Th_rec).sum(axis=1))]    
    Date_end = Date_start + dt.timedelta(hours = N_tot_hours)
    USERS_select = df_swc.columns[df_swc.loc[Date_start] >= Th_rec].values
    Date_range = (Date_start, Date_end)
    
    #[3.a] Define temporal intervals for sparse trajectory sampling
    #week-resolution step for sparse trajectory sampling 
    N_days = 7
    DRSP_before  = pd.date_range(Date_range[1], Study_period[0], freq = f'-{N_days}D')[::-1]
    DRSP_after   = pd.date_range(Date_range[0], Study_period[1], freq = f'{N_days}D')
    List_DRSP = np.union1d(DRSP_before, DRSP_after)
    df_DRSP   = pd.DataFrame(List_DRSP, columns = ['start'])
    df_DRSP['end'] = df_DRSP['start'] + dt.timedelta(days = SW_width_days)
    df_DRSP = df_DRSP[df_DRSP['end'] <= Study_period[1]]
    
    df_DRSP.loc[df_DRSP['start']==Date_range[0], 'weekstep_index'] = 0
    
    def dr_indexing(df):
        reference_index = df[df[  'weekstep_index'].notna()].index[0]
        df.loc[:reference_index,  'weekstep_index'] = range(0, len(df.loc[:reference_index]))[::-1]
        df.loc[:reference_index,  'weekstep_index'] *= -1 
        df.loc[reference_index+1:,'weekstep_index'] = range(1, len(df) - reference_index)
        
    dr_indexing(df_DRSP)
    
    df_DRSP = df_DRSP.reset_index()
    f0 = lambda x : pd.date_range(x['start'], x['end'] - dt.timedelta(minutes =1), freq='h') 
    f1 = lambda x : pd.date_range(Date_range[0], Date_range[1] - dt.timedelta(minutes =1), freq='h') 
    df_DRSP['datetime'] = df_DRSP.apply(f0, axis=1)
    df_DRSP['datetime_dr'] =  df_DRSP.apply(f1, axis=1)
    df_exploded = df_DRSP.apply(lambda row: pd.DataFrame({'datetime': row['datetime'], 
                                                          'datetime_dr': row['datetime_dr'],
                                                          'weekstep_index': row['weekstep_index']}), axis=1)
    df_DRSP = pd.concat(df_exploded.values, keys=df_DRSP.index).reset_index(level=1, drop=True).reset_index()
    
    #[3.b] Hour-indicator sequence sampling over all the user collection
    path_LOC = DICT_paths['TMP'] + 'f_005_D1_filtering/filtered_traj_1min_downsample/'
    USERS = [u.split('.csv')[0] for u in os.listdir(path_LOC)]
    USERS_sparse = np.setdiff1d(USERS, USERS_select)
    df_USERS = pd.DataFrame(USERS, columns = ['id'])
    df_USERS['sparse'] = ~(df_USERS['id'].astype(int).isin(USERS_select))
    print(df_USERS.columns)
    df_MASKS = df_USERS.groupby(['id','sparse']).apply(lambda x : create_mask_single_user(x.name[0], 
                                                                                          Study_period, 
                                                                                          df_DRSP, 
                                                                                          path_LOC))
                                       
    MASK_title = f'epidwindowdays_{SW_width_days}_epsilon_{epsilon}_hourstol_{N_tolerated_hours}'

    if COND_START_DAY_01:
        MASK_title += '_startday01'
        
    if BOOL_tolerate_stops:
        MASK_title += '_withtolstops'
    
    df_MASKS.to_csv(f'{FOLD_SAVE}df_gapseq_{MASK_title}.csv')

