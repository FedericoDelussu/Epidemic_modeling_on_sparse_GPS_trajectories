from __future__ import annotations

import os 
import pandas as pd
import numpy as np
import pickle
import libgeohash as gh
import pygeohash as pgh
import geopandas as gpd
from scipy.spatial.distance import pdist, cdist
import math
import datetime as dt 
import matplotlib.pyplot as plt
import itertools
import optuna
import networkx as nx

from collections import Counter
from collections import OrderedDict

from . import config

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress detailed logs

DICT_paths = {'TMP': '/home/fedde/work/Project_Penn/TMP/', 
              'Data': '/home/fedde/work/Project_Penn/Data/'}


class KIT_conversion_dates:
    """Utilities to handle date-range arrays"""

    #hourofday-weekperiod coverage
    def get_hourofday_weekperiod(v):
        '''
        v : date-range series (will be converted to pandas datetime)
        gen hourofday_weekperiod column
        '''
        v = pd.to_datetime(v)
        _wdays = [0,1,2,3,4,5]
        _wperiod = np.array(['weekend','weekday'])
        
        return [f"{t}_{_wperiod[w*1]}" for t,w in zip(v.hour, v.weekday.isin(_wdays))] 

    def split_df_hourofday_weekperiod(df, 
                                      column = 'hourofday_weekperiod'):
        '''
        split the column in 2 cols
        '''
        #split into two columns
        df[['hourofday', 'weekperiod']] = df['hourofday_weekperiod'].str.split('_', expand=True)
        #cast hourofday to integer
        df['hourofday'] = df['hourofday'].astype(int)
        
    def convert_daterange_from_weekday(Date_range):
        '''
        given data-range let the first entry start from a weekday
        '''
        idx = 0
        while idx < len(Date_range) and Date_range[idx].weekday() >= 5:  # Skip weekends
            idx += 1
        return Date_range[(idx):]


class KIT_conversion_coordinates:
    """Utilities to make coordinate conversions"""
    
    def ConvertCoordinate_4326(lat, lon):
        '''
        function for conversion from crs:3857 to crs:4326
        '''
        
        latInEPSG4326 = (180 / math.pi) * (2 * math.atan(math.exp(lat * math.pi / 20037508.34)) - (math.pi / 2))
        lonInEPSG4326 = lon / (20037508.34 / 180)
        
        return latInEPSG4326, lonInEPSG4326

    def ConvertCoordinate_3587(lat, lon):
        '''
        conversion from crs:4326 to crs:3587
        '''
        
        latInEPSG3857 = (math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)) * (20037508.34 / 180)
        lonInEPSG3857 = (lon* 20037508.34 / 180)
        
        return latInEPSG3857, lonInEPSG3857
 
    def convert_df_coord_4326(df_loc, lat = 'lat', lon = 'lon'):
        '''
        convert df coordinate from crs:3857 to crs:4326    
        '''
        df_loc[[lat, lon]] = df_loc.apply(lambda row: ConvertCoordinate_4326(row[lat], row[lon]), axis=1, result_type='expand')
        
    def convert_df_coord_3587(df_loc, lat= 'lat', lon = 'lon'): 
        '''
        convert df coordinate from crs:4326 to crs:3587    
        '''
        df_loc[[lat, lon]] = df_loc.apply(lambda row: ConvertCoordinate_3587(row[lat], row[lon]), axis=1, result_type='expand')



#############################################
###### COORDINATE CONVERSION FUNCTIONS ######
#############################################

def get_table_count(df, x, y):
    return df.groupby([x,y]).size().reset_index().pivot(index = x, columns = y, values = 0).fillna(0)

def subset_df_feature(df,f):
    '''
    df : dataframe
    f  : feature name 
    '''
    #feature unique values 
    f_vals = df[f].unique()
    #subset dataframe according to feature f records 
    return {v: df[df[f]==v] for v in f_vals}

def stack_dict_to_df(DICT_contacts, f_name = 'level'):
    rows = []
    for key, df in DICT_contacts.items():
        df = df.copy()
        df[f_name] = str(key)
        rows.append(df)
    stacked_df = pd.concat(rows, axis=0)#, ignore_index=True)
    return stacked_df

def convert_daterange_from_weekday(Date_range):
    idx = 0
    while idx < len(Date_range) and Date_range[idx].weekday() >= 5:  # Skip weekends
        idx += 1
    return Date_range[(idx):]

def ConvertCoordinate_4326(lat, lon):
    '''
    function for conversion from crs:3857 to crs:4326
    '''
    
    latInEPSG4326 = (180 / math.pi) * (2 * math.atan(math.exp(lat * math.pi / 20037508.34)) - (math.pi / 2))
    lonInEPSG4326 = lon / (20037508.34 / 180)
    
    return latInEPSG4326, lonInEPSG4326

def convert_df_coord_4326(df_loc, lat = 'lat', lon = 'lon'):
    '''
    convert df coordinate from crs:3857 to crs:4326    
    '''
    df_loc[[lat, lon]] = df_loc.apply(lambda row: ConvertCoordinate_4326(row[lat], row[lon]), axis=1, result_type='expand')

def ConvertCoordinate_3587(lat, lon):
    '''
    conversion from crs:4326 to crs:3587
    '''
    
    latInEPSG3857 = (math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)) * (20037508.34 / 180)
    lonInEPSG3857 = (lon* 20037508.34 / 180)
    
    return latInEPSG3857, lonInEPSG3857
    
def convert_df_coord_3587(df_loc, lat= 'lat', lon = 'lon'): 
    '''
    convert df coordinate from crs:4326 to crs:3587    
    '''
    df_loc[[lat, lon]] = df_loc.apply(lambda row: ConvertCoordinate_3587(row[lat], row[lon]), axis=1, result_type='expand')



##########################################
######## STOP DETECTION FUNCTIONS ########
##########################################

def diameter(coords, metric='euclidean'):
    if len(coords)<2:
        return 0
    if metric == 'haversine':
        coords = np.radians(coords)
        earth_radius_meters = 6371000  # Earth's radius in meters
        return np.max(pdist(coords, metric=metric)) * earth_radius_meters
    return np.max(pdist(coords, metric=metric))

def medoid(coords, metric='euclidean'):
    if len(coords)<2:
        return coords[0]
    coords_radians = np.radians(coords) if metric == 'haversine' else coords
    distances = cdist(coords_radians, coords_radians, metric=metric)
    sum_distances = np.sum(distances, axis=1)
    medoid_index = np.argmin(sum_distances)
    return coords[medoid_index, :]

def update_diameter(c_j, coords_prev, D_prev): 
    '''
    c_j: new point's coordinate
    coords_prev : set of previous coordinates
    D_prev : previous computed diameter
    '''
    
    X_prev = coords_prev[:, 0]
    Y_prev = coords_prev[:, 1]
            
    x_j,y_j = c_j[0], c_j[1]
    
    #set of distances of new coordinate for other points 
    new_dists =  np.sqrt( (X_prev - x_j)**2 + (Y_prev - y_j)**2 )
    
    #compute new diameter
    D_i_jp1 = np.max([D_prev, np.max(new_dists)])

    return D_i_jp1

def lachesis(traj, 
             dur_min, 
             dt_max, 
             delta_roam):
    """
    Extract stays from raw location data.
    Parameters
    ----------
    traj: numpy array - simulated trajectory from simulate_traj.
        - (lat,lon) : 'y', 'x' in crs:3586
        - time is 'unix_timestamp' in 'timestamp' format
    dur_min [minutes]   : float - minimum duration for a stay (stay duration).
    dt_max  [minutes]   : float - maximum duration permitted between consecutive pings in a stay. dt_max should be greater than dur_min 
    delta_roam [meters] : float - maximum roaming distance for a stay (roaming distance).

    Returns
    """
    
    coords = traj[['x', 'y']].to_numpy()
    Stays = np.empty((0,6))

    #[STEP0] starting the search
    i = 0
    
    while i < len(traj)-1:
        
        #[STEP1] - find the least amount of pings over a timerange > dur_min
        timestamp_i = traj['unix_timestamp'].iat[i]
        j_star = next((j for j in range(i, len(traj)) if traj['unix_timestamp'].iat[j] - timestamp_i >= dur_min * 60), -1)
        
        #conditions to block the j_star search
        Cond_exhausted_traj = j_star==-1 
        #diameter over the delta_roam threshold
        D_start = diameter(coords[i:j_star+1])
        Cond_diam_OT = D_start > delta_roam 
        #condition that there is at least consecutive ping pair that has a time-separation greater than dt_max
        Cond_cc_diff_OT = (traj['unix_timestamp'][i:j_star+1].diff().dropna() >= dt_max*60).any()

        #[STEP2] - decide whether index i is a 'stop' or 'trip' ping 
        if Cond_exhausted_traj or Cond_diam_OT or Cond_cc_diff_OT:
            #DISCARD i as a candidate 'stop start' ping and move forward
            i += 1
        else:
            #SELECT i as a 'stop start' ping AND 
            #[STEP3] proceed with the iterative search of j_star
            COND_found_jstar = False
            for j in range(j_star, len(traj)-1): 
                #update diameter
                D_update = update_diameter(coords[j], coords[i:j], D_start)
                #compute the conescutive ping's time difference
                cc_diff = traj['unix_timestamp'].iat[j] - traj['unix_timestamp'].iat[j-1]
                #verify that the new ping does not break the rule
                COND_j = (D_update > delta_roam) or (cc_diff > dt_max*60)
                if COND_j:
                    #new ping broke the rule and the stop detection is completed
                    j_star = j-1 
                    COND_found_jstar = True
                    break
                else:
                    #the stop detection proceeds - update the diameter
                    D_start = D_update
            #handle the case in which no further pings 
            if not COND_found_jstar:
                j_star = len(traj) - 1    

            #COLLECT STOP INFORMATION
            #stay_medoid = medoid(coords[i:j_star+1])
            start,end = traj['unix_timestamp'].iat[i], traj['unix_timestamp'].iat[j_star]
            stay_medoid = medoid(coords[i:j_star+1])
            n_pings = j_star-i+1
            stay  = np.array([[start, end, stay_medoid[0], stay_medoid[1],D_start, n_pings]])

            #UPDATE THE STAY DATAFRAME
            Stays = np.concatenate((Stays, stay), axis=0)

            #updating start index of the stop
            #proceed the search
            i = j_star + 1
  
    Cols_ = ['start_time', 'end_time', 'medoid_x', 'medoid_y', 'diameter_m', 'n_pings'] 
    stays = pd.DataFrame(Stays, columns = Cols_)
    #compute stop duration
    stays['duration_s'] = stays['end_time'] - stays['start_time']
    
    #columns to convert to datetime
    cc_dt = ['start_time','end_time'] 
    stays[cc_dt] = stays[cc_dt].apply(lambda x : pd.to_datetime(x, unit= 's'))

    #convert medoids and compute geohashes 
    convert_df_coord_4326(stays, lat = 'medoid_y', lon = 'medoid_x')
    stays['geohash9'] = stays.apply( lambda row : pgh.encode(row.medoid_y, row.medoid_x, 9), axis=1)

    return stays

#########################################
### COMPLETE USER SELECTION FUNCTIONS ###
#########################################

def gen_daterange(row, 
                  dt_start = 'start_time', 
                  dt_end   = 'end_time', 
                  only_borders= False):
        '''
        gen daterange with hour frequency keeping same borders with higher temporal resolution
        '''

        if only_borders:
            return pd.DatetimeIndex(row[[dt_start, dt_end]])    

        #create the stops daterange with 1hour resolution
        daterange = pd.date_range(start=row[dt_start], end=row[dt_end], freq='h')
        if len(daterange) >2:
            new_daterange = pd.DatetimeIndex([row[dt_start]]).append(daterange[1:-1]).append(pd.DatetimeIndex([row[dt_end]]))
            return new_daterange
        else:
            return pd.DatetimeIndex(row[[dt_start, dt_end]])     


def interpolate_loc_1hd(df_LOC, df_stops): 
    '''
    interpolate location dataset at 1hour downsample resolution with the stop dataset
    '''
    #1-hour downsampled location data
    df_LOC_1hd = df_LOC[['id','unix_timestamp']].copy()
    df_LOC_1hd['unix_timestamp'] = pd.to_datetime(df_LOC_1hd['unix_timestamp'], unit='s').dt.floor('h')
    df_LOC_1hd = df_LOC_1hd.drop_duplicates()
    df_LOC_1hd.rename(columns = {'unix_timestamp':'date_hour'}, inplace = True)
    df_LOC_1hd['lach_inter'] = False 
    
    #lachesis stop exploded dataset 
    df_se = df_stops.reset_index()[['id','level_1', 'start_time','end_time']]
    df_se.loc[df_se.index,'date_hour'] = df_se.apply(lambda row: gen_daterange(row), axis=1)
    df_se = df_se[['id','date_hour']].explode('date_hour')
    df_se['date_hour'] = df_se['date_hour'].dt.floor('h')
    df_se = df_se.drop_duplicates()
    df_se['lach_inter'] = True
    
    #1hour downsampled interpolated dataset
    df_LOC_1hd =  pd.concat([df_LOC_1hd, df_se], axis= 0).drop_duplicates(subset = ['id','date_hour'], 
                                                                          keep = 'first')
    return df_LOC_1hd


def clip_midnight(df_brt_n):  
    
    # Ensure that the 'date_hour' column is in datetime format
    df_brt_n.index = pd.to_datetime(df_brt_n.index)
    
    # Define the start and end of the desired period (from midnight to midnight)
    start = df_brt_n[df_brt_n.index.time == pd.Timestamp('00:00:00').time()].index.min()
    end = df_brt_n[df_brt_n.index.time == pd.Timestamp('23:00:00').time()].index.max()
    
    # Filter the DataFrame to include only rows from the first to the last midnight
    df_clipped = df_brt_n.loc[start:end]

    return df_clipped


def sliding_window_hour_res(df_ts_bool_, 
                            SW_width_days, 
                            SW_step_days):
    '''
    df_ts_bool: table of boolean record presence
    SW_width_days : width of the sliding window
    SW_step_days  : step of the sliding window 
    '''

    #conversion to hours 
    SW_width_hours = SW_width_days*24
    SW_step_hours = SW_step_days*24

    df_ts_bool = df_ts_bool_.copy()
    df_ts_bool.index = range(len(df_ts_bool))
    
    #range of indexes for the sliding window
    #step of 1day 
    
    Indexes = df_ts_bool.index[:-(SW_width_hours-1):(SW_step_hours)]
    
    df_window_counts = []
    
    for I_win in Indexes: 
        
        #counting number of records over the sliding window
        window_counts = df_ts_bool.loc[I_win: I_win + SW_width_hours -1]#
        window_counts = window_counts.sum(axis=0).values
        df_window_counts.append(window_counts)

    DT_index = df_ts_bool_.index[:-(SW_width_hours-1):(SW_step_hours)]
    Users = df_ts_bool.columns 
    
    df_window_counts = pd.DataFrame(df_window_counts, index = DT_index, columns = Users)
    
    return df_window_counts
    
def sw_count(df_brt_n, W, S): 
    
    #clip to first and last midnight
    df_v1 = clip_midnight(df_brt_n).copy()
    #set index to date 
    df_v1['date'] = df_v1.index.date
    
    Users = df_brt_n.columns
    
    date_counts = df_v1.groupby('date').size()
    date_no24 = date_counts[date_counts!=24]
    if len(date_no24)>0:
        print('exception, some dates have less records than 24')
    
    #sliding window count
    df_swc = sliding_window_hour_res(df_v1[Users], W, S)

    return df_swc


def sw_count_hour_records(df_lh, 
                          SW_width_days, 
                          SW_step_days = 1):
    '''
    df_lh : location-hour dataset (should be interpolated by lachesis)
    SW_width_days : sliding window width 
    SW_step_days  : sliding window step
    plot_sw_count : if True, plots the time-series of sliding window counts 
    tolerance : number of tolerated missing hours within the sliding window
    '''

    df_tbh = df_lh.pivot(index = 'date_hour', 
                         columns = 'id', 
                         values = 'lach_inter')
    
    df_rb = (~df_tbh.isna())*1
    
    N_tot_hours = SW_width_days*24
    
    #sliding window counting
    df_swc = sw_count(df_rb, SW_width_days, SW_step_days)
    
    return df_swc
    
def plot_complete_users(ax, 
                        df_swc, 
                        Th_rec):
    '''
    df_swc : slinding window counting dataframe
    Th_rec : threshold on number of records for considering the user complete
    '''
    #boolean over thresold
    df_cot = df_swc >= Th_rec
    N_us = df_cot.sum(axis=1)
    X,Y = N_us.index, N_us.values
    ax.plot(X,Y)

#########################################
###### SPARSE TRAJECTORY SAMPLING  ######
#########################################

def create_mask_single_user(U0, 
                            Study_period, 
                            df_DRSP, 
                            path):
    '''
    U0 : user-id (str)
    Study_period : datetime tuple, default is from 2014-2 to 2015-2
    df_DRSP : temporal-intervals for sparse trajectory sampling
    path : file-path storing the raw individual location data
    
    create hourly mask for a single user 
    over consecutive intervals within the study-period
    the intervals have the same length of the epidemiologic modeling date-range
    the interval shift is by 1week
    '''
    
    Cols_select = ['id','datetime']
    df_U0 = pd.read_csv(os.path.join(path, f'{U0}.csv'), 
                                     usecols=Cols_select, 
                                     index_col=None, 
                                     parse_dates=['datetime'], 
                                     dtype = None)

    #downsample the records to hour resolution and drop duplicates 
    df_U0 = df_U0[df_U0.datetime.between(Study_period[0], Study_period[1])]
    df_U0['datetime'] = df_U0['datetime'].dt.floor('h')
    df_U0.drop_duplicates(inplace = True)

    #merge with the Studyperiod ranges
    df_U0_ext = pd.merge(df_DRSP, 
                         df_U0, 
                         on  = ['datetime'], 
                         how = 'left')

    #remove week-indexes with missing records 
    valid_indices = df_U0_ext.groupby('weekstep_index')['id'].transform(lambda x: x.notna().any())
    df_U0_ext = df_U0_ext[valid_indices]
    if df_U0_ext.empty:
        return 

    df_U0_ext = df_U0_ext.pivot(index= 'weekstep_index', columns = 'datetime_dr', values = 'id')#.fillna(0)
    df_U0_ext = 1*(~df_U0_ext.isna())
    
    return df_U0_ext


########################################
##### CONTACT ESTIMATION FUNCTIONS #####
########################################

def filter_stops(df_stops, 
                 USERS_select, 
                 Date_range, 
                 Cols_select = None, 
                 reset_ghr = None):
    '''
    filter stop-table after complete user selection
    if reset_ghr : proccesses geohash column at the required resolution
    '''
    
    if Cols_select is None:
        Cols_select = df_stops.columns
        
    #user selection
    df_stops_US = df_stops[Cols_select].loc[df_stops.id.isin(USERS_select)]
    #daterange selection
    cs,ce = df_stops_US['start_time'] >= Date_range[0],  df_stops_US['end_time'] < Date_range[1] 
    df_stops_DR = df_stops_US[ cs & ce]
    
    #CORRECT FOR BORDER EFFECTS - INCLUDE STOPS AT THE BORDER
    #stopborder - START
    df_sb0 = df_stops_US[ (df_stops_US['start_time'] < Date_range[0]) & (df_stops_US['end_time'] > Date_range[0]) ].copy()
    #stopborder - END
    df_sb1 = df_stops_US[ (df_stops_US['start_time'] < (Date_range[1] - dt.timedelta(minutes=1)) ) & (df_stops_US['end_time'] > Date_range[1]) ].copy()
    #clip the stops so that they fit within the border
    df_sb0.loc[:,'start_time'] = pd.to_datetime(Date_range[0])
    df_sb0.loc[ df_sb0['end_time'] >= Date_range[1], 'end_time']   = pd.to_datetime(Date_range[1] - dt.timedelta(minutes=1))

    df_sb1.loc[(df_sb1['start_time']< Date_range[0]), 'start_time'] = pd.to_datetime(Date_range[0])
    df_sb1.loc[:,'end_time']   = pd.to_datetime(Date_range[1] - dt.timedelta(minutes=1))
    
    df_stops_DR = pd.concat([df_sb0, df_stops_DR, df_sb1], axis = 0)
    df_stops_DR = df_stops_DR.drop_duplicates()

    if reset_ghr is not None:
        df_stops_DR = df_stops_DR.rename(columns = {'geohash9': 'geohash'})
        df_stops_DR['geohash'] = df_stops_DR['geohash'].str[:reset_ghr]

    return df_stops_DR

def get_stops_CONTACT(df_stops_DR, geohash = 'geohash'):
    '''
    df_stops_DR : exploded stop table 
    returns simple stop table contributing to contacts
    '''

    #count the number of occurring stops for each hour and geohash
    df_stops_DR['count_nstops'] = df_stops_DR.groupby([geohash, 'stop_hour'])['unique_stop_id'].transform('size')
    df_stops_CONTACT = df_stops_DR.drop_duplicates(subset=['unique_stop_id', 'count_nstops']).copy()
    df_stops_DR.drop('count_nstops', axis=1, inplace = True)
    
    #INDICATOR for which (geohash,hour) count is always 1 over all the stop_id occurrences
    df_stops_CONTACT['ID_count_nstops_1'] = df_stops_CONTACT.groupby('unique_stop_id')['count_nstops'].transform(lambda x: (x == 1).all())
    
    #STOPS CONTRIBUTING TO CONTACTS
    df_stops_CONTACT = df_stops_CONTACT[~df_stops_CONTACT['ID_count_nstops_1']] 
    df_stops_CONTACT = df_stops_CONTACT.drop(['stop_hour', 'count_nstops', 'ID_count_nstops_1'], axis=1)
    df_stops_CONTACT = df_stops_CONTACT.drop_duplicates(subset = ['unique_stop_id'])
    
    return df_stops_CONTACT 

#functions for contact estimation
def interp_boolean(lst):
    '''
    given boolean series,
    set to 1 the 0s between odd-occurring and even-occurring 1s
    '''
    ones_indices = np.where(lst == 1)[0]
    idx_odd, idx_even = ones_indices[::2], ones_indices[1::2]

    #create indexes for interpolation
    lengths = idx_even - idx_odd + 1
    ranges = np.add.outer(np.arange(max(lengths)), idx_odd) 
    idx_inter = ranges[ranges <= idx_even[np.newaxis,:]]

    if isinstance(lst, pd.Series):
        lst.iloc[idx_inter] = 1
    else:
        lst[idx_inter] = 1
        
    return lst


def estimate_contacts_hourly(df_gh, time_resolution = '1hour'):#, df_stop_borders= None):
    '''
    def compute_contact_table(df_stops, geohash_col = 'geohash'):
    df_ch = df_stops_DR.groupby('geohash').apply(estimate_contacts_hourly)
    df_ch = df_ch.reset_index()[['geohash','date_hour','couples', 'n_minutes']]
    df_ch[['u1', 'u2']] = pd.DataFrame(df_ch['couples'].tolist(), index=df_ch.index)
    df_ch.drop('couples', axis=1, inplace= True)
    return df_ch

    estimates hourly contacts counting the number of minutes in contact
    - df_gh: stop-table
    - index_1hd : time-range over which contact estimation is performed
    - df_stop_borders: additional stop indicators belonging to the border
    '''

    df_gh.loc[:,'set'] = df_gh.apply(lambda row: [row['start_time'], row['end_time']], axis=1)
    df_ghe = df_gh[['id','set']].explode('set')
    df_ghe['val']= 1
    
    #create indicator table of stop border
    df_th = df_ghe.pivot(index = 'set', columns = 'id', values = 'val')
    
    #reindex the table at the minute resolution
    index_1hd = pd.date_range(start = df_th.index.min().floor('h'), end = df_th.index.max().floor('h'), freq= 'min')
    df_th = df_th.reindex(index_1hd.union(df_th.index))
    df_th = df_th.apply(interp_boolean, axis=0)

    #select only datetime records with more than 1 user 
    df_th = df_th[df_th.sum(axis=1)>=2]
    
    #create a couple dataframe
    df_th['couples'] = df_th.apply(lambda x : list(itertools.combinations(x.index[x == 1].tolist(), 2)), axis=1)
    #explode the couples
    df_thc = df_th[['couples']].explode('couples')

    if time_resolution == '1hour':
        #assess boolean contact at the hourly level
        df_thc = df_thc.reset_index().rename(columns = {'index':'date_hour'})#, inplace = True) 
        df_thc['date_hour'] = df_thc['date_hour'].dt.floor('h')
        df_thc = df_thc.groupby(['date_hour', 'couples']).size().reset_index().rename(columns = {0:'n_minutes'})  
        
    if time_resolution == '1minute':
        df_thc =  df_thc.reset_index().rename(columns = {'index':'date_time'})

    return df_thc

def compute_contact_table(df_stops, geohash_col = 'geohash', time_resolution = '1hour'):
    
    df_ch = df_stops.groupby(geohash_col).apply(lambda x: estimate_contacts_hourly(x, time_resolution))
    
    if time_resolution == '1hour':
        df_ch = df_ch.reset_index()[[geohash_col,'date_hour','couples', 'n_minutes']]
        df_ch[['u1', 'u2']] = pd.DataFrame(df_ch['couples'].tolist(), index=df_ch.index)
        df_ch.drop('couples', axis=1, inplace= True)

    if time_resolution == '1minute':
        df_ch[['u1', 'u2']] = pd.DataFrame(df_ch['couples'].tolist(), index=df_ch.index)
        df_ch.drop('couples', axis=1, inplace= True)
        df_ch = df_ch.reset_index().drop('level_1', axis=1)#.set_index(geohash_col)
        #df_ch.index.name = None     
        
    return df_ch

#marginal contact estimation

def get_gh_pop_neighbour(df_gph0):
    '''
    groupby function at hour level 
    selects only geohashes which have at least one populated neighbouring geohash
    '''
    
    #keep only the geohash which have neighbouring populated records
    df_gph0['gh_ns'] = df_gph0.apply(lambda x :gh.neighbors(x['geohash']).values(), axis = 1)
    df_gph0 = df_gph0.explode('gh_ns')
    df_gph0['pair_set'] = df_gph0.apply(lambda row: set([row['geohash'], row['gh_ns']]), axis=1)
    df_filtered = df_gph0[df_gph0.duplicated('pair_set', keep=False)]
    
    #[0] check correct function working
    df_filtered = df_filtered.drop(columns='pair_set')#[['geohash']]#.values
    df_filtered = df_filtered.reset_index(drop=True)['geohash'].unique()

    #CHECK = (np.sort(df_h0_un['geohash'].unique()) == np.sort(df_h0_un['gh_ns'].unique())).all()

    #[1] check correct function working (the neighbourhood pairs are always 2 for each hour)
    #on a single hour dataframe
    #Hours = df_SDR['stop_time'].unique()
    #df_h0 = df_SDR[df_SDR['stop_time'] ==Hours[0]]
    #df_h0_un = keep_gh_pop_neighbour(df_h0)
    #df_h0_un['pair_set_str'] = df_h0_un['pair_set'].apply(lambda x: str(sorted(x)) )
    #df_h0_un.sort_values(by = 'pair_set_str')

    return df_filtered

def get_stops_NEIGHBOURS(df_stops_DR): 
    '''
    df_stops_DR : exploded stop table 
    returns simple stop table contributing to contacts
    '''

    #get the neighbouring geohashes for each hour
    df_stops_DR_gh_neigh = df_stops_DR[['stop_hour', 'geohash']].groupby('stop_hour').apply(get_gh_pop_neighbour)
    df_stops_DR_gh_neigh = df_stops_DR_gh_neigh.explode(0)
    df_stops_DR_gh_neigh = df_stops_DR_gh_neigh.reset_index().rename(columns={0: 'geohash'})
    
    #INPUT FOR MARGINAL CONTACT ESTIMATION
    #dataframe which includes stops that included at least 1 hour in a neighbouring geohash
    df_stops_NEIGHBOURS = df_stops_DR.merge(df_stops_DR_gh_neigh[['stop_hour', 'geohash']],
                                            on = ['stop_hour', 'geohash'],
                                            how = 'inner')
    
    #df_stops_NEIGHBOURS = df_stops_NEIGHBOURS.drop_duplicates(subset=['unique_stop_id'])
    
    return df_stops_NEIGHBOURS


#COMPUTE COLLECTION GEOHASH WIDTHS
def get_gh_widths(df_gh_input): 
    '''
    compute the widths of a set of geohashes of dataframe df_gh_input
    '''
    df_gh = pd.DataFrame(df_gh_input['geohash'].unique(), columns=['geohash']) 
    df_gh['geometry'] = df_gh.apply(lambda x : gh.geohash_to_polygon([x['geohash']]), axis=1)
    df_gh = gpd.GeoDataFrame(df_gh, 
                             crs = 'EPSG:4326', 
                             geometry = 'geometry')
    
    df_gh[['minx', 'miny', 'maxx', 'maxy']] = df_gh.apply(lambda x : x['geometry'].bounds, axis=1, result_type = 'expand') 
    
    Grid_lines = {'Xs': np.unique(df_gh[['minx','maxx']].values.ravel()), 
                  'Ys': np.unique(df_gh[['miny','maxy']].values.ravel())}
    
    df_gh['width_x'] = df_gh['maxx'] - df_gh['minx']
    df_gh['width_y'] = df_gh['maxy'] - df_gh['miny']
    df_gh = df_gh[['width_x','width_y']].drop_duplicates()

    if len(df_gh)>1: 
        #check if geohash have different widths
        print('WARNING: geohash set has multiple widths')
        
    return df_gh

def geohash_shift(df_ps, Shift, wx, wy, perc_width = 0.5, ghr = 7, new_gh_col = 'geohash_shift'): 
    '''
    df_ps: set of points to be shifted
    Shift : unit vector indicating the shift direction over [x,y] axis
    wx,wy : widths of the geohash grid-cells over [x,y] direction
    perc_width : scale of the step; by default is half steps
    
    shift points and reassign geohash value
    '''
    
    df_ps['medoid_x'] += perc_width*wx*Shift[0]
    df_ps['medoid_y'] += perc_width*wy*Shift[1]
    df_ps[new_gh_col] = df_ps.apply( lambda row : pgh.encode(row.medoid_y, row.medoid_x, ghr), axis=1)

def join_original_geohashes(df_contacts_shift, 
                            df_stops_NEIGHBOUR_1min, 
                            geohash_col = 'geohash'):
    '''
    given contact table add info on originary geohash of each id at 1minute resolution
    '''

    Cols_select= ['id', 'stop_minute',geohash_col]

    for u in ['u1','u2']: 
        df_contacts_shift = df_contacts_shift.merge(
            df_stops_NEIGHBOUR_1min[Cols_select],
            how='left',
            left_on=[u, 'date_time'],
            right_on=['id', 'stop_minute']
        ).rename(columns={geohash_col: f'{u}_geohash'}).drop(columns=['id', 'stop_minute'])
    
    return df_contacts_shift 

def compute_tot_contacts(df_cm,df_cw):
    '''
    df_cw : within-cell contacts
    df_cm : marginal contacts
    '''
    Cs = ['u1','u2', 'date_hour', 'n_minutes']
    df_ctot = pd.concat([df_cm[Cs], df_cw[Cs]], axis=0)
    df_ctot = df_ctot.groupby(['u1','u2','date_hour'])['n_minutes'].sum().reset_index()
    
    return df_ctot

##############################################
######### SPARSITY FUNCTIONS ###########
##############################################

def get_user_sparse_stats(df_LOC_, 
                          t  = 'timestamp', 
                          id = 'id',
                          Date_window=None): 
    '''
    compute sparse statistics
    '''
    
    df_LOC = df_LOC_.copy()
    if Date_window is not None: 
        Date_window_utc = (
            int(pd.Timestamp(Date_window[0]).timestamp()),
            int(pd.Timestamp(Date_window[1]).timestamp()))
        utc_start, utc_end = Date_window_utc
        df_LOC = df_LOC[df_LOC[t].between(utc_start, utc_end, inclusive='left')]
        
    df_uid_lifespan = df_LOC.groupby(id).apply(lambda x: x[[t]].describe())
    df_uid_lifespan = df_uid_lifespan.reset_index()
    Cols_select = ['min', 'max', 'count']
    df_uid_lifespan = df_uid_lifespan.pivot(index=id, 
                                            columns='level_1', 
                                            values=t)[Cols_select]
    
    df_uid_lifespan.rename(columns={'count': 'n_records'}, inplace=True)
    df_uid_lifespan['min'] = pd.to_datetime(df_uid_lifespan['min'], unit='s')
    df_uid_lifespan['max'] = pd.to_datetime(df_uid_lifespan['max'], unit='s')
    if Date_window is not None:
        df_uid_lifespan['min'] = pd.to_datetime(Date_window[0])
        df_uid_lifespan['max'] = pd.to_datetime(Date_window[1])
    
    df_uid_lifespan['lifespan_days'] = (df_uid_lifespan['max'] - df_uid_lifespan['min']).dt.days
    df_uid_lifespan['lifespan_hours'] = df_uid_lifespan['lifespan_days'] * 24
    
    # record indicator dataframe
    df_rid = df_LOC[[id, t]].copy()
    df_rid['date_time'] = pd.to_datetime(df_rid[t], unit='s')
    df_rid['date_hour'] = df_rid['date_time'].dt.floor('h')
    unique_date_hours = df_rid.groupby(id)['date_hour'].nunique()
    df_uid_lifespan = df_uid_lifespan.merge(
        unique_date_hours.rename('n_hours'),
        left_index=True,
        right_index=True)
    
    N = df_uid_lifespan['n_hours']
    N_tot = df_uid_lifespan['lifespan_hours']
    df_uid_lifespan['perc_missing_hours'] = 100 * (N_tot - N) / N_tot
    
    return df_uid_lifespan

def gen_sparsity_metric_ranges(): 
    '''
    Set of sparsity metrics and corresponding ranges 
    employied for sparsity sequence sampling
    '''
    
    #SPARSITY METRICS AND STUDIED RANGES
    Metrics_ranges = {'perc-missing-hours' : [(0,0.1), (0.1,0.2), (0.2,0.3),(0.3,0.4), (0.4,0.5), (0.5,0.6)]}
    
    Ths = [2,3,6,10]
    Metrics_ranges.update({f'n-gaps-ot{th}hours': [(1,2),(3,4),(5,6),(7,10)] for th in Ths})

    #TODO 
    #pmh_gaps_over_threshold
    #pmh_gap_over_threhsold_in_nightime

    return Metrics_ranges 

def gen_gap_lims(Seq):
    '''
    create gap list from a sequence
    '''
    Ind_gaps = Seq.diff() == -1
    Ind_gaps_end = Seq.diff() == 1
    Date_start_gaps = list(Ind_gaps[Ind_gaps.values].index)  # Convert to list to allow insertions
    Date_end_gaps = list(Ind_gaps_end[Ind_gaps_end.values].index)  # Same for end gaps           
    if Seq.iloc[0] == 0:
        Date_start_gaps = [Seq.index[0]] + Date_start_gaps
    if Seq.iloc[-1] == 0:
        Date_end_gaps.append(Seq.index[-1])    
    return [ (ds,de) for ds,de in zip(Date_start_gaps, Date_end_gaps) ]


def compute_seq_sparsity(df_id_gaps, metric, N_hours, th = None):
    '''
    compute sparsity metric for each sequence 
    input sequence resolution is 1-hour
    '''
    if metric == 'perc-missing-hours':
        #[m1] percentage of missing records 
        S_missing_hours = df_id_gaps.groupby('sequence_index')['gap_duration_hours'].sum()
        S_pmh = S_missing_hours/(N_hours)
        
        return S_pmh
        
    if metric == 'n-gaps-ot':
        #[m2] number of gaps >= threshold
        invalid_row_indices = df_id_gaps[df_id_gaps['gap_duration_hours'] > 24]['sequence_index'].unique()
        df_igs = df_id_gaps[~df_id_gaps['sequence_index'].isin(invalid_row_indices)]
        df_igs_count = get_table_count(df_igs, 'sequence_index', 'gap_duration_hours')
        df_ngaps_over_th = df_igs_count.iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
        
        return df_ngaps_over_th[th]


def get_unselected_users(df_lh, USERS_select, Date_range):
    '''
    df_lh: LOC_1hdi 
    get unselected users with gaps at the hour resolution before Lachesis interpolation
    '''
    
    #BOOLEAN RECORD INDICATOR, AT ONE HOUR DOWNSAMPLED RESOLUTION
    
    #select only raw records
    df_lh_raw = df_lh[~df_lh['lach_inter']]

    #create raw table of boolean records
    df_tbr = df_lh_raw.pivot(index = 'id', 
                             columns = 'date_hour', 
                             values = 'lach_inter') 
    
    df_tbr = (~df_tbr.isna())*1

    #Unselected useres
    df_tbr_UNS = df_tbr.loc[~df_tbr.index.isin(USERS_select)]

    #filter over the date_range
    cols_select = df_tbr_UNS.columns[(df_tbr_UNS.columns>=Date_range[0]) & (df_tbr_UNS.columns<Date_range[1])]
    df_tud = df_tbr_UNS[cols_select]
    
    return df_tud

def sparse_sample_weigthed(user_gaps_perc, N_select, P_level):
    '''
    user_gaps_perc : individual series of sparsity level 
    impelement weighted sampling without replacement
    '''
    weights = 1 / (np.abs(user_gaps_perc - P_level) + 1e-6)
    weights /= weights.sum()
    
    selected_indices = np.random.choice(user_gaps_perc.index, 
                                        size=N_select, 
                                        replace=False, 
                                        p=weights)
    
    return selected_indices
    

def sparse_unif_sample(user_gaps_perc, 
                       N_select, 
                       P, 
                       P_hw = 0.1):
    '''
    define a subsample of unselected users 
    with sparsity belonging to the P centered interval with half width P_hw
    extract N_select from this subsample with replacement
    '''

    U_phw_ind = user_gaps_perc.between(P-P_hw, P+P_hw)
    U_phw = user_gaps_perc[U_phw_ind]
    
    selected_users = np.random.choice(U_phw.index, 
                                      size=N_select, 
                                      replace=True)
    
    return selected_users, len(U_phw)

def gen_gap_overlay_mask(P, df_ugh, N_select, P_hw = 0.1):
    
    #percentage of missing temporal records over the study period for each user 
    N_recs = df_ugh.shape[1]
    user_gaps_perc = (N_recs - df_ugh.sum(axis=1))/N_recs
    
    selected_users, sample_dim = sparse_unif_sample(user_gaps_perc, N_select, P, P_hw = P_hw)
    
    #select mask for sparsity P selection
    df_GP = df_ugh.loc[selected_users]
    df_GP.index.name = None

    return df_GP

def unpivot_mask(df_mask, v_name = 'datetime'):
    '''
    unpivot the gap mask
    useful for gap overlaying
    '''
    
    df_mask_reset = df_mask.reset_index()
    df_unpivoted = pd.melt(df_mask_reset, id_vars='index', var_name= v_name, value_name='value')
    df_unpivoted = df_unpivoted.rename(columns={'index': 'id'})
    df_unpivoted[v_name] = pd.to_datetime(df_unpivoted[v_name])
    df_unpivoted = df_unpivoted.sort_values(by = ['id', v_name])
    
    return df_unpivoted

def from_mask_to_record_indicator(df_mask, t_res = 'datetime'):
    '''
    converts a gap mask to a record indicator
    useful for gap overlaying
    '''
    
    df_mask_up = unpivot_mask(df_mask, v_name = t_res)
    df_record_select = df_mask_up[df_mask_up['value']==1].drop('value', axis=1)
    if t_res == 'date_hour':
        #extend the date_hour resolution
        f = lambda x : pd.date_range(x['date_hour'], x['date_hour'].replace(minute = 59), freq='min') 
        df_record_select['datetime'] = df_record_select.apply(f, axis =1)
        df_record_select = df_record_select[['id','datetime']].explode('datetime')
        
    return df_record_select


#######################################
######### EPIDEMIC MODELING ###########
#######################################

def convert_rate_from_second_to_minute(beta_sec, gamma_sec): 
    #COMPOUNDED PROBABILITIES - 1 minute resolution 
    beta_min = 1 - (1 - beta_sec)**60
    gamma_min = 1 - (1 - gamma_sec)**60
    pars = (beta_min, gamma_min) 
    return pars
    
def gen_dict_epid_pars(): 
    '''
    parameters for 2 epidemic scenarios
    from Colizza paper: 'Simulation of an SEIR infectious disease model 
    on the dynamic contact network of conference attendees'
    '''
    
    #SCENARIO1 - VERY SHORT incubation and infectious periods
    #beta  : probability of being infected in 1-second
    beta_sec = 3e-4            
    #gamma : inverse of recovery period in seconds 
    gamma_sec = 1/(24*3600)
    
    DICT_pars = {}
    DICT_pars['very-short-epidemic'] = convert_rate_from_second_to_minute(beta_sec, gamma_sec)
    #SCENARIO2 - SHORT incubation and infectious periods
    DICT_pars['short-epidemic'] = convert_rate_from_second_to_minute(beta_sec/2, gamma_sec/2)

    #rescaled epidemiological parameters
    DICT_pars['recov4'] = convert_rate_from_second_to_minute(beta_sec/4, gamma_sec/4)
    DICT_pars['recov8'] = convert_rate_from_second_to_minute(beta_sec/8, gamma_sec/8)
    DICT_pars['recov16'] = convert_rate_from_second_to_minute(beta_sec/16, gamma_sec/16)

    #selected (beta,gamma) after grid-search
    Betas  = np.linspace(1e-5, DICT_pars['recov16'][0],5)
    Gammas = np.linspace( DICT_pars['recov8'][1], DICT_pars['short-epidemic'][1] ,5)
    inds_select = (3,2)
    i,j = inds_select
    DICT_pars['epid_grid_select_n1'] = ( Betas[i], Gammas[j])
    
    return DICT_pars

def to_dense_sym(W_D0, USERS_select):
    '''
    converts contacts dataframe for a given date
    to a symmetrical dense contact matrix
    '''
    #conversion to a dense matrix - matrix is uppertriangular
    W_D0_dense = W_D0.pivot(index = 'u1', columns = 'u2', values = 'n_minutes')
    W_D0_dense = W_D0_dense.reindex(USERS_select, columns = USERS_select).fillna(0)
    #sum it to its transpose to get a symmetrical matrix
    W_D0_sym = W_D0_dense + W_D0_dense.T
    
    return W_D0_sym


def gen_contact_daily(df_contact_hour, 
                      USERS_select, 
                      clip_max_minutes = True):
    '''
    aggregate hour contact at daily resolution level
    clip_max_minutes: clip the edges weight to the max number of minutes in a day
    '''
    df_contact_daily = df_contact_hour.copy()
    df_contact_daily['date'] = pd.to_datetime(df_contact_daily['date_hour'], errors = 'coerce').dt.date

    df_contact_daily = df_contact_daily.groupby(['u1', 'u2', 'date']).agg({
    'n_minutes': 'sum' }).reset_index()
    #df_contact_daily = df_contact_daily.groupby(['u1','u2','date']).sum().reset_index()
    W = subset_df_feature(df_contact_daily, 'date')
    W = {D:to_dense_sym(W_D, USERS_select) for D, W_D in W.items()}

    if clip_max_minutes:
        W = {k: c.clip(upper = 1440) for k,c in W.items()}

    return W 

def apply_transitions(X,X_d):
    
    #apply the transitions
    X[:,0] -= X_d[:,0]
    X[:,1] += X_d[:,0] - X_d[:,1]*X[:,1]

    #set the state vector to 0 if the user is recovered
    X[:,0] *= (1 - X_d[:,1])
    X[:,1] *= (1 - X_d[:,1])

def init_state(X, X_d, p, n_init = None):
    '''
    p : probability of initial infected
    n_init: size of the initial seed
    '''

    X[:,0] = 1
    X_d[:,0] = np.random.permutation([1] * n_init + [0] * (X.shape[0] - n_init))
    apply_transitions(X,X_d)
    
def sample_transition(X, X_d, W_D0, pars, gamma_daily = False):
    '''
    D0: day at which contacts are estimated 
    '''
    beta_min, gamma_min = pars
    #Symmetric contact matrix 
    #W_D0 = to_dense_sym(W[D0], USERS_select)
    #get the probability of transitions
    # Count of minutes in contact with an infectious during day D0
    # W_D0 is the contact matrix with entries corresponding to contact durations
    # X[:,1] is the infection indicator vectoe
    N_minutes_contact = np.dot(W_D0, X[:,1])
    #compounded probabilities for (S->I) transition during day D0
    prob_S_to_I = (1 - (1 - beta_min)**(N_minutes_contact) )*X[:,0]
    #compounded probabilities for (I->R) transition during day D0
    if gamma_daily:
        prob_I_to_R = gamma_min*X[:,1]
    else:
        prob_I_to_R = (1 - (1 - gamma_min)**(24*60) )*X[:,1]

    X_d[:,0] = np.random.binomial(1, prob_S_to_I)  
    X_d[:,1] = np.random.binomial(1, prob_I_to_R)
    
    apply_transitions(X,X_d)

#run the simulation
def epid_simulation(W,
                    USERS_select,
                    pars, 
                    p = 0.1, 
                    n_init = None,
                    seed_number = None,
                    gamma_daily = False):

    if seed_number is not None: 
        np.random.seed(seed_number)
    
    #state vector (columns are [S,I])
    X = np.zeros((len(USERS_select), 2))    
    #transition indicator (columns are [StoI, ItoR])
    X_d = np.zeros((len(USERS_select), 2))

    init_state(X, X_d, p, n_init)    
    #MONITORING NUMBER OF SUCEPTIBLES AND INFECTED
    ts_SI = []
    ts_SI.append(X.sum(axis=0))
    
    Dates = np.sort(list(W.keys()))
    for D0 in Dates:
        sample_transition(X, X_d, W[D0], pars, gamma_daily = gamma_daily)
        ts_SI.append(X.sum(axis=0))

    return np.array(ts_SI)

def iter_epid_simulation(W, 
                         Date_range,
                         USERS_select, 
                         pars, 
                         p_init = 0.05, 
                         n_init = None,
                         N_iter=100, 
                         from_weekday = False, 
                         gamma_daily = False):
    '''
    W: dictionary of contact data
    pars : (beta, gamma) at minute resolution 
    p_init : fraction of initial infected
    '''

    COLLECT_simulations = []
    COLLECT_epid_metrics = []
    
    if from_weekday:
        #remove the first dates if they belong to the weekend
        W_dates = Date_range.date
        #print(W_dates)
        while W_dates[0].weekday() >= 5:  # 5 and 6 correspond to Saturday and Sunday
            W_dates = W_dates[1:]
        W = {w:W[w] for w in W_dates}

    for i in range(N_iter):
       
        #compute the time-series of suscpetible-infected and R0
        ts_SI_R0 = epid_simulation(W, 
                                   USERS_select,
                                   pars = pars, 
                                   p = p_init, 
                                   n_init = n_init,
                                   seed_number = None, 
                                   gamma_daily = gamma_daily)
        
        #compute the epidemiological metrics 
        epid_metrics = compute_epid_metrics(ts_SI_R0)
        
        COLLECT_simulations.append(ts_SI_R0)    
        COLLECT_epid_metrics.append(epid_metrics)

    COLLECT_simulations = np.array(COLLECT_simulations)
    COLLECT_epid_metrics = np.array(COLLECT_epid_metrics)

    return COLLECT_simulations, COLLECT_epid_metrics

def gen_epid_calibration_grid_dict():
    '''
    generate grid of parameters for epidemic calibration
    '''
    
    DICT_epid_pars = gen_dict_epid_pars()
    DICT_grid = {}
    
    Betas  = np.linspace(1e-5, 1e-2, 5)
    Gammas = np.linspace(DICT_epid_pars['short-epidemic'][1], 
                         DICT_epid_pars['recov16'][1], 5)
    N_inits = [1,3,5,10]
    DICT_grid['v0'] = epid_pars_grid = list(itertools.product(Betas, Gammas, N_inits))

    #Generate the grid of parameters for the grid search
    Betas  = np.linspace(1e-4, 1e-1, 10)
    Gammas = np.linspace(DICT_epid_pars['short-epidemic'][1], 
                         DICT_epid_pars['recov16'][1], 5)
    N_inits = [1,3,5,10]
    DICT_grid['v1'] = epid_pars_grid = list(itertools.product(Betas, Gammas, N_inits))

    #Generate the grid of parameters for the grid search
    Betas  = np.linspace(1e-4, 1e-3, 10)
    Gammas = np.linspace(DICT_epid_pars['very-short-epidemic'][1], 
                         DICT_epid_pars['recov16'][1], 10)
    N_inits = [1,2,3,4,5,6,10]
    DICT_grid['v2'] =  list(itertools.product(Betas, Gammas, N_inits))

    #Generate the grid of parameters for the grid search
    Betas  = np.linspace(.5e-4, 1e-2, 20)
    Gammas = np.linspace(DICT_epid_pars['very-short-epidemic'][1], 
                         DICT_epid_pars['recov8'][1], 10)
    N_inits = [1,3,5,10]#,15,20,30,50]
    DICT_grid['v3'] = list(itertools.product(Betas, Gammas, N_inits))
    
    return DICT_grid
    
    
def compute_epid_metrics(ts_SI):
    '''
    epidemic metric computation on a single experimental realization
    return ['peak_day', 'peak_size', 'final_size', 'epid_duration', 'day_final_case']
    '''
    #time-series of susceptibles
    S = ts_SI[:,0]
    #time-series of infected
    I = ts_SI[:,1]
    
    #infected peak
    peak_size = np.max(I)
    #first time in which infected time-series reached its peak
    peak_day = np.where(I==peak_size)[0][0] + 1
    
    #duration of the epidemic - computed as the day in which the I became constantly 0
    epid_duration = len(I)
    cumsum_condition = np.cumsum(I[::-1]) > 0
    indices = np.argwhere(cumsum_condition) 

    #time of final case
    #corresponding to the day since when the susceptible curve became a plateau
    #np.argmin returns the first index of occurring minimum in case of multiple occurrences
    day_final_case = np.argmin(S) + 1 
    
    if indices.size > 0:  # Check if indices array is not empty
        epid_duration -= (indices[0, 0] + 1)
    
    #final size of the epidemic - number of individuals which became infected
    final_size = I[0] + S[0] - S[-1]
    
    Epid_metrics = [peak_day, peak_size, final_size, epid_duration, day_final_case]

    return Epid_metrics

def compute_epid_stats(COLLECT_epid_metrics):
    '''
    compute epidemiological statistics
    '''

    qmin, q25, qmedian, q75, qmax = np.percentile(COLLECT_epid_metrics, 
                                                  [0, 25, 50, 75, 100], 
                                                  axis=0, 
                                                  method = 'nearest')
    
    metrics = ['peak_day', 'peak_size', 'final_size', 'epid_duration', 'day_final_case']
    df_epid_stats = pd.DataFrame([qmin, q25, qmedian, q75, qmax], columns = metrics)
    df_epid_stats.index = ['min','25IQR', 'median', '75IQR','max']

    return df_epid_stats


def compute_SI_stats(COLLECT_simulations):
    
    q25, q50, q75 = np.percentile(COLLECT_simulations, [25, 50, 75], axis=0)
    Q_columns = ['S_perc25', 'I_perc25', 'S_median', 'I_median', 'S_perc75', 'I_perc75' ]
    df_SI_stats = pd.DataFrame(np.column_stack([q25,q50,q75]), columns = Q_columns)
    
    return df_SI_stats


def compute_R0_daily(df_CD_level, 
                     epid_pars, 
                     gamma_daily = False): 
    '''
    df_CD_level : contact daily data for a specific sparsity level
    '''
    df_contacts, USERS_select, Date_range = import_complete_contacts_users_drange() 

    beta, gamma = epid_pars
    
    #Probability of recovering during a day
    if not gamma_daily:
        #perform the conversion of gamma to daily resolution
        gamma =  (1 - (1 - gamma)**(24*60) )

    #compute symmetric contact matrices for each day
    #the entries are the minutes in contact over the whole day for each couple
    df_CD_level_scm = df_CD_level.groupby('date').apply(lambda x : to_dense_sym(x, USERS_select), include_groups = False)
    
    #compute R0 for each individual
    R0_func = lambda v : (1 - (1-beta)**v) 
    
    df_R0_individual = df_CD_level_scm.map(R0_func).sum(axis=1).reset_index()
    df_R0_individual[0] += (1-gamma)
    
    #aggregate R0 estimates at a daily level
    df_R0_daily = df_R0_individual.groupby('date')[0].mean()

    return df_R0_daily

def compute_R0_individual(C_d, epid_pars, gamma_daily = False):
    '''
    C_d : daily symmetric contact matrix
        - contact durations are at 1-minute resolution
    epid_pars : (beta, gamma) tuple
    '''
    
    #(beta,gamma) at 1/minute resolution
    beta, gamma = epid_pars
    
    #Probability of recovering during a day
    Gamma =  (1 - (1 - gamma)**(24*60) )
    
    #compute R0 for each individual - probability of infecting a neighbour
    R0_func = lambda v : (1 - (1-beta)**v) 
    df_R0_individual = C_d.map(R0_func).sum(axis=1)
    if gamma_daily: 
        df_R0_individual += 1-gamma
    else:
        df_R0_individual += (1-Gamma)
       
    return df_R0_individual

def compute_R0_individual_period(DICT_contacts, epid_pars, gamma_daily = False):
    '''
    return dataframe of individual R0 components
    '''
    DICT_R0 = {day: compute_R0_individual(contacts, epid_pars, gamma_daily) for day, contacts in DICT_contacts.items()}
    df_R0 = pd.DataFrame(DICT_R0)
    return df_R0

def compute_avg_R0(Contacts, epid_pars, Dates_select, gamma_daily = False):
    '''
    Compute average R0 on the collection of contacts 
    '''
    df_R0_complete = compute_R0_individual_period(Contacts, epid_pars, gamma_daily)
    R0_global_mean = df_R0_complete[Dates_select].mean().mean() 
    return R0_global_mean




#############################################
########## EXPERIMENTAL PIPELINE ############
#############################################

def get_complete_user_drange(df_fn,
                             SW_width_days, 
                             par_lach,
                             ghr = 8,
                             t_step = '1hour'): 

    #get selected users
    Cols_info = ['id','sparse', 'weekstep_index']
    df_info = df_fn[Cols_info]
    #get selected epidemic time-period
    USERS_select = df_info[~df_info['sparse']]['id'].unique()
    Date_start = pd.to_datetime(df_fn.columns[3])
    Date_end = Date_start + dt.timedelta(days = SW_width_days-1)
    Date_range = pd.date_range(Date_start, Date_end)

    return USERS_select, Date_range

def get_complete_contacts(df_fn,
                          SW_width_days, 
                          par_lach,
                          ghr = 8,
                          t_step = '1hour',
                          FOLD_save = None):
    '''
    df_fn: sequence-dataset after complete user search
    '''
    
    USERS_select, Date_range = get_complete_user_drange(df_fn,
                                                        SW_width_days, 
                                                        par_lach,
                                                        ghr = 8,
                                                        t_step = '1hour')

    if FOLD_save is None:
        mintime, maxdtime, maxdiam = par_lach
        FOLD_contact = f'LACH_mintime_{mintime}_maxdtime_{maxdtime}_maxdiam_{maxdiam}_ghr{ghr}_tstep_{t_step}/'
        FOLD_save = DICT_paths['TMP'] + 'f_008_D1_All_Contacts/' + FOLD_contact
    
    df_contacts = []
    for D0 in Date_range:
        #select a specific date
        D0 = str(D0.date())
        #import within and marginal contacts
        df_D0_cw = pd.read_csv(f'{FOLD_save}df_cwithin_{D0}.csv')
        df_D0_cm = pd.read_csv(f'{FOLD_save}df_cmargin_{D0}.csv')
        ###############################################join all contacts
        df_D0_contacts = compute_tot_contacts(df_D0_cm, df_D0_cw)
        c_u1 = df_D0_contacts['u1'].isin(USERS_select)
        c_u2 = df_D0_contacts['u2'].isin(USERS_select)
        df_D0_contacts = df_D0_contacts[c_u1 & c_u2]
        df_contacts.append(df_D0_contacts)
        
    df_contacts = pd.concat(df_contacts, axis = 0)
    
    return df_contacts, USERS_select, Date_range

def get_complete_sequences(df_fn, USERS_select):
    #select the study period
    c_w0 = df_fn['weekstep_index']==0
    c_sparse  = df_fn['sparse']
    df_select = df_fn.loc[c_w0 & ~c_sparse]
    df_select = df_select.drop(['sparse', 'weekstep_index'], axis = 1).set_index('id')
    df_select.loc[USERS_select]
    return df_select

def compute_sparse_metric(df_sparse_seq, 
                          metric,
                          SW_width_days):
    '''
    df_sparse_seq : collection of sparse sequences
    '''
    
    #generate the gap dataframe from the sparse sequences
    df_gaps = gen_gaps_df(df_sparse_seq)

    #SPARSITY METRICS AND STUDIED RANGES
    Seq_series = compute_seq_sparsity(df_gaps, 
                                      metric, 
                                      N_hours = SW_width_days*24).sort_values()

    N_recs = df_sparse_seq.sum(axis=1) 
    N_tot_hours = len(df_sparse_seq.columns)
    Inds_complete = N_recs[N_recs == N_tot_hours].index
    Seq_complete = pd.Series(np.zeros(len(Inds_complete)), index = Inds_complete) 
    Seq_series = Seq_series.add(Seq_complete, fill_value = 0)
    
    return Seq_series.sort_values()

def gen_mask(level, 
             Seq_series,
             df_seq,
             df_seq_complete, 
             metric, 
             SW_width_days):
    '''
    level: sparsity level
    Seq_series: sparsity-series of sequence sample
    df_seq : sequence sample
    df_seq_complete: complete sequence undergoing sparsification
    metric: employed sparsity metric
    SW_width_days : slidng window period

    Generate mask conditionally on sparsity of the (1-epsilon) complete user
    '''

    #reorder the eps-complete users based on their sparsity
    Seq_sparsity_complete = compute_sparse_metric(df_seq_complete, 
                                                  metric, 
                                                  SW_width_days)

    #compute the sparsity-level of the eps-complete users
    sco_vals_2d = np.floor(Seq_sparsity_complete.values*100)/100 + 0.01
    #count by grouping by 2-digits
    sco_vals_2d, sco_vals_count = np.unique(sco_vals_2d, return_counts=True)
    
    lev_mask = []
    
    #iterative sampling conditioned on the eps-complete sparsity level
    for v,c in zip(sco_vals_2d, sco_vals_count):
        
        v_indexes = Seq_series[Seq_series.between(level[0], level[1] - v)].index
        
        v_indexes_sampled = np.random.choice(v_indexes, size = c, replace = True)
        
        v_gaps_sampled = df_seq.iloc[v_indexes_sampled]
        
        lev_mask.append(v_gaps_sampled)
        
    lev_mask = pd.concat(lev_mask, axis=0)
    lev_mask.index = Seq_sparsity_complete.index
    
    return lev_mask

def detect_stops(df_LOC, par_lach): 
    '''
    df_LOC: raw location data
    par_lach : parameter of the lachesis algorithm
    '''
    
    dur_min, dt_max, delta_roam = par_lach
    df_LOC = df_LOC.rename(columns = {'lng':'lon'})
    df_LOC['datetime'] = pd.to_datetime(df_LOC['datetime']).astype('int64') // 10**9
    time, lat, lon = 'datetime', 'lat','lon'
    
    c_dict = {time: 'unix_timestamp', lat: 'y', lon: 'x'}
    df_LOC = df_LOC[['id',time,lat,lon]].rename(columns = c_dict)
    convert_df_coord_3587(df_LOC, lat = 'y', lon = 'x')

    df_stops = df_LOC.groupby('id').apply(lambda x: lachesis(x, dur_min, dt_max, delta_roam))
    df_stops = df_stops.reset_index().drop('level_1',axis=1)
    
    return df_stops


def estimate_contacts(df_stops_DR, 
                      ghr, 
                      time_step = '1hour'):
    '''
    df_stops_DR : simple stop location table
    ghr: geohash resolution for contact estimation
    time_resolution: can be '1hour' or '1minute'
        
    Performs within and marginal contact estimation
    Returns : df_cwithin_hour, df_cmargin_hour
    '''

    #set the geohash resolution for contact estimation
    df_stops_DR = df_stops_DR.rename(columns = {'geohash9': 'geohash'})
    df_stops_DR['geohash'] = df_stops_DR['geohash'].str[:ghr]
    #unique stop-indicator (univoque association of a stop to a user)
    df_stops_DR['unique_stop_id'] = range(len(df_stops_DR))
    df_stops_DR['stop_hour'] = df_stops_DR.apply(lambda x : pd.date_range(start = x['start_time'].floor('h'), 
                                                                          end = x['end_time'].floor('h'), 
                                                                          freq='h'), axis=1)
    
    df_stops_DR = df_stops_DR.explode('stop_hour').drop_duplicates(['geohash', 'stop_hour', 'unique_stop_id'])
    
    #SIMPLE STOP TABLE OF CELLS CONTRIBUTING TO CONTACTS
    df_stops_CONTACT = get_stops_CONTACT(df_stops_DR) 

    #[OUTPUT1] - WITHIN CELL CONTACTS (defaut is 1hour resolution with number of minutes in contacts)
    df_cwithin = compute_contact_table(df_stops_CONTACT, 
                                       geohash_col = 'geohash', 
                                       time_resolution = time_step)
    
    #[OUTPUT2] - MARGINAL CONTACT ESTIMATION
    df_ch_margin = []
    #compute the widths of the collection of geohashes
    df_gh_widths = get_gh_widths(df_stops_DR)
    wx,wy = df_gh_widths.loc[0,'width_x'], df_gh_widths.loc[0,'width_y']
    #compute collection of stops which have populated neighbours
    #for contact-contributing stop selection
    df_stops_NEIGHBOUR = get_stops_NEIGHBOURS(df_stops_DR)
    #collection of shift sequences
    Shift_seq = [ (-1,0), (0,-1), (1,0)]
    for Shift in Shift_seq:    
        #print(f'\t Shift {Shift}')
        #perform medioid shift and new geohash assignation  
        geohash_shift(df_stops_NEIGHBOUR, Shift, wx,wy, perc_width = 0.5, ghr = ghr, new_gh_col = 'geohash_shift')
        #select stops contributing to contact after the shift 
        df_stops_NEIGHBOUR_shift_CONTACT = get_stops_CONTACT(df_stops_NEIGHBOUR, 'geohash_shift')
        #explode the contact stops at 1minute resolution for original geohash assignment and marginal stop selection
        df_stops_NEIGHBOUR_1min = df_stops_NEIGHBOUR_shift_CONTACT.copy()
        df_stops_NEIGHBOUR_1min['stop_minute'] = df_stops_NEIGHBOUR_1min.apply(lambda x : pd.date_range(start = x['start_time'], 
                                                                                                      end = x['end_time'], 
                                                                                                      freq='min'), axis=1)
        
        df_stops_NEIGHBOUR_1min = df_stops_NEIGHBOUR_1min.explode('stop_minute')
        #compute contacts   
        df_contacts_shift = compute_contact_table(df_stops_NEIGHBOUR_shift_CONTACT, 
                                                  geohash_col = 'geohash_shift', 
                                                  time_resolution = '1minute')
        #compute original geohashes at 1 minute resolution
        df_contacts_shift = join_original_geohashes(df_contacts_shift, 
                                                    df_stops_NEIGHBOUR_1min, 
                                                    geohash_col = 'geohash')
        #filter out unchanged geohash couples
        df_contacts_shift = df_contacts_shift[df_contacts_shift['u1_geohash'] != df_contacts_shift['u2_geohash']]
        df_ch_margin.append(df_contacts_shift)

    #Join all marginal contacts from the 3 shifts 
    df_ch_margin = pd.concat(df_ch_margin, axis=0)
    #Dropping duplicated contacts over the shifts
    df_ch_margin_unique = df_ch_margin.drop_duplicates(subset = ['date_time','u1','u2'], keep = 'first').copy()
    
    if time_step == '1minute':
        return df_cwithin, df_ch_margin_unique
        
    df_ch_margin_unique['date_hour'] = df_ch_margin_unique['date_time'].dt.floor('h')
    Cols_select = ['u1','u2','u1_geohash','u2_geohash','date_hour']
    df_cmargin_hour = df_ch_margin_unique.groupby(Cols_select).size().reset_index()
    df_cmargin_hour.rename(columns = {0:'n_minutes'}, inplace = True)

    return df_cwithin, df_cmargin_hour

################################################
########## RESULTS ANALYSIS FUNCTIONS ##########
################################################

def gen_sparsity_ranges():
    '''
    generate sparsity ranges
    '''
    
    Metrics_ranges = {'perc-missing-hours' : [(0.1, 0.2), 
                                              (0.2, 0.3), 
                                              (0.3, 0.4),
                                              (0.4, 0.5),
                                              (0.5, 0.6)]}
    
    metric = 'perc-missing-hours'
    Levels = Metrics_ranges[metric]
    
    return Levels

def gen_gapseq_filenames(FOLD_SAVE,
                         LIST_days, 
                         LIST_epsilon,
                         tol_stops, 
                         startday01 = False):

    DICT_tol_hours = {(e,w): int(24*e*w) for e in LIST_epsilon for w in LIST_days}
    DICT_filenames = {(e,w): f'df_gapseq_epidwindowdays_{w}_epsilon_{e}_hourstol_{DICT_tol_hours[(e,w)]}' 
                      for e in LIST_epsilon for w in LIST_days}  
    if startday01:
        DICT_filenames = {p:f'{n}_startday01' for p,n in DICT_filenames.items()}

    if tol_stops: 
        DICT_filenames = {p:f'{n}_withtolstops' for p,n in DICT_filenames.items()}
        
    return DICT_filenames

def import_complete_contacts_users_drange():
    '''
    under complete scenario
    returns contact dataframe, list of selected users and date range of epidemic simulation
    '''

    #import gap sequences for the overall sample
    FOLD_seq = DICT_paths['TMP'] + 'f01_009_D1_gap_sequence_sample/'
    LIST_days    = [14, 21, 28]
    LIST_epsilon = [0, 0.05, 0.1]
    tol_stops = False
    DICT_filenames = gen_gapseq_filenames(FOLD_seq,
                                          LIST_days, 
                                          LIST_epsilon,
                                          tol_stops)
    
    #parameters for lachesis stop-detection
    par_lach = (10,360,50)
    #epidemic-window search parameters
    #days of the window
    SW_width_days = 28
    #tolerance on percentage of missing hours 
    epsilon = 0.05
    fn = DICT_filenames[(epsilon, SW_width_days)]
    df_seq = pd.read_csv(f'{FOLD_seq}{fn}.csv')
    #[1] get contacts of complete trajectories 
    df_contacts, USERS_select, Date_range = get_complete_contacts(df_seq,
                                                                  SW_width_days,
                                                                  par_lach,
                                                                  ghr = 8,
                                                                  t_step = '1hour')
    return df_contacts, USERS_select, Date_range

def import_gap_seq(): 
    '''
    import default collection of gap sequences 
    '''
    
    #epidemic-window search parameters - days of the window
    SW_width_days = 28
    #tolerance on percentage of missing hours 
    epsilon = 0.05
    
    #[0.1] import gap sequences sample
    FOLD_seq = DICT_paths['TMP'] + 'f01_009_D1_gap_sequence_sample/'
    LIST_days    = [14, 21, 28]
    LIST_epsilon = [0, 0.05, 0.1]
    tol_stops = False
    
    DICT_filenames = gen_gapseq_filenames(FOLD_seq,
                                          LIST_days, 
                                          LIST_epsilon,
                                          tol_stops)
    
    fn = DICT_filenames[(epsilon, SW_width_days)]
    df_seq = pd.read_csv(f'{FOLD_seq}{fn}.csv')
    
    return df_seq

def import_sparse_contacts(s,l,N_si):
    '''
    Import sparse contacts 
    under the new iterated sparsification scenario
    
    N_sparse_iter : sparsification iteration realization
    level: sparsification level
    ss: sparsification scenario
    '''
    
    FOLD_sparse = DICT_paths['TMP'] + 'f01_013_D1_iter_sparsified_mask_loc_stops_contacts/'
    FOLD_save = f'{FOLD_sparse}Iter_sparse_{N_si}/{s}/{l}/'
    df_cmargin_hour = pd.read_csv(FOLD_save + 'df_contact_marginal.csv')
    df_cwithin_hour = pd.read_csv(FOLD_save + 'df_contact_within.csv')
    df_contacts_level = compute_tot_contacts(df_cmargin_hour, df_cwithin_hour)
    
    return df_contacts_level

def import_contacts_complete(return_contact_dataframe = False):
    
    df_contacts, USERS_select, Date_range = import_complete_contacts_users_drange() 
    if return_contact_dataframe: 
        return df_contacts
    W_complete = gen_contact_daily(df_contacts, USERS_select)
    
    return W_complete  

def import_contacts_complete_sparse(List_ss, 
                                    Levels, 
                                    Ranges_iter, 
                                    import_complete = True): 
    
    df_contacts, USERS_select, Date_range = import_complete_contacts_users_drange() 

    DICT_contacts = {}
    DICT_contacts['Complete'] = import_contacts_complete()
    DICT_contacts= {(s, l, N_si): gen_contact_daily(import_sparse_contacts(s,l,N_si), USERS_select) 
                          for s in List_ss 
                          for l in Levels 
                          for N_si in Ranges_iter}
    if import_complete:
        DICT_contacts.update({'Complete': import_contacts_complete()})
    
    return DICT_contacts
    
def import_sparse_mask(s,l,N_si):
    '''
    Import sparse mask
    under the new iterated sparsification scenario
    
    N_sparse_iter : sparsification iteration realization
    level: sparsification level
    ss: sparsification scenario
    '''
    
    FOLD_sparse = DICT_paths['TMP'] + 'f01_013_D1_iter_sparsified_mask_loc_stops_contacts/'
    FOLD_save = f'{FOLD_sparse}Iter_sparse_{N_si}/{s}/{l}/'
    df_mask = pd.read_csv(f'{FOLD_save}df_mask.csv', index_col = 0)
    
    return df_mask


def gen_complete_epid_groundtruth(scenario = 'epid_grid_select_n1'): 
    
    DICT_epid_pars = gen_dict_epid_pars()
    epid_pars = DICT_epid_pars[scenario]
    n_init = 3
    N_iter = 100
    
    return epid_pars, n_init, N_iter
    

def import_str_complete_epid_groundtruth_str():
    '''
    returns string representation of epidemic modeling groundtruth parameters
    '''
    scenario = 'epid_grid_select_n1'
    epid_pars, n_init, N_iter = gen_complete_epid_groundtruth(scenario)
    _beta   = np.round(epid_pars[0]*100, 3)
    _gamma  = np.round(epid_pars[1]*100, 3)
    str_epid = scenario
    str_epid += f'_beta_{_beta}_gamma_{_gamma}_ninit_{n_init}_Niter_{N_iter}' 
    
    return str_epid

#import sparse stops dataset
def import_sparse_stops(s,l,N_si):
    '''
    s : sparsity mechanism
    l : sparsity level
    N_si : number of sparsification iterations
    '''
    FOLD_sparse = DICT_paths['TMP'] + 'f01_013_D1_iter_sparsified_mask_loc_stops_contacts/'
    FOLD_iter = f'{FOLD_sparse}Iter_sparse_{N_si}/'
    FOLD_iter_ss = f'{FOLD_iter}{s}/'
    FOLD_iter_ss_l = f'{FOLD_iter_ss}{l}/'
    df_stops = pd.read_csv(FOLD_iter_ss_l + 'df_stops.csv', index_col = 0, parse_dates = ['start_time', 'end_time'])
    
    return df_stops

def import_stops_complete_sparse(List_ss, 
                                 Levels, 
                                 Ranges_iter, 
                                 Date_range, 
                                 USERS_select):

    #import sparse stops
    DICT_stops = {(ss, l, N_sparse_iter) : import_sparse_stops(ss, l, N_sparse_iter)
                  for ss in List_ss 
                  for l in Levels 
                  for N_sparse_iter in Ranges_iter}

    #import complete stops
    PATH_stops_complete = DICT_paths['TMP'] + 'f_007_D1_lachesis_stops_AND_loc-1hdi/'
    dur_min, dt_max, delta_roam = (10,360,50)
    par_str = f'mintime_{dur_min}_maxdtime_{dt_max}_maxdiam_{delta_roam}'
    df_stops_complete = pd.read_csv(PATH_stops_complete + 'df_stops_' + par_str + '.csv', 
                                    index_col = 0, 
                                    parse_dates = ['start_time', 'end_time']) 
    df_stops_complete = filter_stops(df_stops_complete, 
                                     USERS_select, 
                                     (Date_range[0],Date_range[-1]))
    DICT_stops['Complete'] = df_stops_complete
    
    return DICT_stops

#[TODO] functions for importing contact metrics    

#functions for importing epidemic modeling outcomes
def get_groundtruth_epidpars(): 
    '''
    static function generating the groudtruth epidemiological parameters for simulation
    '''
    #epidemic groundtruth parameters 
    scenario_epid = 'epid_grid_select_n1'
    n_init = 3
    N_iter = 100
    
    return scenario_epid, n_init, N_iter

def get_groundtruth_epidpars_v2():
    
    #epidemic groundtruth parameters 
    scenario_epid = 'epid_grid_select_n1'
    n_init = 3
    N_iter = 100
    DICT_epid_pars = gen_dict_epid_pars()
    epid_pars = DICT_epid_pars[scenario_epid]
    beta, gamma = epid_pars[0], epid_pars[1]
    
    return beta, gamma, n_init

    
def gen_fold_epid(FOLD_exp = 'f01_011_D1_epidemic_simulations/'): 
    '''
    FOLD_exp: experimental folder, the current variants are:
        - starting simulation from first day
        - starting the simulation from first weekday

    Returns the folder in which epidemiological simulations are stored
    '''

    scenario_epid, n_init, N_iter = get_groundtruth_epidpars()
    DICT_epid_pars = gen_dict_epid_pars()
    epid_pars = DICT_epid_pars[scenario_epid]
    
    _beta   = np.round(epid_pars[0]*100, 3)
    _gamma  = np.round(epid_pars[1]*100, 3)
    #_p_init = np.round(p_init*100, 2)
    str_epid = scenario_epid
    str_epid += f'_beta_{_beta}_gamma_{_gamma}_ninit_{n_init}_Niter_{N_iter}' 
    
    FOLD_emo = DICT_paths['TMP'] + FOLD_exp
    FOLD_scenario_epid = FOLD_emo + str_epid + '/'

    return FOLD_scenario_epid

def get_emo_data(path_epid_stats, path_simulations):
    
    df_epid_stats = pd.read_csv(path_epid_stats, index_col = 0)
    df_epid_stats = df_epid_stats.set_index('index').T

    with open(path_simulations, 'rb') as f:
        COLLECT_simulations = pickle.load(f)
        
    return df_epid_stats, COLLECT_simulations


def get_emo_complete(FOLD_exp = 'f01_011_D1_epidemic_simulations/'):

    FOLD_scenario_epid = gen_fold_epid(FOLD_exp = FOLD_exp)    
    
    path_epid_stats  = f'{FOLD_scenario_epid}df_epid_stats.csv'
    path_simulations = f'{FOLD_scenario_epid}simulations.pkl' 
    df_epid_stats, COLLECT_simulations = get_emo_data(path_epid_stats, path_simulations)
    
    return df_epid_stats, COLLECT_simulations

def get_emo_sparse(FOLD_exp,
                   Sparse_scenario_emv): 
    '''
    FOLD_exp: experiments folder
    Sparse_scenario_emv: (s,l, N_si) 
        - s : sparsity mechanism 
        - l : sparsity level
        - N_si : sparsification iteration
        - emv  : implemented epidemic modeling variation
            - standard, contact_rescaled, param_calibration_grid_v3
    '''
    
    FOLD_epid_scenario = gen_fold_epid(FOLD_exp)
    ss, l, N_si, emv = Sparse_scenario_emv
    FOLD_iter = f'{FOLD_epid_scenario}/Iter_sparse_{N_si}/' 
    FOLD_ss = f'{FOLD_iter}{ss}/'

    if emv == 'standard':
        path_epid_stats  = f'{FOLD_ss}df_epid_stats_{l}.csv'
        path_simulations = f'{FOLD_ss}simulations_{l}.pkl' 
        df_es_level, sim_level = get_emo_data(path_epid_stats, path_simulations)
        return df_es_level, sim_level 
    else: 
        path_epid_stats  = f'{FOLD_ss}df_epid_stats_{l}_emv_{emv}.csv'
        path_simulations = f'{FOLD_ss}simulations_{l}_emv_{emv}.pkl' 
        df_es_level, sim_level = get_emo_data(path_epid_stats, path_simulations)
        return df_es_level, sim_level

def import_EMO_complete(FOLD_exp):
    DICT_EMO = {}
    df_epid_stats, COLLECT_simulations = get_emo_complete(FOLD_exp)
    DICT_EMO['Complete'] = (df_epid_stats, COLLECT_simulations)
    return DICT_EMO 

def import_EMOs(FOLD_exp,
                List_ss, 
                Levels, 
                Ranges_iter, 
                EMVs):

    DICT_EMO = import_EMO_complete(FOLD_exp) 
    #update joining the results also from the other sparsity scenarios
    DICT_EMO.update({(s,l, N_si, emv): get_emo_sparse(FOLD_exp, (s,l, N_si, emv)) 
                     for s in List_ss 
                     for l in Levels
                     for N_si in Ranges_iter 
                     for emv in EMVs}) 
    
    return DICT_EMO



def get_emo_sparse_calibration_gridrmse(FOLD_exp,
                                        Sparse_scenario, 
                                        grid_version = 'v3'): 
    '''
    Returns the rmse df for each grid elements for a specific sparsity scenario
    rmse default computation is over the curve of susceptibles

    FOLD_exp: experiments folder
    Sparse_scenario_emv: (s,l, N_si) 
        - s : sparsity mechanism 
        - l : sparsity level
        - N_si : sparsification iteration
    '''
    
    FOLD_epid_scenario = gen_fold_epid(FOLD_exp)
    ss, l, N_si = Sparse_scenario
    FOLD_iter = f'{FOLD_epid_scenario}/Iter_sparse_{N_si}/' 
    FOLD_ss = f'{FOLD_iter}{ss}/'
    FOLD_grid = f'{FOLD_ss}grid_search_info/'
    path_grid_rmse = f'{FOLD_grid}df_epg_rmse_{l}_grid_{grid_version}.csv'
    df_grid_rmse = pd.read_csv(path_grid_rmse, index_col = 0)
    
    return df_grid_rmse

def import_EMO_calibration_rmse(FOLD_exp,
                                List_ss, 
                                Levels, 
                                Ranges_iter, 
                                grid_version = 'v3'):
    '''
    dictionary of rmse values for each (s, l, N_si) instance
    '''

    DICT_calibration_grid_rmse = {(s,l,N_si): get_emo_sparse_calibration_gridrmse(FOLD_exp,
                                                                                  (s,l,N_si), 
                                                                                  grid_version = grid_version)
                                  for s in List_ss 
                                  for l in Levels
                                  for N_si in Ranges_iter}
    
    return DICT_calibration_grid_rmse    

def get_path_sparse_scenario(FOLD_exp, 
                             Sparse_scenario):
    '''
    Import grid simulations for a given sparsity scenario
    '''
    FOLD_epid_scenario = gen_fold_epid(FOLD_exp)
    ss, l, N_si = Sparse_scenario
    FOLD_iter = f'{FOLD_epid_scenario}Iter_sparse_{N_si}/' 
    FOLD_ss = f'{FOLD_iter}{ss}/'
    return FOLD_ss
    
def get_path_grid_experiments(FOLD_exp,
                              Sparse_scenario): 
    
    FOLD_ss = get_path_sparse_scenario(FOLD_exp, 
                                       Sparse_scenario)
    
    FOLD_grid = f'{FOLD_ss}grid_search_info/'
    
    return FOLD_grid

def get_path_grid_rmse(FOLD_exp, 
                       Sparse_scenario, 
                       grid_version = 'v3'):

    s,l,N_si = Sparse_scenario

    FOLD_grid = get_path_grid_experiments(FOLD_exp, Sparse_scenario)
    path_grid_rmse = f'{FOLD_grid}df_epg_rmse_{l}_grid_{grid_version}.csv'
    return path_grid_rmse

#simulate for a specific grid element
def import_grid_element_simulation(FOLD_grid,
                                   grid_point, 
                                   l, 
                                   grid_version, 
                                   N_grid_iter = 100):
    
    grid_par, n_init = grid_point[:2], grid_point[2]
    _beta    = np.round(grid_par[0]*100, 6)
    _gamma   = np.round(grid_par[1]*100, 6)
    str_epid = f'_beta_{_beta}_gamma_{_gamma}_ninit_{n_init}_Niter_{N_grid_iter}' 
    with open(f'{FOLD_grid}simulations_{l}_{str_epid}_grid_{grid_version}.pkl', 'rb') as f:
        COLLECT_simulations = pickle.load(f)
        
    return COLLECT_simulations

def import_grid_simulations(FOLD_exp, 
                            Sparse_scenario, 
                            GRID, 
                            grid_version):
    '''
    Import all simulations for the grid points
    '''
    
    s,l,N_si = Sparse_scenario
    
    FOLD_grid = get_path_grid_experiments(FOLD_exp,
                                          Sparse_scenario)

    DICT_grid_sims = {grid_point: import_grid_element_simulation(FOLD_grid,
                                                                 grid_point, 
                                                                 l, 
                                                                 grid_version,
                                                                 N_grid_iter = 100)
                      for grid_point in GRID}
    
    return DICT_grid_sims

def compute_mean_sim(Sim_c):
    '''
    Colection of simulations
    returns mean_S, mean_I, mean_CC 
    where [S,I] : [Susceptibles, Infected]#, Cumulative cases]
    '''
    #compute average [S,I, Cumulative cases] over simulations
    mean_S  = np.mean(Sim_c[:,:,0], axis = 0)
    mean_I  = np.mean(Sim_c[:,:,1], axis = 0) 
    #mean_CC = np.mean(N_pop - Sim_c[:,:,0], axis = 0)
    return mean_S, mean_I

def compute_median_sim(Sim_c):
    '''
    Colection of simulations
    returns mean_S, mean_I
    where [S,I] : [Susceptibles, Infected]
    '''
    #compute average [S,I, Cumulative cases] over simulations
    median_S  = np.median(Sim_c[:,:,0], axis = 0)
    median_I  = np.median(Sim_c[:,:,1], axis = 0) 
    #mean_CC = np.mean(N_pop - Sim_c[:,:,0], axis = 0)
    return median_S, median_I

def compute_mean_delta_sim(Sim_c):
    '''
    Colection of simulations
    returns mean_dS, mean_dI
    where [S,I] : [Susceptibles, Infected]
    '''
    #compute average [delta_S, delta_I] over simulations
    mean_dS  = np.mean(np.diff(Sim_c[:,:,0], axis=1), axis=0)
    mean_dI  = np.mean(np.diff(Sim_c[:,:,1], axis=1), axis=0)
    
    return mean_dS, mean_dI


def compute_rmse(Sim_c, Sim_g0):
    '''
    Sim_c, Sim_g0 : two collection of epidemic simulations
    returns [rmse_S, rmse_I]
    '''
    
    Means_c = compute_mean_sim(Sim_c)
    Means_g0 = compute_mean_sim(Sim_g0) 

    Medians_c = compute_median_sim(Sim_c)
    Medians_g0 = compute_median_sim(Sim_g0)
    
    Rmse_g0 = [np.sqrt( np.mean((c - c_g0)**2) ) for c,c_g0 in zip(Means_c, Means_g0)]
    Rmse_median_g0 = [np.sqrt( np.mean((c - c_g0)**2) ) for c,c_g0 in zip(Medians_c, Medians_g0)]
    
    return Rmse_g0, Rmse_median_g0


def get_em_vals(DICT_EMO_metrics, k, em, N_users, 
                norm = False):
    '''
    Import epidemic metric values
    k  : scenario
    em : epidemic metric
    '''
    #Ordered list of epidemic metrics as computed from EMO collection
    Emo_metrics_names = ['peak_day', 'peak_size', 'final_size', 'epid_duration', 'day_final_case'] 
    ind_epid_metric = Emo_metrics_names.index(em)
    
    if norm and em in ['peak_size', 'final_size']:
        return 100*DICT_EMO_metrics[k][:,ind_epid_metric]/N_users
    else:
        return DICT_EMO_metrics[k][:,ind_epid_metric]


#given an epidemic ensemble, compute the fraction of epidemics which die-out
#FRACTION OF FALSE NEGATIVES UNDER DIFFERENT MECHANISMS
def get_fraction_levels_FN(DICT_EMO_metrics,
                           Levels,
                           s,N_si, emv,
                           N_users,
                           Th_max_infected_perc = 1, 
                           em = 'peak_size',
                           include_complete = False):
    '''
    Return fractions of epidemics which die-out for each sparsity level
    List having length of Levels
    get for each sparsity range the fraction of False Negative epidemic realizations
    that is, epidemics for which the Max of daily infected percentage is below a Thresold (default 1%)
    '''
    #function for computing the percentage of false negatives
    p_FN = lambda x : np.sum(x<Th_max_infected_perc)/len(x)

    Ks = [(s,l, N_si, emv) for l in Levels]
    if include_complete:
        Ks = ['Complete'] + Ks

    p_FNs = [p_FN(get_em_vals(DICT_EMO_metrics, k, em, N_users, norm = True)) 
             for k in Ks]
    
    return np.array(p_FNs)


def get_fraction_levels_FN_complete(DICT_EMO_metrics,
                                    N_users,
                                    Th_max_infected_perc = 1,     
                                    em = 'peak_size'):
    '''
    Return fractions of epidemics which die-out for each sparsity level
    List having length of Levels
    get for each sparsity range the fraction of False Negative epidemic realizations
    that is, epidemics for which the Max of daily infected percentage is below a Thresold (default 1%)
    '''
    #function for computing the percentage of false negatives
    p_FN = lambda x : np.sum(x < Th_max_infected_perc)/len(x)
    p_FNs = [p_FN(get_em_vals(DICT_EMO_metrics, 'Complete', em, N_users, norm = True))]
    
    return np.array(p_FNs)

#import gap masks
def get_gap_mask(Sparse_scenario):
    s,l,N_si = Sparse_scenario
    FOLD_sparse = DICT_paths['TMP'] + 'f01_013_D1_iter_sparsified_mask_loc_stops_contacts/'
    path_ss = f'{FOLD_sparse}Iter_sparse_{N_si}/{s}/{l}/'
    df_mask = pd.read_csv(f'{path_ss}df_mask.csv', index_col = 0)
    return df_mask

def import_masks(List_ss, 
                 Levels, 
                 Ranges_iter):

    #update joining the results also from the other sparsity scenarios
    DICT_masks = {(s,l, N_si): get_gap_mask((s,l, N_si)) 
                  for s in List_ss 
                  for l in Levels
                  for N_si in Ranges_iter} 
    
    return DICT_masks


#########################################################
######### MODEL CALIBRATION WITH OPTUNA #################
#########################################################

def gen_dict_coi(EMO_sims): 
    '''
    compute the curves of interest (coi) 
    from an ensemble of population
    '''
    
    #Construct the possible reference curves
    mean_S, mean_I = compute_mean_sim(EMO_sims)
    median_S, median_I = compute_median_sim(EMO_sims)
    mean_delta_S, mean_delta_I = compute_mean_delta_sim(EMO_sims)
    
    DICT_ref_curves = {'mean_S': mean_S, 
                       'mean_I': mean_I, 
                       'median_S': median_S, 
                       'median_I': median_I, 
                       'mean_delta_S': mean_delta_S, 
                       'mean_delta_I': mean_delta_I}
    
    return DICT_ref_curves

def gen_reference_curve(DICT_EMO, metric_ref): 
    '''
    generate the reference curve used for the objective function computation
    '''
    Sims_complete = DICT_EMO['Complete'][1]
    curve_reference = gen_dict_coi(Sims_complete)[metric_ref]
    return curve_reference

#define the function for obtai
def generate_sparse_sims(params,
                         Contacts_sparse, 
                         USERS_select,
                         Dates, 
                         N_iter, 
                         from_weekday = True):
    '''
    params: set of epidemiological parameters (beta, gamma, seedsize)
    Contacts_sparse: daily sequence of sparse contacs
    USERS_select: selected users
    Dates: dates involved in the simulation
    N_iter: number of simulations
    from_weekday: condition for starting the simulations from the first weekday
    '''

    beta, gamma, seedsize = params

    Sims, _ = iter_epid_simulation(Contacts_sparse, 
                                   Dates,
                                   USERS_select, 
                                   pars = (beta, gamma), 
                                   n_init = seedsize,
                                   N_iter = N_iter, 
                                   from_weekday = from_weekday)

    return Sims


def compute_objective_function(params, 
                               Curve_ref,
                               metric_ref,
                               Contacts_sparse, 
                               USERS_select,
                               Date_range, 
                               N_iter = 100, 
                               obj_metric = 'RMSE',
                               from_weekday = True): 
    
    Sims_sparse = generate_sparse_sims(params,
                                       Contacts_sparse, 
                                       USERS_select,
                                       Date_range, 
                                       N_iter = N_iter, 
                                       from_weekday = from_weekday)
    
    Curve_sparse = gen_dict_coi(Sims_sparse)[metric_ref]

    if obj_metric == 'RMSE':
        RMSE = np.sqrt(np.mean((Curve_sparse - Curve_ref)**2))
        return RMSE
    if obj_metric =='MAE':
        MAE = np.mean(np.abs(Curve_sparse - Curve_ref))
        return MAE

def get_study_best_params(study):
    
    #Compare the groundtruth with the outcome from sparse data
    dict_best_params = study.best_params 
    beta  = dict_best_params['beta']
    gamma = dict_best_params['gamma']
    seedsize = dict_best_params['seedsize']
    
    best_params = (beta, gamma, seedsize)

    return best_params

class Objective_param_search:

    def __init__(self, 
                 Curve_ref,
                 obj_metric,
                 GRID_stats, 
                 Contacts_dict, 
                 sparse_scenario, 
                 metric_ref,
                 USERS_select, 
                 Date_range, 
                 N_iter=100, 
                 from_weekday=True):
        
        self.Curve_ref = Curve_ref
        self.obj_metric = obj_metric
        self.GRID_stats = GRID_stats
        self.Contacts_dict = Contacts_dict
        self.sparse_scenario = sparse_scenario
        self.metric_ref = metric_ref
        self.USERS_select = USERS_select
        self.Date_range = Date_range
        self.N_iter = N_iter
        self.from_weekday = from_weekday
    
    def __call__(self, trial):

        GRID_stats = self.GRID_stats
        beta = trial.suggest_float("beta", GRID_stats.loc["min", "beta"], GRID_stats.loc["max", "beta"])
        gamma = trial.suggest_float("gamma", GRID_stats.loc["min", "gamma"], GRID_stats.loc["max", "gamma"])
        seedsize = trial.suggest_int("seedsize", int(GRID_stats.loc["min", "seedsize"]), int(GRID_stats.loc["max", "seedsize"]))
        
        params = (beta, gamma, seedsize) 
        
        Objective_param_search = compute_objective_function(params, 
                                                            self.Curve_ref,
                                                            self.metric_ref,
                                                            self.Contacts_dict[self.sparse_scenario], 
                                                            self.USERS_select,
                                                            self.Date_range, 
                                                            N_iter = self.N_iter,
                                                            obj_metric = self.obj_metric,
                                                            from_weekday = self.from_weekday)
        
        return Objective_param_search

def optuna_param_search(Curve_ref, 
                        metric_obj,
                        GRID_stats, 
                        DICT_contacts, 
                        sparse_scenario, 
                        metric_ref,
                        USERS_select, 
                        Date_range, 
                        N_iter=100, 
                        from_weekday=True, 
                        n_trials = 100):

    study = optuna.create_study(direction="minimize") 
        
    Obj = Objective_param_search(Curve_ref, 
                                 metric_obj,
                                 GRID_stats, 
                                 DICT_contacts, 
                                 sparse_scenario, 
                                 metric_ref,
                                 USERS_select, 
                                 Date_range, 
                                 N_iter= N_iter, 
                                 from_weekday = from_weekday)

    study.optimize(Obj, 
                   n_trials = n_trials, 
                   show_progress_bar = True) 
    
    return study

###############################################
######### CONTACT CORRECTIONS #################
###############################################

def import_record_indicator_complete_users(SW_width_days = 28, 
                                           epsilon = 0.05):
    '''
    dataframe of hourly record indicator for complete users obtained from 
    Sliding window of 28 days
    tolerance of 5%
    '''
    
    #import sample of gap sequences
    path_gap_seq = '/home/fedde/work/Project_Penn/TMP/f01_009_D1_gap_sequence_sample/'
    path_gap_seq += f'df_gapseq_epidwindowdays_{SW_width_days}_epsilon_{epsilon}_hourstol_33.csv'
    df_gap_seq = pd.read_csv(path_gap_seq)
    
    #hourly record indicator for complete users
    df_hri_uc = df_gap_seq.query(f'sparse == {False} & weekstep_index ==0')
    
    df_hri_uc = df_hri_uc.drop(['sparse','weekstep_index'], axis =1).set_index('id')
    
    return df_hri_uc

def get_sparse_record_indicator(Date_range, k):
    '''
    Return the hourly record indicator for the sparsified complete users within a given date-range
    Date_range : list of 2 datetimes
    k = (s, l, N_si) ;
        s : mechanism
        l : level 
        N_si : sparsification iteration  
    '''
    
    #[0] import the missingness mask within the given daterange
    mask_k = import_sparse_mask(*k).T
    mask_k.index = pd.to_datetime(mask_k.index)
    #select the mask within the study period
    mask_k = mask_k.loc[(mask_k.index >= Date_range[0]) 
                         & (mask_k.index <= Date_range[-1]+dt.timedelta(days=1))]
    
    #[1] import the hourly record indicator
    df_hri_uc = import_record_indicator_complete_users().T
    df_hri_uc.index = pd.to_datetime(df_hri_uc.index)
    df_hri_uc = df_hri_uc.loc[mask_k.index][mask_k.columns]
    
    #[2] overlay the mask on the hourly record indicator
    df_hri_uc = df_hri_uc*mask_k

    return df_hri_uc

def gen_coefs_coupled(coef):
    '''
    coef: series with:
        - index : associated to the user 
        - value : completness coefficient (global or hourly) 
    return matrix with coupled coefficients : 1/(c_i*c_j)
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_outer = 1 / np.outer(coef.values, coef.values)
        inv_outer[np.isinf(inv_outer)] = 0  # Set infinities to 0
        inv_outer[np.isnan(inv_outer)] = 0 
        R_coefs = pd.DataFrame(inv_outer, 
                               index   = coef.index, 
                               columns = coef.index)
        return R_coefs

def gen_normalized_hourly_coefs(coef_hourly):
    '''
    compute normalized hourly coefficients
    '''
    
    #Compute the coupled coefficients for each hour
    List_coefs_coupled = [gen_coefs_coupled(coef_hourly.loc[h]) for h in range(24)]
    
    #Normalize each set of 24 hourly coefficients for each trajectory couple
    Alpha_coefs = 24/sum(List_coefs_coupled)
    
    #Rescale the coefficients by the normalization factor
    List_coefs_coupled = [Alpha_coefs*C for C in List_coefs_coupled]
    
    #Rearrange the coefs into one single dataframe
    df_coefs_coupled = []
    for hour in range(24):
        df_long = List_coefs_coupled[hour].stack()#.reset_index()
        df_long = List_coefs_coupled[hour].stack()
        df_long.index.set_names(['u1', 'u2'], inplace=True)
        df_long = df_long.reset_index(name='rescaling_coef')
        df_long['hour'] = hour
        df_coefs_coupled.append(df_long)
        
    df_coefs_coupled = pd.concat(df_coefs_coupled, axis=0)

    return df_coefs_coupled



def aggregate_daily_contacts(df_c):
    '''
    aggregate contact durations at daily resolution
    clip duration to max. duration in a day: 1440
    '''
    df_c['date'] = pd.to_datetime(df_c['date_hour']).dt.date
    df_c_daily = df_c.groupby(['date','u1','u2']).agg('n_minutes').sum().clip(upper = 1440)
    df_c_daily = df_c_daily.reset_index()
    return df_c_daily

def gen_df_compare_contacts(df_contacts_daily, 
                            C_k_daily):
    '''
    Generate dataframe of daily contacts (compare Complete against Sparse)

    df_contacts_daily : dataset of groundtruth contacts
    C_k_daily: dataset of contacts from sparse trajectories
    '''

    df_merged = df_contacts_daily.rename(columns={'n_minutes': 'n_minutes_complete'}).merge(
        C_k_daily.rename(columns={'n_minutes': 'n_minutes_sparse'}),
        on=['date', 'u1', 'u2'],
        how='outer').fillna(0)
    
    return df_merged 

def classify_contacts_sparse(df_merged):
    '''
    classify a sparse contact according to its change in duration
    new classification column is 'sparse_contact_class'
    '''

    conditions = [df_merged['n_minutes_sparse']== df_merged['n_minutes_complete'], 
                  (df_merged['n_minutes_sparse'] < df_merged['n_minutes_complete']) & (df_merged['n_minutes_sparse']!=0),
                  df_merged['n_minutes_sparse'] > df_merged['n_minutes_complete'],
                  df_merged['n_minutes_sparse'] == 0.0]
    
    choices = ['unchanged', 
               'reduced', 
               'increased', 
               'eliminated']

    df_merged['sparse_contact_class'] = np.select(conditions, choices)

def compare_contact_stats(df_contacts_daily, 
                          C_k_daily):
    
    List_contact_durations = [df_contacts_daily['n_minutes'].values, 
                              C_k_daily['n_minutes'].values]
    
    df_merged = gen_df_compare_contacts(df_contacts_daily, 
                                        C_k_daily)
    
    classify_contacts_sparse(df_merged)
    df_merged['delta_duration'] = df_merged['n_minutes_sparse'] - df_merged['n_minutes_complete']
    
    return df_merged

def bin_contact_durations(df, bin_size):
    
    x_bins = np.arange(0, 1440 + bin_size, bin_size)
    y_bins = np.arange(0, 1440 + bin_size, bin_size)
    range_bins = np.arange(len(x_bins))
    
    heatmap_data, xedges, yedges = np.histogram2d(
        df["n_minutes_complete"].values,
        df["n_minutes_sparse"].values,
        bins=[x_bins, y_bins])
    
    return heatmap_data


def gen_contact_comparison(ipw_type = 'ipw_global',
                           s = 'Data_driven',
                           N_iter = 0): 

    df_contacts, USERS_select, Date_range = import_complete_contacts_users_drange() 
    df_contacts_daily = aggregate_daily_contacts(df_contacts)
    Date_range_fromweekday = convert_daterange_from_weekday(Date_range)
    Dates = Date_range_fromweekday.date
    Dates_plus1 = np.append(Dates, Dates[-1]+dt.timedelta(days=1))

    Levels = gen_sparsity_ranges()

    DICT_compare_contacts = {}
    for l in Levels: 
        #generate the contacts (groundtruth, sparsity-induced and corrected)
        Sparse_scenario = (s, l, N_iter)
        Contacts_sparse = import_sparse_contacts(*Sparse_scenario)
        Contacts_sparse_rescaled = ipw_rescale_contacts(Sparse_scenario, 
                                                        USERS_select, 
                                                        Date_range_fromweekday, 
                                                        rescaling = ipw_type,
                                                        return_contact_dataframe = True) 
        
        DICT_contacts = {'sparse': Contacts_sparse,
                         'sparse_corrected': Contacts_sparse_rescaled}
        
        DICT_contacts_daily = {k: aggregate_daily_contacts(C) 
                               for k,C in DICT_contacts.items()}
        
        DICT_compare_contacts.update({(k,l): compare_contact_stats(df_contacts_daily, C) 
                                     for k, C in DICT_contacts_daily.items()})
        
    return DICT_compare_contacts

def compute_R0_groundtruth(C, gamma_daily = False):
    C_ = C.copy().rename(columns = {'n_minutes_sparse': 'n_minutes'})
    epid_pars, n_init, N_iter = gen_complete_epid_groundtruth()    
    C_R0 = compute_R0_daily(C_, epid_pars, gamma_daily = gamma_daily)
    return C_R0

def compute_R0_epid_pars(C, epid_pars, gamma_daily = False):
    C_ = C.copy().rename(columns = {'n_minutes_sparse': 'n_minutes'})
    C_R0 = compute_R0_daily(C_, epid_pars, gamma_daily = gamma_daily)
    return C_R0


#########################################################
######### SPARSITY MODELING - FOR FUTURE WORK ###########
#########################################################


def alternate_exp_sampling(P, N_hours):
    '''
    this algorithm tries to emulate the alternating mechanism of phone usage and not-usage 
    it start with a phone-usage interval
    '''
    j = 0
    gap_seq = [j]
    
    while j < N_hours:
        #record interval generation
        j+= np.random.exponential(1/(1-P))
        gap_seq.append(j)
        #gap interval generation
        j+= np.random.exponential(1/P)
        gap_seq.append(j)

    #minute conversion
    gap_seq_min = (np.array(gap_seq)*60).astype(int)
    gap_seq_min = gap_seq_min[gap_seq_min < (60*N_hours - 1)]
    
    return gap_seq_min

def gen_alternate_poisson_mask(P, USERS_select, Date_range):
    '''
    iplement alternate extraction from a geometric distribution
    '''

    #date range at minute resolution
    N_hours = int((Date_range[1] - Date_range[0]).total_seconds() /3600)
    dr_min = pd.date_range(Date_range[0], Date_range[1],freq = 'min', inclusive = 'left')
    df_rec_ind = pd.DataFrame(0, index=dr_min, columns=USERS_select)
    
    for i in range(len(USERS_select)):
        g_seq = alternate_exp_sampling(P, N_hours)
        df_rec_ind.iloc[g_seq,i] = 1
    
    #ensure that all boolean sequences sum up to an even number
    Ind = (df_rec_ind.sum(axis=0)%2==0)
    Ind_users = df_rec_ind.sum(axis=0)[~Ind]
    df_rec_ind.loc[dr_min[-1],Ind_users.index]=1
    
    #create the record minute indicator
    df_rec_ind = df_rec_ind.apply(lambda x : interp_boolean(x), axis=0)
    df_rec_ind = df_rec_ind.T
    
    return df_rec_ind


def create_graph(df_CT): 
    
    G = nx.Graph()
    G.add_edges_from((row['u1'], row['u2'],
                      {'weight': row['n_minutes']}) for _, row in df_CT.iterrows())
    return G 

def compute_contact_metrics(df_CT, 
                            Percs = [0,25,50,75,100], 
                            return_size = False):

    G = create_graph(df_CT)

    if return_size: 
        n,e = len(G.nodes), len(G.edges)
        return pd.DataFrame([[n,e]], columns = ['nodes', 'edges'])
    
    DICT_Percs = {"contact_duration_iqr": np.percentile(df_CT['n_minutes'], Percs, method = 'nearest'),
                  "degree_iqr": np.percentile(list(dict(G.degree()).values()), Percs), 
                  "weighted_degree_iqr": np.percentile(list(dict(G.degree(weight = 'weight')).values()), Percs),
                  "clustering_iqr": np.percentile(list(nx.clustering(G).values()), Percs)} 
                  #"weighted_clustering_iqr": np.percentile(list(nx.clustering(G, weight = 'weight').values()), Percs)}
    
    df_results = pd.DataFrame(DICT_Percs)
    df_results.index = ['min', 'iqr25', 'median', 'iqr75', 'max']
    
    return df_results

    
def contact_estimate_daily_partition(df_lcf_stops, ghr, only_last_day = False):
    '''
    subset the stop table by day and evaluate the contacts and concatenate them
    '''
    
    USERS = df_lcf_stops['id'].unique()
    Study_period = (df_lcf_stops['start_time'].min().date(),
                    df_lcf_stops['end_time'].max().date())
    #Partition the study period into 1day intervals
    SP_partition = pd.date_range(Study_period[0], Study_period[1] + dt.timedelta(days = 2), freq = 'D')
    SP_partition = [(a1,a2) for a1,a2 in zip(SP_partition[:-1], SP_partition[1:]) ]

    if only_last_day: 

        SP0 = SP_partition[-1]
        #print(SP0)
        df_stops_SP = filter_stops(df_lcf_stops, 
                                   USERS,  
                                   Date_range = SP0)
        
        df_cwithin, df_cmargin = estimate_contacts(df_stops_SP, ghr)
        return df_cwithin, df_cmargin
    
        
    #FIRST DAY
    SP0 = SP_partition[0]
    #print(SP0)
    df_stops_SP = filter_stops(df_lcf_stops, 
                               USERS,  
                               Date_range = SP0)
    
    df_cwithin, df_cmargin = estimate_contacts(df_stops_SP, ghr)

    #FOLLOWING DAYS
    for SP0 in SP_partition[1:-1]:
        
        #print(SP0)
        df_stops_SP = filter_stops(df_lcf_stops, 
                                   USERS,  
                                   Date_range = SP0)
    
        df_cwithin_s, df_cmargin_s = estimate_contacts(df_stops_SP, ghr)
        
        #attach these 2dfs to df_cwithin and to df_cmargin
        df_cwithin = pd.concat([df_cwithin, df_cwithin_s], ignore_index=True)
        df_cmargin = pd.concat([df_cmargin, df_cmargin_s], ignore_index=True)

    return df_cwithin, df_cmargin


def compute_features_nx(G_d, 
                        return_names = False):
    '''
    daily contact weightx nx Graph
    '''

    if return_names:
        return ['nodes_positive_degree', 'size_largest_cc']
    #number of nodes with positive degree
    N_nodes_degree_pos = sum(1 for node, degree in G_d.degree() if degree > 0)

    #size of largest connected component
    Size_largest_cc = len(max(nx.connected_components(G_d), key=len)) 

    NX_metrics = np.array([[N_nodes_degree_pos, Size_largest_cc]])

    return np.array(NX_metrics)

def compute_node_networkcontraint(P, i):
    #P is the normalized adjacency matrix
    p_i = P[i]
    #index of the neighbours
    Gamma_i_inds = np.argwhere(p_i != 0).ravel()
    if len(Gamma_i_inds) > 0:
        p_i = p_i[Gamma_i_inds]
        P_gamma_i = P[np.ix_(Gamma_i_inds, Gamma_i_inds)]
        p_nc_i = np.dot(P_gamma_i, p_i)
        p_i = p_i + p_nc_i
        return np.sum(p_i**2)
    else:
        return 0

def compute_features_node(G_d, 
                          return_names = False):
    '''
    G_d: networkx graph
    '''

    if return_names: 
        return ['degree', 'cumulative_time', 'network_constraint', 'clustering']

    X = nx.to_numpy_array(G_d)
    #degree
    Degrees = np.sum(1*(X>0), axis = 0)
    #cumulative time in contact
    CTC = np.sum(X, axis=0)
    
    #network contraints values
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = X / row_sums
    NCs = np.array([compute_node_networkcontraint(P,i) for i in range(len(P))])

    #clustering coefficient values
    CCs = np.array(list(nx.clustering(G_d).values()))

    #collection of metrics
    Metrics = [Degrees, CTC, NCs, CCs]

    return np.stack(Metrics, axis=1)

#hourofday-weekperiod coverage
def gen_hw_col(v):
    '''
    gen hourofday_weekday column
    '''
    v = pd.to_datetime(v)
    _wdays = [0,1,2,3,4,5]
    _wperiod = np.array(['weekend','weekday'])
    return [f"{t}_{_wperiod[w*1]}" for t,w in zip(v.hour, v.weekday.isin(_wdays))] 

def align_temporally_contact_coverages(C_k, R_k):
    '''
    align the coverage and the contact-data
    '''
    Dates_common = np.intersect1d(C_k['date'], np.unique(R_k.index.date))
    C_k = C_k[C_k['date'].isin(Dates_common)]
    R_k =  R_k[ [d in Dates_common for d in R_k.index.date]]#.isin(Dates_common) ]
    return C_k, R_k

def compute_coef_aligned(R_k):
    '''
    compute globally aligned coefficients
    '''
    return pd.DataFrame(np.dot(R_k.T,R_k)/len(R_k), 
                        index = R_k.columns, 
                        columns = R_k.columns)

def compute_coef_aligned_timebucket(R_k, h):
    '''
    compute aligned coefficients on different temporal buckets
    '''
    R_k_h = R_k.copy()
    if h == 'hourofday_weekday':
        R_k_h[h] = gen_hw_col(R_k.index.values)
    if h == 'date':
        R_k_h[h] = R_k.index.date
    return R_k_h.groupby(h).apply(lambda df : compute_coef_aligned(df))

def import_contacts_coverage(k, Date_range):
    '''
    return contact and coverage data within a specific missingness scenario
    '''
    
    #[0] import the sparsified user hourly record indicator
    R_k = get_sparse_record_indicator(Date_range, k)
    
    #[1] import the sparsity induced contacts
    C_k = import_sparse_contacts(*k)
    C_k['hour'] = pd.to_datetime(C_k['date_hour']).dt.hour
    C_k['date'] = pd.to_datetime(C_k['date_hour']).dt.date
    C_k['hourofday_weekday'] = gen_hw_col(C_k['date_hour'].values)
    
    C_k, R_k = align_temporally_contact_coverages(C_k,R_k)
    
    return C_k, R_k

def compute_weight_trim(df_wg2, weight):
        '''
        compute weight trim from inverse probabilistic weighting 
        '''
        Ws = df_wg2[weight].dropna().values
        Ws_cv = np.std(Ws)/np.mean(Ws)
        Ws_med = np.median(Ws)
        Ws_trim = 3.5*np.sqrt(1 + Ws_cv**2)*Ws_med
        return Ws_trim

def compute_coefs(C_k, R_k, Contact_corrections): 
    '''
    compute all coefficients on different temporal buckets
    '''

    #[0] Coverage coefficients
    coef_global = compute_coef_aligned(R_k)
    coef_daily  = compute_coef_aligned_timebucket(R_k, 'date')
    coef_hw     = compute_coef_aligned_timebucket(R_k, 'hourofday_weekday')
    coefs_dict = {'global': coef_global, 
                  'daily': coef_daily, 
                  'hourofday_weekday': coef_hw}
    
    C_k['c1_global']  = C_k.apply(lambda row: coef_global.loc[row['u1'], row['u1']], axis = 1)
    C_k['c2_global']  = C_k.apply(lambda row: coef_global.loc[row['u2'], row['u2']], axis = 1)
    C_k['c12_global']  = C_k.apply(lambda row: coef_global.loc[row['u1'], row['u2']], axis = 1)
    C_k['c1_daily']  = C_k.apply(lambda row: coef_daily.loc[row['date']].loc[row['u1'], row['u1']], axis = 1)
    C_k['c2_daily']  = C_k.apply(lambda row: coef_daily.loc[row['date']].loc[row['u2'], row['u2']], axis = 1)
    C_k['c12_daily']  = C_k.apply(lambda row: coef_daily.loc[row['date']].loc[row['u1'], row['u2']], axis = 1)
    C_k['c1_how']  = C_k.apply(lambda row: coef_hw.loc[row['hourofday_weekday']].loc[row['u1'], row['u1']], axis = 1) 
    C_k['c2_how']  = C_k.apply(lambda row: coef_hw.loc[row['hourofday_weekday']].loc[row['u2'], row['u2']], axis = 1) 
    C_k['c12_how']  = C_k.apply(lambda row: coef_hw.loc[row['hourofday_weekday']].loc[row['u1'], row['u2']], axis = 1)
    
    def compute_denom(rescaling):
        if rescaling == 'ipw_global':
            denom = C_k['c1_global'] * C_k['c2_global']
        elif rescaling == 'ipw_daily':
            denom = C_k['c1_daily'] * C_k['c2_daily']
        elif rescaling == 'ipw_hourofday_weekday':
            denom = C_k['c1_how'] * C_k['c2_how']
        elif rescaling == 'ipw_global_aligned':
            denom = C_k['c12_global']
        elif rescaling == 'ipw_daily_aligned':
            denom = C_k['c12_daily']
        elif rescaling == 'ipw_hourofday_weekday_aligned':
            denom = C_k['c12_how']
        else:
            raise ValueError(f"Unknown rescaling type: {rescaling}")
        # Replace 0 or NaN with 1
        denom_safe = denom.replace(0, 1).fillna(1)
        return denom_safe

        
    #[1] Weights (that is the factor multiplied to the sparse contact duration)
    for c in Contact_corrections: 
        
        #weight: factor of multiplication for contact duration
        denom = compute_denom(c)
        C_k[f'w_{c}'] = 1/denom
        
        #weight clipping for anomalous factors
        W_cap = compute_weight_trim(C_k, f'w_{c}')
        C_k[f'w_{c}_trim'] = C_k[f'w_{c}'].clip(upper = W_cap)

    return coefs_dict


def impute_undetected_contacts(C_k, h = 'hourofday_weekday', min_obs = 1): 
    
    C_k['date_hour'] = pd.to_datetime(C_k['date_hour'])
    
    # 1) compute observed contacts at the time-bucket h
    C_k_observed_contacts = (
        C_k.groupby(['u1', 'u2', h], as_index=False)
          .size()
          .rename(columns={'size': 'observed_contacts'})
    )
    C_k_observed_contacts = C_k_observed_contacts[C_k_observed_contacts['observed_contacts'] >= min_obs]
    
    H_ranges = pd.date_range(C_k['date_hour'].min(),
                             C_k['date_hour'].max(), freq='h')
    pairs = C_k[['u1','u2']].drop_duplicates()
    hours = pd.DataFrame({'date_hour': H_ranges})
    
    C_imputed = pairs.merge(hours, how='cross')  # cartesian product
    C_imputed = (C_imputed
             .merge(C_k[['u1','u2','date_hour']], 
                    on=['u1','u2','date_hour'], 
                    how='left', indicator=True)
             .query('_merge == "left_only"')
             .drop('_merge', axis=1)
    )
    
    C_imputed['n_minutes'] = 60
    C_imputed['hour'] = pd.to_datetime(C_imputed['date_hour']).dt.hour
    C_imputed['date'] = pd.to_datetime(C_imputed['date_hour']).dt.date
    C_imputed['hourofday_weekday'] = gen_hw_col(C_imputed['date_hour'].values)
    
    C_imputed = C_imputed.merge(
        C_k_observed_contacts[['u1', 'u2', 'hourofday_weekday']],
        on=['u1', 'u2', 'hourofday_weekday'],
        how='inner'
    )
    
    C_k = pd.concat([C_k, C_imputed], axis = 0)
    
    return C_k

def ipw_rescale_contacts(k, 
                         USERS_select, 
                         Date_range,
                         rescaling = 'ipw_global', 
                         return_contact_dataframe = False, 
                         import_contact_weights = True):
    '''
    -rescaling
        ipw_str = 'ipwt_global'
        first part of the string defines the type of ipw
         ipw   : basic
         ipwt  : with trimmed weights
         ipwt1h : clipping and trimming to 60 minutes)
        second part of the string defines the contact-correction type
    '''
    
    if import_contact_weights:
        FOLD_save = '/home/fedde/work/Project_Penn/TMP/f01_015_D1_contacts_with_weights/'
        C_k = pd.read_csv(FOLD_save + '_'.join([str(i) for i in k]) + '.csv', index_col = 0)
    else:
        #time-expensive
        C_k, R_k = import_contacts_coverage(k, Date_range)
        Contact_corrections = ['ipw_global', 'ipw_daily', 'ipw_hourofday_weekday']
        Contact_corrections += [f'{c}_aligned' for c in Contact_corrections]
        _ = compute_coefs(C_k, R_k, Contact_corrections) 
    
    ipw_type = rescaling.split('_')[0]
    cc = '_'.join(rescaling.split('_')[1:])
    wcol = f'w_{cc}'

    if ipw_type == 'ipw':
        #basic ipw
        C_k['n_minutes'] *= C_k[f'w_ipw_{cc}']
        
    if ipw_type == 'ipwt': 
        #ipw with trimmed coefficients
        C_k['n_minutes'] *= C_k[f'w_ipw_{cc}_trim']
        
    if ipw_type == 'ipwt1h':
        #ipw with trimmed coefficients
        C_k['n_minutes'] *= C_k[f'w_ipw_{cc}_trim']
        #rescaled date-hour duration is trimmed to 60 minutes
        C_k['n_minutes'] = C_k['n_minutes'].clip(upper=60)

    #trimmed inverse probabilistic weighiting plus imputation
    if rescaling == 'ipwtplusimp_hourofday_weekday_minobs1':
        wcol = 'w_ipw_hourofday_weekday'
        W_cap = compute_weight_trim(C_k, wcol)
        C_k[f'{wcol}_trim'] = C_k[wcol].clip(upper = W_cap)
        C_k['n_minutes'] *= C_k[f'{wcol}_trim']
        C_k['n_minutes'] = C_k['n_minutes'].clip(upper = 60)
        C_k = impute_undetected_contacts(C_k, h = 'hourofday_weekday', min_obs=1)

    #TRY: SET ALL CONTACT WEIGHTS TO 60 MINUTES
    if rescaling == 'ipw_CU_60min':
        print('rescaling 60 min')
        C_k['n_minutes'] = 60
        
    #SET ALL CONTACT WEIGHTS TO 1440 MINUTES
    if rescaling == 'CU_1440min':
        C_k['n_minutes'] = 1440
    
    if return_contact_dataframe:
        return C_k
        
    #[3] convert to a daily contact dataframe and clip to 1440 (max minutes in a day)
    Contacts_k = gen_contact_daily(C_k, USERS_select)
    Contacts_k = {k: c.clip(upper = 1440) for k,c in Contacts_k.items()}
    
    return Contacts_k

#################################################################
######### EVALUATE FEASIBILITY OF CONTACT CORRECTION  ###########
#################################################################

def compute_targets(C_k): 
    '''
    compute target of contact estimation in comparison to the ground truth
    '''
    _L_det =  lambda C_k : C_k[C_k['n_minutes'] > 0]
    _L_udet =  lambda C_k : C_k[C_k['n_minutes'] == 0]
    Links_det  = _L_det(C_k) 
    Links_udet = _L_udet(C_k)
    CV_det_k  = Links_det['n_minutes'].sum() 
    CV_det_gt = Links_det['n_minutes_gt'].sum()
    CV_udet_gt = Links_udet['n_minutes_gt'].sum()
    CV_gt_tot = CV_det_gt + CV_udet_gt
    
    output_sr = pd.Series([len(Links_det), len(Links_det)+len(Links_udet), 
                           CV_det_k, CV_det_gt, CV_udet_gt, CV_gt_tot], 
                           index = ['nlinks_detected', 'nlinks_target', 
                                    'CV_detected', 'CV_detected_target', 'CV_undetected_target','CV_tot_target'])

    output_sr['nlinks_retention'] = output_sr['nlinks_detected']/output_sr['nlinks_target']
    output_sr['CV_retention'] = output_sr['CV_detected']/output_sr['CV_tot_target']
    output_sr['CV_loss_detected'] = (output_sr['CV_detected_target'] - output_sr['CV_detected'])/output_sr['CV_tot_target']
    output_sr['CV_loss_undetected'] = output_sr['CV_undetected_target']/output_sr['CV_tot_target']
    
    return output_sr

def compare_ipw_gt_weights(C_k, 
                           bucket, 
                           Date_range, 
                           k):
    '''
    # Adjust the IPW capping threshold so that contact-duration correction 
    # accounts only for the portion of duration loss assigned 
    # to within-link underestimation (per Step 1’s decomposition).
    # the adjustment must be done at the individual-couple level
    # the adjustment requires a previous comparison of the weights and of the groundtruth weight
    ''' 
    
    bucket_couple_targets = C_k.groupby(['u1','u2',bucket]).apply(lambda df: compute_targets(df))
    
    #COMPUTE THE IDEAL WEIGHT
    BCT_det = bucket_couple_targets.query('nlinks_detected > 0')
    BCT_det['w_gt'] = BCT_det['CV_detected_target']/BCT_det['CV_detected']
    BCT_det = BCT_det.reset_index()
    
    #COMPUTE THE WEIGHTS DERIVED BY IPW ON COVERAGE
    R_k = get_sparse_record_indicator(Date_range, k)
    coefs_bucket = compute_coef_aligned_timebucket(R_k, bucket)
    BCT_det['c1'] = BCT_det.apply(lambda row: coefs_bucket.loc[row[bucket]].loc[row['u1'], row['u1']], axis = 1)
    BCT_det['c2'] = BCT_det.apply(lambda row: coefs_bucket.loc[row[bucket]].loc[row['u2'], row['u2']], axis = 1)
    denom = BCT_det['c1'] * BCT_det['c2']
    BCT_det['w_ipw'] = 1 / denom.replace(0, np.nan)
    
    return BCT_det

###########################################
######### ANALYSIS MISSINGNESS  ###########
###########################################


def gen_gap_lims(Seq):
    '''
    create gap list from a sequence
    '''
    Ind_gaps = Seq.diff() == -1
    Ind_gaps_end = Seq.diff() == 1
    Date_start_gaps = list(Ind_gaps[Ind_gaps.values].index)  # Convert to list to allow insertions
    Date_end_gaps = list(Ind_gaps_end[Ind_gaps_end.values].index)  # Same for end gaps           
    if Seq.iloc[0] == 0:
        Date_start_gaps = [Seq.index[0]] + Date_start_gaps
    if Seq.iloc[-1] == 0:
        Date_end_gaps.append(Seq.index[-1])    
    return [ (ds,de) for ds,de in zip(Date_start_gaps, Date_end_gaps) ]


def gen_gaps_df(df_gaps):
    '''
    generate gap dataframe with information on starting hour and duration 
    df_gaps : input matrix of record indicators
    '''
    
    #gap dataframe indicator
    df_gh = df_gaps.copy()
    df_gh.columns = pd.to_datetime(df_gh.columns)

    #adding a last column (1hours exceeding the time-range) to account for border gaps
    last_col = df_gh.columns[-1]
    last_col_plus1 = last_col + dt.timedelta(hours=1)
    df_gh[last_col_plus1] = 1 

    #generating the list of gaps for each sequence
    df_gh['gaps'] = df_gh.apply(gen_gap_lims, axis=1)
    df_gh['sequence_index'] = df_gaps.index
    df_gh = df_gh[df_gh['gaps'].apply(len) > 0]
    df_id_gaps = df_gh[['sequence_index', 'gaps']].explode(['gaps'])
    df_id_gaps['start'] = df_id_gaps['gaps'].apply(lambda x : x[0])
    df_id_gaps['gap_duration_hours'] =  df_id_gaps['gaps'].apply(lambda A : (A[1]-A[0]).total_seconds()/3600)
    df_id_gaps = df_id_gaps[df_id_gaps['gap_duration_hours'] != 0]
    df_id_gaps['start_hour'] = df_id_gaps['start'].dt.hour 
    
    return df_id_gaps.set_index('sequence_index')


def filter_sequences(df_seq_ss):
    '''
    remove sequences which have a gap greater than 1 week (24*7 hours)
    
    '''
    df_gap_ss = gen_gaps_df(df_seq_ss)
    
    df_gap_ss_max = df_gap_ss.groupby(level = 'sequence_index').agg({'gap_duration_hours':'max'}) 
    
    df_gap_ss_overmax = df_gap_ss_max[df_gap_ss_max['gap_duration_hours'] > 24*7]
    
    df_seq_ss = df_seq_ss[~df_seq_ss.index.isin(df_gap_ss_overmax.index)]

    return df_seq_ss
    
#shuffle the gaps keeping the length
def shuffle_gaps_keep_durations(row):
    '''
    shuffle gaps keeping their durations
    '''
    
    # Step 1: Count runs of 0s and 1s
    counts = [(k, sum(1 for _ in g)) for k, g in itertools.groupby(row)]
    
    # Step 2: Split into even and odd index blocks
    c_even = counts[::2]
    c_odds = counts[1::2]
    
    # Step 3: Shuffle both lists
    np.random.shuffle(c_even)
    np.random.shuffle(c_odds)

    # Step 4: Randomize start
    if np.random.rand() < 0.5:
        first, second = c_even, c_odds
    else:
        first, second = c_odds, c_even

    # Step 5: Alternate elements
    count_shuffled = []
    min_len = min(len(first), len(second))
    for i in range(min_len):
        count_shuffled.append(first[i])
        count_shuffled.append(second[i])
    
    if len(first) > len(second):
        count_shuffled.append(first[-1])
    elif len(second) > len(first):
        count_shuffled.append(second[-1])

    # Step 6: Expand to full row
    row_shuffled = [val for val, count in count_shuffled for _ in range(count)]
    
    return row_shuffled

def add_feature_qrange(df_X):
    '''
    qrange is a feature 
    '''
    Q = (df_X ==0).sum(axis=1)/df_X.shape[1]
    Q_ranges =  np.arange(0,1.1,0.1)
    df_X['q_range'] = np.digitize(Q, Q_ranges)

#compute the overall entropy
def compute_entropy(df_gaps): 
    '''
    entropy computation over a dataframe of gaps
    df_gaps: datasets containing the duration of each gap for each sequence in the dataframe
    '''
    from scipy.stats import entropy

    col = 'gap_duration_hours'
    m_vals = df_gaps[col].values
    bins = np.arange(1, np.max(m_vals)+1)
    freqs = np.histogram(m_vals,bins)[0]/len(m_vals)
    return entropy(freqs)


###########################################
######### ANALYSIS CONTACTS  ###########
###########################################

def contact_share(df,
                  s = 'Data driven',
                  l = config.Levels[3],
                  wp = 'weekday'):
    '''
    from each weekperiod; compute the share of contacts over the hour of days
    '''
    
    feature = 'count_contacts'
    
    l_str = str(l)
    df_sl = df[(df['sparsity']==s) & (df['sparsity_level'] ==l_str)]
    df_slwp =  df_sl[(df_sl['weekperiod'] == wp)]
    df_slwp = df_slwp.groupby('hourofday').agg({feature:'mean'})
    df_slwp['contact_share'] = df_slwp[feature]/df_slwp[feature].sum()
    
    df_slwp = df_slwp.reset_index()
    df_slwp['sparsity'] = s
    df_slwp['sparsity_level'] = l_str
    df_slwp['weekperiod'] = wp
    
    return df_slwp

def stats_missing_users(df,
                  s = 'Data driven',
                  l = config.Levels[3],
                  wp = 'weekday'):
    '''
    from each weekperiod; compute the average percentage of missing users during the hour of days
    '''
    
    feature = 'missing_users_perc'
    
    l_str = str(l)
    df_sl = df[(df['sparsity']==s) & (df['sparsity_level'] ==l_str)]
    df_slwp =  df_sl[(df_sl['weekperiod'] == wp)]
    df_slwp = df_slwp.groupby('hourofday').agg({feature:'mean'})
    #df_slwp['contact_share'] = df_slwp[feature]/df_slwp[feature].sum()
    
    df_slwp = df_slwp.reset_index()
    df_slwp['sparsity'] = s
    df_slwp['sparsity_level'] = l_str
    df_slwp['weekperiod'] = wp
    
    return df_slwp


###################################
######### ANALYSIS EMO  ###########
###################################


def sem(x):
    """standard deviation of the mean (i.e., standard error)."""
    return np.std(x, ddof=1) / np.sqrt(len(x))
    
funcs = OrderedDict([
    ('whislo',  lambda x: x.quantile(0.025)),
    ('q1',      lambda x: x.quantile(0.25)),
    ('med',     lambda x: x.quantile(0.50)),
    ('q3',      lambda x: x.quantile(0.75)),
    ('whishi',  lambda x: x.quantile(0.975)),
    ('mean',  lambda x : np.mean(x)),
    ('std',   sem),         
])

def metric_stats(df_metrics, 
                 col = 'size_total',
                 funcs=funcs, 
                 df_cols = ['s','l','N_si','emv'],
                 th_final_size=0.05,
                 flatten=True):
    '''
    compute statistical indicators for a given epidemic metric
    '''
    
    agg = {f'{col}_{name}': (col, fn)
           for name, fn in funcs.items()}

    out = df_metrics.groupby(df_cols).agg(**agg)
    return out.reset_index() if flatten else out


def build_freq_table(
    df,
    index_cols,
    event_col):
    '''
    frequency tables for the epidemic metric of dynamic
    '''
    df_freq = (
        df.groupby(index_cols + [event_col])
          .size()
          .reset_index(name="count")
    )
    df_freq["freq"] = (
        df_freq.groupby(index_cols)["count"]
               .transform(lambda x: x / x.sum())
    )
    df_freq = (
        df_freq.pivot(index=index_cols, columns=event_col, values="freq")
               .fillna(0)
    )
    return df_freq

from sklearn.metrics import r2_score

def rsq(df, col1, col2, corr_type = 'pearson_r2'):
    if corr_type =='pearson_r2':
        return (df[col1].corr(df[col2]))
    else:
        return r2_score(df[col1], df[col2])

def _unify_limits(axes):
    # compute global min/max over all axes for both x and y
    xmin = min(ax.get_xlim()[0] for ax in axes.ravel())
    xmax = max(ax.get_xlim()[1] for ax in axes.ravel())
    ymin = min(ax.get_ylim()[0] for ax in axes.ravel())
    ymax = max(ax.get_ylim()[1] for ax in axes.ravel())
    for ax in axes.ravel():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

def rescale_to_bins(values, bins):
    """Linear rescale into bin-index space (uniform bins)."""
    
    values = np.asarray(values, dtype=float)
    bins   = np.asarray(bins, dtype=float)
    step   = np.diff(bins)
    
    if not np.allclose(step, step[0]):
        raise ValueError("For linear rescale, bins must be uniformly spaced.")
    return (values - bins[0]) / step[0]

def map_values_to_bincoords(values, edges, fractional=True):
    """
    Map real values to heatmap coordinates for (possibly non-uniform) bin edges.
    Returns continuous coordinates: i + t within each bin if fractional=True,
    otherwise bin centers (i + 0.5).
    """
    vals = np.asarray(values, dtype=float)
    out = np.full_like(vals, np.nan, dtype=float)

    m = np.isfinite(vals)
    if not m.any():
        return out

    idx = np.searchsorted(edges, vals[m], side='right') - 1
    idx = np.clip(idx, 0, len(edges) - 2)

    if fractional:
        left = edges[idx]
        right = edges[idx + 1]
        t = (vals[m] - left) / (right - left)
        t = np.clip(t, 0.0, 1.0)
        out[m] = idx + t
    else:
        out[m] = idx + 0.5
    return out



def _filter_df(df, 
               s, 
               hours_daytime, 
               weekday = None, 
               daytime = None):
    
    df_s = df[df['ss'] == s].copy()

    if daytime is not None:
        cond_daytime = df_s['hourofday'].isin(hours_daytime)
        df_s = df_s[cond_daytime if daytime else ~cond_daytime]

    if weekday is not None:
        cond_weekday = (df_s['weekperiod'] == 'weekday')
        df_s = df_s[cond_weekday if weekday else ~cond_weekday]

    return df_s

def _plot_panel(ax, 
                df, 
                level, 
                color, *, 
                logy=True, 
                hide_xticks=False, 
                hide_yticks=False):
    
    dict_level = subset_df_feature(df, 'level')
    df_sl = dict_level[str(level)]
    scatter_df(ax,
               df_sl,
               x='count_users',
               y='count_contacts',
               y_rename='',
               s=.1,
               c=color)
    if logy:
        ax.set_yscale('log')
    if hide_xticks:
        ax.set_xlabel('')
        remove_axis_ticktext(ax, axis='x')
    if hide_yticks:
        remove_axis_ticktext(ax, axis='y')

def _heatmap_count(ax, 
                   df, level, 
                   xbins, 
                   ybins, 
                   cmap = 'Reds', 
                   vmin = 0,
                   vmax = 150, 
                   log10 = False, 
                   cbar = True, 
                   plot_percentile = False): 

    dict_level = subset_df_feature(df, 'level')
    df_sl = dict_level[str(level)]
    
    kit_hm.visual_heatmap_2d(ax, 
                             df_sl, 
                             'count_users', 
                             'count_contacts', 
                             xbins, 
                             ybins,#bin_contacts, 
                             #xticks = bin_users,
                             #yticks = bin_contacts,
                             cmap = cmap,
                             normalize = False, 
                             log10 = log10, 
                             vmin = vmin, 
                             vmax = vmax, 
                             cbar = cbar) 
                             #cbar_orientation= 'horizontal')

    if plot_percentile:
        plot_binned_percentile(ax,
                               df_sl,
                               'count_users', 
                               'count_contacts', 
                               bin_users, 
                               bin_contacts, 
                               percentile=50,
                               color='black',
                               linestyle = '-',
                               linewidth = 2,
                               scatter=True,
                               scatter_size = 10,
                               use_weight = False, 
                               weight_col = "weight",
                               bin_non_linear=True,   # <--- NEW
                               _plot = False, 
                               no_binning = False)



def ax_colorbar_inset(ax):

    # clear axis
    ax.clear()
    
    # colormap + norm
    cmap = plt.get_cmap("Blues")
    norm = mpl.colors.Normalize(vmin=0, vmax=23)
    
    # horizontal colorbar inside this axis
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal'
    )
    
    # ----- DEFINE MAJOR / MINOR TICKS -----
    
    minor_ticks = np.arange(24)                      # 0..23
    major_ticks = np.array([0, 12, 23])       # labels only here
    
    cbar.set_ticks(major_ticks)                      # major ticks
    cbar.ax.set_xticks(minor_ticks, minor=True)      # add minor ticks
    
    # ----- SET TICK LABELS ONLY FOR MAJORS -----
    
    major_labels = [
        '8 am',      # hour 0
        '8 pm',     # hour 12
        '7 am'       # hour 23
    ]
    
    cbar.set_ticklabels(major_labels, size=10)
    
    # optional style
    #ax.set_xlabel('hour of day')
    restyle_ax(ax, label_size=ax_tick_size)
    
    # ensure horizontal alignment
    for label in cbar.ax.get_xticklabels():
        label.set_rotation(0)







