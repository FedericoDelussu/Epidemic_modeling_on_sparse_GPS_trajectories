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
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress detailed logs
import networkx as nx

import skmob
from sklearn.metrics import r2_score


from collections import Counter
from collections import OrderedDict

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.path import Path        # <-- Path defined here
from matplotlib.markers import MarkerStyle
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from . import config

#####################################
#### MOBILITY DATA PREPROCESSING ####
#################################k####

def preprocess_GPS_mobility_trajectory(traj, 
                                       tz = None):
    """
    INPUT: 
    > traj : dataframe for a single individual GPS trajectory 
        contains columns ['id', 'time_utc', 'lat', 'lon'] 
    > tz : timezone
    
    Preprocess GPS trajectory:
    - build TrajDataFrame
    - filter unrealistic speeds (max speed 100kmh)
    - convert timezone (optional)
    - downsample to 1-minute resolution
    """

    # Build TrajDataFrame (WGS84)
    traj_df = skmob.TrajDataFrame(
        traj.rename(columns={'lon': 'lng', 'time_utc': 'datetime'}),
        latitude='lat',
        longitude='lng',
        datetime='datetime',
        timestamp=True
    )

    #Remove speeds > 100 km/h
    traj_df = skmob.preprocessing.filtering.filter(traj_df, 
                                                   max_speed_kmh = 100)
    traj_df.rename(columns = {'lng':'lat', 
                              'id':'user_id'})

    #Parse timestamps as UTC
    dt = pd.to_datetime(traj_df['datetime'], utc = True)
    #convert to the local hour of the timezone tz
    dt = dt.dt.tz_convert(tz)
    dt = dt.dt.tz_localize(None)

    
    #Downsample to 1-minute resolution
    traj_df['datetime'] = dt.dt.floor('min', ambiguous = True)
    
    #Drop duplicates
    traj_df = traj_df.drop_duplicates(subset=['datetime'])

    return traj_df

#####################################################
#### CREATE SEQUENCE OF HOURLY RECORD INDICATORS ####
#####################################################

def collect_sequences(traj_sample, 
                      Study_period, 
                      Time_range):
    '''
    traj_sample : df with columns ['id', 'datetime']
    Study_period : list with start and end day of study period
    Time_range : list with start and end day of the broader time-range
    
    The function collect_sequences builds hourly record indicators (HRI) from GPS trajectories for 28-day sequences 
    which correspond to the Study_period [2014/02/08 - 2015/03/07] or to shifted sequences -- which start on 
    the same weekday of the first day of the  Study_period -- collected over the broader Time_Range [2014/02/01 - 2015/02/01].
    
    For each id, it marks presence of records (boolean 1 or 0) at every hour.
    (1) it scans the Time_range to find _candidate start days_ that:
        - Have the same weekday alignment with the start of the Study_period with a window of length of 28 days
        - For each valid start day, extracts an hourly sequence with the same temporal extent as the study period.
        - Reindexes each sequence to the Study_period hours, enabling stacking and comparison.
        
    (2) it concatenates all sequences into a single dataframe df_sequences with a MultiIndex (id, weekstep_index).
        id : user identifier 
        weekstep_index : Integer number of weeks between the candidate start day and the study period start. 
                         It identifies temporal shifts of a 28-day window across the Time_range. 
    '''

    print('collecting sequences')
    
    #[1] table of hourly record indicators (HRI)
    traj_sample_hour = traj_sample[['user_id','datetime']].copy()
    traj_sample_hour['datehour'] = pd.to_datetime(traj_sample_hour['datetime']).dt.floor('h')
    traj_sample_hour = traj_sample_hour[['user_id','datehour']].drop_duplicates()
    traj_sample_hour['values'] = 1
    HRI = traj_sample_hour.pivot(index = 'user_id',
                                 columns = 'datehour',
                                 values = 'values')

    #[2] Select the HRI within the time-range
    HRI = HRI.reindex(columns = pd.date_range(Time_range[0], Time_range[1],
                                              freq = 'h',
                                              inclusive = 'left'), fill_value=0)

    if Time_range == Study_period:
        #add weekstep_index column to ensure consistency in the dataframe generation pipeline
        HRI['weekstep_index'] = 0
        HRI = HRI.set_index('weekstep_index', append = True)
        HRI = HRI.fillna(0)
        return HRI
    
    #[3]search for temporal intervals which:
    Time_range_days = pd.date_range(Time_range[0], Time_range[1]) 
    Study_period_days = (Study_period[1] - Study_period[0]).days +1
    
    Time_range_days = [t for t in Time_range_days
                       if ((t - Study_period[0]).days % 7 == 0)  #step of 7 days
                       and (Time_range[1] -t).days > Study_period_days] #require same temporal extent of the study period 
    
    nweek_distances = [int((t-Study_period[0]).days/7) for t in Time_range_days] 
    dict_weeksteps = dict(zip(Time_range_days, nweek_distances))
    
    df_sequences = []
    
    for daystart, weekstep_index in dict_weeksteps.items():
        #print(daystart)
    
        #select the records within each time interval 
        HRI_ti = HRI[pd.date_range(daystart, 
                                   daystart + dt.timedelta(days = Study_period_days), 
                                   freq = 'h', 
                                   inclusive = 'left')].copy()
    
        #Assign for each time-interval the columns: sequence of hours within the study period 
        #in order to stack all the datasets generated in the for loop
        HRI_ti.columns = pd.date_range(Study_period[0], 
                                       Study_period[0] + dt.timedelta(days = Study_period_days), 
                                       freq = 'h', 
                                       inclusive = 'left')
        
        #Assign a column 'weekstep_index' in order to identify each time-interval
        HRI_ti['weekstep_index'] = weekstep_index
        df_sequences.append(HRI_ti)
    
    df_sequences = pd.concat(df_sequences, axis = 0).fillna(0)
    
    df_sequences = df_sequences.reset_index().set_index(['user_id','weekstep_index'])
    
    return df_sequences


##########################
##### STOP DETECTION  ####
##########################

def convert_coordinate_4326(lat, lon):
    '''
    function for conversion from crs:3857 (meters) to crs:4326 (degrees)
    '''
    
    latInEPSG4326 = (180 / math.pi) * (2 * math.atan(math.exp(lat * math.pi / 20037508.34)) - (math.pi / 2))
    lonInEPSG4326 = lon / (20037508.34 / 180)
    
    return latInEPSG4326, lonInEPSG4326

def convert_coordinate_3587(lat, lon):
    '''
    conversion from crs:4326 (degrees) to crs:3587 (meters)
    '''
    
    latInEPSG3857 = (math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)) * (20037508.34 / 180)
    lonInEPSG3857 = (lon* 20037508.34 / 180)
    
    return latInEPSG3857, lonInEPSG3857

def convert_df_coord_4326(df_loc, lat = 'lat', lon = 'lon'):
    '''
    convert df coordinate from crs:3857 (meters) to crs:4326 (degrees)    
    '''
    df_loc[[lat, lon]] = df_loc.apply(lambda row: convert_coordinate_4326(row[lat], row[lon]), axis=1, result_type='expand')

    
def convert_df_coord_3587(df_loc, lat= 'lat', lon = 'lon'): 
    '''
    convert df coordinate from crs:4326 (degrees) to crs:3587 (meters)    
    '''
    df_loc[[lat, lon]] = df_loc.apply(lambda row: convert_coordinate_3587(row[lat], row[lon]), axis=1, result_type='expand')

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
             dur_min = 10, 
             dt_max = 360, 
             delta_roam = 50):
    """
    INPUT : single individual GPS trajectory
    > traj : df with columns ['user_id','datetime','lat','lon']
        
    dur_min [minutes]   : float - minimum duration for a stay (stay duration).
    dt_max  [minutes]   : float - maximum duration permitted between consecutive pings in a stay. dt_max should be greater than dur_min 
    delta_roam [meters] : float - maximum roaming distance for a stay (roaming distance).

    Return stays: stop table of the GPS trajectory

    OUTPUT
    stays: stop table
    df with columns: ['start_time', 'end_time', 'medoid_x', 'medoid_y', 'diameter_m', 'n_pings', 'duration_s', 'geohash9']
    """

    #[STEP0] preprocess the trajectory
    #unix timestamp in seconds
    traj['unix_timestamp'] = pd.to_datetime(traj['datetime'], utc=True).astype('int64') // 10**9
    #convert coordinates to CRS:3587 (meters)
    convert_df_coord_3587(traj, lat = 'lat', lon = 'lon')
    traj = traj.rename(columns = {'lat': 'y',
                                  'lon': 'x'})

    #input coordinates for the Lachesis algorithm
    coords = traj[['x', 'y']].to_numpy()
    #initialize the collection of stays
    Stays = np.empty((0,6))
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


###############################
##### CONTACT ESTIMATION  #####
###############################

def filter_stops(df_stops, 
                 USERS_select, 
                 Date_range, 
                 Cols_select = None, 
                 reset_ghr = None):
    '''
    filter stop-table within a desired Date_range   
    '''
    
    if Cols_select is None:
        Cols_select = df_stops.columns
        
    #user selection
    df_stops_US = df_stops[Cols_select].loc[df_stops.user_id.isin(USERS_select)]
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
    returns simple stop table potentially contributing to contacts
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
    df_ghe = df_gh[['user_id','set']].explode('set')
    df_ghe['val']= 1

    #create indicator table of stop border
    df_th = df_ghe.pivot(index = 'set', columns = 'user_id', values = 'val')
    
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
    df_filtered = df_filtered.drop(columns='pair_set')#[['geohash']]#.values
    df_filtered = df_filtered.reset_index(drop=True)['geohash'].unique()

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

    Cols_select= ['user_id', 'stop_minute',geohash_col]

    for u in ['u1','u2']:
        df_contacts_shift = df_contacts_shift.merge(
            df_stops_NEIGHBOUR_1min[Cols_select],
            how='left',
            left_on=[u, 'date_time'],
            right_on=['user_id', 'stop_minute']
        ).rename(columns={geohash_col: f'{u}_geohash'}).drop(columns=['user_id', 'stop_minute'])
    
    return df_contacts_shift 

def compute_contact_marginal(df_stops_DR, 
                             ghr, 
                             time_resolution = '1minute'):
    '''
    compute marginal contacts from the stop table
    '''
    
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

    if time_resolution == '1minute':
        return df_ch_margin_unique

    if time_resolution == '1hour':
        df_ch_margin_unique['date_hour'] = df_ch_margin_unique['date_time'].dt.floor('h')
        Cols_select = ['u1','u2','u1_geohash','u2_geohash','date_hour']
        df_cmargin_hour = df_ch_margin_unique.groupby(Cols_select).size().reset_index()
        df_cmargin_hour.rename(columns = {0:'n_minutes'}, inplace = True)
        return df_cmargin_hour

def compute_tot_contacts(df_cm,df_cw):
    '''
    Joing the hourly within-cell and marginal contacts
    
    df_cw : within-cell contacts
    df_cm : marginal contacts
    '''
    Cs = ['u1','u2', 'date_hour', 'n_minutes']
    df_ctot = pd.concat([df_cm[Cs], df_cw[Cs]], axis=0)
    df_ctot = df_ctot.groupby(['u1','u2','date_hour'])['n_minutes'].sum().reset_index()
    
    return df_ctot    

def estimate_contacts(df_stops, 
                      ghr = 8, 
                      time_step = '1hour',
                      Date_range = None,
                      return_within_marginal = False):
    '''
    INPUT 
    > df_stops dataframe
    Consists of the stop tables for the sample of students in the DTU Campus data
    Contains columns ['start_time', 'end_time', 'medoid_x', 'medoid_y', 'diameter_m', 'n_pings', 'duration_s', 'geohash8']  

    > ghr: geohash resolution for contact estimation
    > Date_range : selected date range for contact estimation
        - For reasons of computational cost, it is suggested to use a Date_range of the order of 1 day.   
    
    OUTPUT 
    > df_contacts
    df with columns: ['u1', 'u2', 'date_hour', 'n_minutes']
    Each record reports a contact duration between user 'u1' and 'u2' on hour 'date_hour' for 'n_minutes' minutes
    
    In our experiment we perform contact estimation for each day and save the contacts as a collection of daily csv files, 
    we save separately within-cell and marginal contacts
    '''

    #[0] filter temporally the stop table within the desired Date_range
    if Date_range is not None:
        df_stops = filter_stops(df_stops,
                                USERS_select = list(df_stops['user_id'].unique()),
                                Date_range = Date_range)

    if len(df_stops) == 0:
        return pd.DataFrame(columns=['u1', 'u2', 'date_hour', 'n_minutes'])

    #set the geohash resolution for contact estimation
    df_stops = df_stops.rename(columns = {'geohash9': 'geohash'})
    df_stops['geohash'] = df_stops['geohash'].str[:ghr]

    #explode temporally the stop table
    df_stops['unique_stop_id'] = range(len(df_stops))
    df_stops['stop_hour'] = [list(pd.date_range(start=r.start_time.floor('h'),
                                                end=r.end_time.floor('h'),
                                                freq='h'))
                             for r in df_stops.itertuples()]
    df_stops = df_stops.explode('stop_hour').drop_duplicates(['geohash', 'stop_hour', 'unique_stop_id'])
    
    #select stops populated by more than 1 user
    #potentially contributing to the contact estimation
    df_stops_CONTACT = get_stops_CONTACT(df_stops) 

    #WITHIN CELL CONTACTS (uses df_stops_CONTACT)
    df_cwithin = compute_contact_table(df_stops_CONTACT, 
                                       geohash_col = 'geohash', 
                                       time_resolution = time_step)
    
    #MARGINAL CONTACTS (uses df_stops)
    df_cmargin = compute_contact_marginal(df_stops, 
                                          ghr, 
                                          time_resolution = time_step)

    if return_within_marginal:
        return df_cwithin, df_cmargin

    if time_step == '1minute':
        _cols_select = ['date_time','u1','u2']
        return pd.concat([df_cwithin[_cols_select], df_cmargin[_cols_select]], axis = 0)

    if time_step == '1hour':
        return compute_tot_contacts(df_cmargin, df_cwithin)    
   
def estimate_daily_contacts(df_stops, 
                            Study_period, 
                            ghr = 8,
                            time_step = '1hour'):
    '''
    given a stop table; estimates daily contacts over a study period of interest
    
    df_stops : stop location table
    Study_period: list with start and end day of study period for contact estimation
    '''
    #collect contacts for each day of a given study period
    DICT_contacts = {}
    for day in pd.date_range(Study_period[0], Study_period[1]):
        df_contacts_day = estimate_contacts(df_stops,
                                            ghr = 8, 
                                            time_step = '1hour',
                                            Date_range = [day, day + dt.timedelta(days=1)]) 
        DICT_contacts[day] = df_contacts_day
        
    return DICT_contacts


###################################################
##### GROUND TRUTH EPIDEMIC MODELING OUTCOMES #####
###################################################

def get_complete_users(df): 
    '''
    df: sequence of hourly record indicators
    '''
    #select complete users which have more than 95% of hours over the study period ([0-5]% missing hours)
    df = df[df.index.get_level_values("weekstep_index") == 0]
    df_users_complete = df[df.mean(axis=1) >= 0.95]
    USERS_select = list(df_users_complete.index.get_level_values("id").unique())
    return USERS_select

#import the groundtruth contacts for epidemic modeling
def get_groundtruth_contacts(df,
                             FOLD_groundtruth_contacts, 
                             Study_period):
    '''
    df: sequence of hourly record indicators
    FOLD_contact: folder where the daily contacts are saved
    Study_period: list with start and end day of study period 
    '''

    USERS_select = get_complete_users(df)
    
    #collection of days 
    Date_range = pd.date_range(Study_period[0], Study_period[1])
    
    df_contacts = []
    for D0 in Date_range:
        
        #select a specific date
        D0 = str(D0.date())
        
        #import within and marginal contacts
        df_D0_cw = pd.read_csv(f'{FOLD_groundtruth_contacts}df_cwithin_{D0}.csv')
        df_D0_cm = pd.read_csv(f'{FOLD_groundtruth_contacts}df_cmargin_{D0}.csv')
        
        #join all contacts
        df_D0_contacts = compute_tot_contacts(df_D0_cm, df_D0_cw)
        c_u1 = df_D0_contacts['u1'].isin(USERS_select)
        c_u2 = df_D0_contacts['u2'].isin(USERS_select)
        df_D0_contacts = df_D0_contacts[c_u1 & c_u2]
        df_contacts.append(df_D0_contacts)
        
    df_contacts = pd.concat(df_contacts, axis = 0)
    
    return df_contacts, USERS_select, Date_range

def subset_df_feature(df,f):
    '''
    df : dataframe
    f  : feature name 
    '''
    #feature unique values 
    f_vals = df[f].unique()
    #subset dataframe according to feature f records 
    return {v: df[df[f]==v] for v in f_vals}

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

    W = subset_df_feature(df_contact_daily, 'date')
    W = {D:to_dense_sym(W_D, USERS_select) for D, W_D in W.items()}

    #clipping to maximum number of minutes in a day
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
                    Dates,
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
    
    for D0 in Dates:
        sample_transition(X, X_d, W[D0], pars, gamma_daily = gamma_daily)
        ts_SI.append(X.sum(axis=0))

    return np.array(ts_SI)
    
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

def iter_epid_simulation(W, 
                         Date_range,
                         USERS_select, 
                         pars, 
                         p_init = 0.05, 
                         n_init = None,
                         N_iter=100, 
                         gamma_daily = False):
    '''
    W: dictionary of contact data
    pars : (beta, gamma) at minute resolution 
    p_init : fraction of initial infected
    '''

    COLLECT_simulations = []
    COLLECT_epid_metrics = []
    
    for i in range(N_iter):
       
        #compute the time-series of suscpetible-infected and R0
        ts_SI_R0 = epid_simulation(W, 
                                   USERS_select,
                                   pars = pars, 
                                   p = p_init, 
                                   n_init = n_init,
                                   Dates = Date_range,
                                   seed_number = None, 
                                   gamma_daily = gamma_daily)
        
        #compute the epidemiological metrics 
        epid_metrics = compute_epid_metrics(ts_SI_R0)
        
        COLLECT_simulations.append(ts_SI_R0)    
        COLLECT_epid_metrics.append(epid_metrics)

    COLLECT_simulations = np.array(COLLECT_simulations)
    COLLECT_epid_metrics = np.array(COLLECT_epid_metrics)

    return COLLECT_simulations, COLLECT_epid_metrics

def epidemic_modeling(df_contacts, 
                      USERS_select, 
                      dict_epid_pars,
                      Date_range,
                      N_iter):
    '''
    INPUT
    df_contacts  : df with columns ['u1', 'u2', 'date_hour', 'n_minutes']
    USERS_select : list of unique users in the contact dataframe
    dict_epid_pars : dict with values:
        - beta : probability of infection given 1-minute contact
        - gamma : probability of recovery over 1 day
        - seed size : number of initial infected of the SIR model
    Date_range : ordered list of consecutive days of the simulation
    N_iter: number of iterations of the epidemic simulation 

    OUTPUT
    [1] epid_curves 
        array of shape (N_iter, #simulation days, 2)
            epid_curves[i] has as columns the count of Susceptibles and Infected
            
    [2] epid_metrics (computed metrics from epid curves)
        array of shape (N_iter, 5)
    '''

    #convert the contacts as an ordered list of daily contact matrices
    W = gen_contact_daily(df_contacts, USERS_select)
    
    #[1] Launch the epidemic simulation
    epid_pars = (dict_epid_pars['beta'], dict_epid_pars['gamma'])
    n_init = dict_epid_pars['n_init']
    
    #[1.2] Collect the simulation outcomes
    #epid_curves : daily time-series of Susceptible and Infected 
    #epid_metrics: associated epidemic metrics computed from epid_curves
    epid_curves, epid_metrics = iter_epid_simulation(W, 
                                                     Date_range,
                                                     USERS_select, 
                                                     epid_pars, 
                                                     n_init = n_init, 
                                                     N_iter = N_iter,
                                                     gamma_daily = True)
    
    return epid_curves, epid_metrics

############################################################
##### SPARSIFICATION AND ESTIMATION OF BIASED CONTACTS #####
############################################################

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

def gen_mask(level, 
             df_seq,
             df_seq_complete):
    '''    
    level: sparsity level
    df_seq : dataframe of hourly record indicators (employed for sparsification of the complete trajectories)
        index has to be the id of the sequence 
    df_seq_complete: dataframe of hourly record indicators of the complete users
        index has to be the id of the user
        
    Generate mask for sparsification of the complete user trajectories within a sparsity level 
    '''

    #ensure that there are no duplicated indexes for the sample of all sequences
    df_seq.index = range(len(df_seq))
    
    #compute the sparsity (fraction of missing hours) for each sequence
    seq_sparsity = (1- df_seq.mean(axis=1)).sort_values()
    seq_complete_sparsity = (1- df_seq_complete.mean(axis=1)).sort_values()

    #reorder the sequences based on the sparsity levels
    df_seq = df_seq.loc[seq_sparsity.index]
    df_seq_complete = df_seq_complete.loc[seq_complete_sparsity.index]
    
    #bucket the complete users based on intervals of length .01 
    buckets, buckets_count = np.unique(np.floor(seq_complete_sparsity.values*100)/100 + 0.01, 
                                            return_counts=True)
    #iterative sampling conditioned on the sparsity level of the complete trajectories
    lev_mask = []
    for v,c in zip(buckets, buckets_count):

        #this condition ensures that the selected sparse trajectories
        v_indexes = seq_sparsity[seq_sparsity.between(level[0], level[1] - v)].index
        
        #the sampling of trajectories in df_seq is conditional on the sparsity of the complete users
        #such that the each sampled sparse sequence -- when added to the complete trajctory -- 
        #results in a sparsified trajectory which falls within the desired sparsity level
        v_indexes_sampled = np.random.choice(v_indexes, size = c, replace = True)
        v_gaps_sampled = df_seq.loc[v_indexes_sampled]
        lev_mask.append(v_gaps_sampled)
        
    lev_mask = pd.concat(lev_mask, axis=0)
    
    #re-indexing corresponding to the id of the complete users
    #in order to have correct association when implementing the sparsification
    lev_mask.index = df_seq_complete.index
    
    return lev_mask

#shuffle the gaps keeping the length
def shuffle_gaps_keep_durations(row):
    '''
    shuffle gaps in a sequence keeping their durations
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
        count_shuffled.extend(first[min_len:])
    elif len(second) > len(first):
        count_shuffled.extend(second[min_len:])

    # Step 6: Expand to full row
    row_shuffled = [val for val, count in count_shuffled for _ in range(count)]
    
    return row_shuffled

def get_complete_users(df_seq):
    '''
    df_seq : dataframe of hourly record indicators (employed for selection of the complete trajectories)
    return the list of complete users
    '''
    df_seq_w0 = df_seq[df_seq.index.get_level_values("weekstep_index") == 0]
    df_seq_complete = df_seq_w0[df_seq_w0.mean(axis=1) >= 0.95]
    USERS_complete = list(df_seq_complete.index.get_level_values('user_id'))
    return USERS_complete 
    
def gen_sparsification_masks(_df_seq,
                             Levels):
    '''
    df_seq : dataframe of hourly record indicators (employed for sparsification)
    Levels : list of sparsity levels

    create a collection of sparsification masks
    for each sparsity level
    for sparsification approacches: ['Data_driven', 'Random_shuffling', Random_uniform']   
    '''
    
    #dataframe of hourly record indicators (employed for sparsification of the complete trajectories)
    df_seq= _df_seq.copy()
    
    #remove sequences which have gaps larger than 1week
    df_seq = filter_sequences(df_seq)
    
    #[2] select the complete users
    df_seq_w0 = df_seq[df_seq.index.get_level_values("weekstep_index") == 0]
    #dataframe of hourly record indicators of the complete users
    df_seq_complete = df_seq_w0[df_seq_w0.mean(axis=1) >= 0.95]
    df_seq = df_seq.droplevel('weekstep_index')
    df_seq_complete = df_seq_complete.droplevel('weekstep_index')
    
    #Data-driven sparsification masks
    DICT_mask = {level: gen_mask(level, 
                                 df_seq, 
                                 df_seq_complete) for level in Levels}

    #Random baseline : shuffling gaps keeping durations
    DICT_mask_random_shuffling = {l : m.apply(lambda row: pd.Series(shuffle_gaps_keep_durations(row.values), index=row.index), axis=1)
                                        for l,m in DICT_mask.items()}

    #Random baseline: shuffling record indicators -- record missing uniformly at random
    DICT_mask_random_uniform =  {l : m.apply(lambda row: pd.Series(np.random.permutation(row.values), index=row.index), axis=1)
                                        for l,m in DICT_mask.items()}

    DICT_masks = {'Data_driven'     :      DICT_mask,
                  'Random_shuffling': DICT_mask_random_shuffling, 
                  'Random_uniform'  : DICT_mask_random_uniform}

    return DICT_masks

def unpivot_mask(df_mask):
    '''
    unpivot the gap mask for sparsification 
    '''

    v_name = 'date_hour'
    df_mask_reset = df_mask.reset_index()
    df_unpivoted = pd.melt(df_mask_reset, id_vars='user_id', var_name= v_name, value_name='value')
    df_unpivoted[v_name] = pd.to_datetime(df_unpivoted[v_name])
    df_unpivoted = df_unpivoted.sort_values(by = ['user_id', v_name])
    
    return df_unpivoted

def from_mask_to_record_indicator(df_mask, t_res = 'minute'):
    '''
    converts a gap mask to a record indicator
    '''
    
    df_mask_up = unpivot_mask(df_mask.copy())
    df_record_select = df_mask_up[df_mask_up['value'] == 1].drop('value', axis=1)
    
    if t_res == 'minute':
        #explode the dataset at the minute resolution in order to match the trajectory records
        f = lambda x : pd.date_range(x['date_hour'], x['date_hour'].replace(minute = 59), freq='min') 
        df_record_select['datetime'] = df_record_select.apply(f, axis =1)
        df_record_select = df_record_select[['user_id','datetime']].explode('datetime')
        
    return df_record_select

def sparsification_pipeline(traj_complete,
                            DICT_masks,
                            sparsity_approach,
                            level,
                            FOLD_iter=None,
                            file_prefix=None):
    '''
    INPUT traj_complete dataframe
        Consists of the complete GPS location trajectories (0-5% missing hours) during the study period 2014-2-10 to 2014-3-7
        it contains columns ['user_id', 'datetime', 'lat', 'lon']
        traj_complete : complete GPS trajectories
        DICT_masks : nested dict containing the sparsficiation masks
            (see section 6.1 Creation of the hourly record indicator mask for sparsification)
        sparsity_approach : can take values ['Data_driven', 'Random_shuffling', 'Random_uniform']
        level : range of missing hours from config.Levels
        FOLD_iter : optional output folder (iteration root). If provided, saves mask,
            traj_complete_sparsified, df_stops and df_contacts to
            FOLD_iter/<sparsity_approach>/<level>/

    OUTPUT DICT_daily_contacts dict of type (dt.datetime, df_contact)
        for each day it assigns the dataframe of hourly contacts on that day
        dataframe contains columns ['u1', 'u2', 'date_hour', 'n_minutes']
    '''

    #[1] SPARSIFICATION
    #selection of the mask
    mask = DICT_masks[sparsity_approach][level]
    #unpivoting the mask for peforming sparsification
    df_record_select = from_mask_to_record_indicator(mask,t_res = 'minute')
    #adding the gaps of the mask into the complete trajectory
    traj_complete_sparsified = pd.merge(df_record_select,
                                        traj_complete,
                                        on = ['user_id', 'datetime'],
                                        how = 'left').dropna(subset=['lat', 'lon'])

    #[2] STOP DETECTION
    #lachesis parameters
    dur_min    = 10
    dt_max     = 360
    delta_roam = 50
    #run stop ddetection for each id trajectory
    df_stops = traj_complete_sparsified.groupby('user_id').apply(lambda x: lachesis(x, dur_min, dt_max, delta_roam))
    if isinstance(df_stops.index, pd.MultiIndex):
        df_stops = df_stops.reset_index(level=1, drop=True).reset_index()
    else:
        df_stops = pd.DataFrame(columns=['user_id', 'start_time', 'end_time',
                                         'medoid_x', 'medoid_y', 'diameter_m',
                                         'n_pings', 'duration_s', 'geohash9'])

    #[3] CONTACT ESTIMATION
    #estimate daily contacts for each day in the study period
    #contact records have hour resolution and report the number of minutes in contact
    DICT_daily_contacts = estimate_daily_contacts(df_stops,
                                                  config.Study_period,
                                                  ghr = 8,
                                                  time_step = '1hour')

    #[4] OPTIONAL SAVE
    if FOLD_iter is not None:
        if file_prefix is not None:
            prefix = f'{FOLD_iter}{file_prefix}_'
            mask.to_csv(f'{prefix}mask.csv')
            traj_complete_sparsified.to_csv(f'{prefix}traj_complete_sparsified.csv', index=False)
            df_stops.to_csv(f'{prefix}df_stops.csv', index=False)
            df_contacts = pd.concat(DICT_daily_contacts.values(), axis=0)
            df_contacts.to_csv(f'{prefix}df_contacts.csv', index=False)
        else:
            FOLD_level = f'{FOLD_iter}{sparsity_approach}/{level}/'
            mask.to_csv(f'{FOLD_level}mask.csv')
            traj_complete_sparsified.to_csv(f'{FOLD_level}traj_complete_sparsified.csv', index=False)
            df_stops.to_csv(f'{FOLD_level}df_stops.csv', index=False)
            df_contacts = pd.concat(DICT_daily_contacts.values(), axis=0)
            df_contacts.to_csv(f'{FOLD_level}df_contacts.csv', index=False)


###################################################
##### CONTACT CORRECTION – WEIGHT COMPUTATION #####
###################################################

#compute hourofday-weekperiod for a datetime series
def gen_hw_col(v):
    '''
    gen hourofday_weekday column
    '''
    v = pd.to_datetime(v)
    #index of weekday days (from Monday to Friday)
    _wdays = [0,1,2,3,4,5]
    _wperiod = np.array(['weekend','weekday'])
    return [f"{t}_{_wperiod[w*1]}" for t,w in zip(v.hour, v.weekday.isin(_wdays))] 

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

def compute_contact_correction_weights(df_contacts_biased,
                                       df_hri_sparsified):
    '''
    INPUTS 
    [1] df_contacts_biased dataframe 
    Cotains columns: ['u1', 'u2', 'date_hour', 'n_minutes']
    Each record documents a contact between user 'u1' and 'u2' during hour 'date_hour' for 'n_minutes' minutes
    
    [2] df_hri_sparsified dataframe
    hourly record indicator of the complete sparsified trajectories employied for computing df_contacts_biased
    has as columns the sequence of hours in the study period
    has index 'id'
    
    OUTPUT
    > df_contacts_biased with new columns:   
        hourofday_weekday : string indicating the hourofday (from 0 to 23) and the weekperiod (weekday or weekend)
        coverage_u1 : coverage of user 'u1' during the specific hourofday_weekday over the study period
        coverage_u2 : coverage of user 'u2' during the specific hourofday_weekday over the study period
        weight : 1/(coverage_u1*coverage_u2)  
    > coefs : coefficients  
    '''
    
    #compute the individual-level coverage matrix for each time bucket
    #the diagonal elements of this matrix contain the individual coverage level 
    #for each specific time bucket (hourofday_weekday)
    coefs = compute_coef_aligned_timebucket(df_hri_sparsified.T, 
                                            'hourofday_weekday')
    
    
    #assign each user to the coverage coefficients bucketed by hourofday weekday
    df_contacts_biased['hourofday_weekday'] = gen_hw_col(df_contacts_biased['date_hour'].values)
    df_contacts_biased['coverage_u1']  = df_contacts_biased.apply(lambda row: coefs.loc[row['hourofday_weekday']].loc[row['u1'], row['u1']], axis = 1) 
    df_contacts_biased['coverage_u2']  = df_contacts_biased.apply(lambda row: coefs.loc[row['hourofday_weekday']].loc[row['u2'], row['u2']], axis = 1) 
    
    #compute the weight according to the IPW formula: 1/(p_i*p_j)
    #where p_i is the probability of observing one record on the given time-bucket for user i 
    df_contacts_biased['weight'] = 1/(df_contacts_biased['coverage_u1']*df_contacts_biased['coverage_u2'])

    return df_contacts_biased, coefs
    

#################################
##### CALIBRATION MODELING  #####
#################################

def compute_objective_function(params, 
                               df_contacts, 
                               USERS_select,
                               Date_range, 
                               N_iter,
                               Curve_ref):
    '''
    params : (beta, gamma, seedsize) parameters of the SIR model
    df_contacts : dataframe of contacts [u1,u2, datetime, n_minutes]
    USERS_select : list of users id in the contact dataframe
    Date_range : (dt.datetime.date) array of consecutive days for running the epidemic simulation 
    N_iter : number of iteration of the epidemic simulation

    Curve_ref : reference curve for calibration 
        computed as the median of infected over the ensemble of groundtruth curves

    return the RMSE between Curve_ref and the median of the simulations using df_contacts and params
    '''
    
    beta, gamma, seedsize = params

    #convert the contacts as an ordered list of daily contact matrices
    W = gen_contact_daily(df_contacts, USERS_select)

    #launch the epidemic simulations N_iter times
    Sims, _ = iter_epid_simulation(W, 
                                   Date_range,
                                   USERS_select, 
                                   pars = (beta, gamma), 
                                   n_init = seedsize,
                                   N_iter = N_iter)
   
    #compute the Median of infected over the ensemble of simulations
    Median_infected = np.median(Sims[:,:,1], axis = 0) 

    #compute the score as the RMSE between the reference curve (Curve_ref) and 
    RMSE = np.sqrt(np.mean((Curve_ref - Median_infected)**2))

    return RMSE

class Objective_param_search:
    

    def __init__(self, 
                 df_contacts, 
                 USERS_select,
                 Date_range, 
                 N_iter,
                 Curve_ref,
                 GRID_stats):
        '''
        df_contacts : dataframe of contacts [u1,u2, datetime, n_minutes]
        USERS_select : list of users id in the contact dataframe
        Date_range : (dt.datetime.date) array of consecutive days for running the epidemic simulation 
        N_iter : number of iteration of the epidemic simulation
        Curve_ref : reference curve for calibration 
            computed as the median of infected over the ensemble of groundtruth curves
        GRID_stats : parameter boundaries 
            dataframe with columns (beta, gamma, n_init) and rows (min, max)
        '''
        
        self.df_contacts  = df_contacts
        self.USERS_select = USERS_select
        self.Date_range   = Date_range
        self.N_iter       = N_iter
        self.Curve_ref    = Curve_ref
        self.GRID_stats   = GRID_stats

    def __call__(self, trial):

        GRID_stats = self.GRID_stats
        beta = trial.suggest_float("beta", GRID_stats.loc["min", "beta"], GRID_stats.loc["max", "beta"])
        gamma = trial.suggest_float("gamma", GRID_stats.loc["min", "gamma"], GRID_stats.loc["max", "gamma"])
        seedsize = trial.suggest_int("seedsize", int(GRID_stats.loc["min", "seedsize"]), int(GRID_stats.loc["max", "seedsize"]))
        
        params = (beta, gamma, seedsize) 
        
        Objective_param_search = compute_objective_function(params, 
                                                            self.df_contacts, 
                                                            self.USERS_select,
                                                            self.Date_range,
                                                            self.N_iter,
                                                            self.Curve_ref)
        
        return Objective_param_search

def optuna_param_search(df_contacts_biased, 
                        USERS_select,
                        Date_range, 
                        N_iter,
                        Curve_ref,
                        GRID_stats, 
                        n_trials,
                        show_progress_bar = True):
    '''
    INPUTS 
    df_contacts : dataframe of contacts [u1, u2, datetime, n_minutes]\
        Each record reports a contact between user 'u1' and 'u2' during hour 'date_hour' for 'n_minutes' minutes
    USERS_select : list of users id in the contact dataframe
    Date_range : (dt.datetime.date) array of consecutive days for running the epidemic simulation 
    N_iter : number of iteration of the epidemic simulation

    Curve_ref : reference curve for calibration 
        computed as the median of infected over the ensemble of groundtruth curves
        
    GRID_stats : parameter boundaries 
        dataframe with columns (beta, gamma, n_init) and rows (min, max)
        
    n_trials : number of trials of the optimization algorithm 

    returns 
    > best_params: dict with keys 'beta', 'gamma', 'seedsize'
        optimal parameters generated by the calibration algorithm
    > best_value: score of the parameter search
    '''

    #minimize the RMSE
    study = optuna.create_study(direction="minimize") 
        
    Obj = Objective_param_search(df_contacts_biased, 
                                 USERS_select,
                                 Date_range, 
                                 N_iter,
                                 Curve_ref,
                                 GRID_stats)

    study.optimize(Obj, 
                   n_trials = n_trials, 
                   show_progress_bar = show_progress_bar) 
    
    return study.best_params, study.best_value

def epid_modeling(df_contacts,
                  USERS_select,
                  Date_range,
                  N_iter,
                  groundtruth_params,
                  modeling_type,
                  Curve_ref=None,
                  GRID_stats=None,
                  n_trials=None,
                  show_progress_bar=True):
    '''
    Epidemic modeling function for Oracle and Calibration workflows.

    df_contacts       : dataframe of contacts [u1, u2, date_hour, n_minutes]
    USERS_select      : list of user ids
    Date_range        : (datetime.date) array of consecutive days for the simulation
    N_iter            : number of iterations of the epidemic simulation
    groundtruth_params: dict with keys 'beta', 'gamma', 'seedsize'
    modeling_type     : 'Oracle' or 'Calibration'

    Oracle-only args  : (ignored when modeling_type == 'Calibration')
        (none beyond the common ones)

    Calibration-only args:
    Curve_ref         : reference infected curve (median of groundtruth ensemble)
    GRID_stats        : parameter boundaries — DataFrame with columns (beta, gamma, n_init)
                        and rows (min, max)
    n_trials          : number of Optuna optimization trials
    show_progress_bar : whether to display the Optuna progress bar

    Returns
    -------
    Oracle      : epid_curves, epid_metrics
    Calibration : epid_curves, epid_metrics, best_params, best_values
    '''

    if modeling_type == 'Oracle':
        W = gen_contact_daily(df_contacts, USERS_select)
        epid_curves, epid_metrics = iter_epid_simulation(
            W,
            Date_range,
            USERS_select,
            (groundtruth_params['beta'], groundtruth_params['gamma']),
            n_init=groundtruth_params['seedsize'],
            N_iter=N_iter,
            gamma_daily=True)
        return epid_curves, epid_metrics

    elif modeling_type == 'Calibration':
        best_params, best_values = optuna_param_search(
            df_contacts,
            USERS_select,
            Date_range,
            N_iter,
            Curve_ref,
            GRID_stats,
            n_trials,
            show_progress_bar=show_progress_bar)

        W = gen_contact_daily(df_contacts, USERS_select)
        epid_curves, epid_metrics = iter_epid_simulation(
            W,
            Date_range,
            USERS_select,
            (best_params['beta'], best_params['gamma']),
            n_init=best_params['seedsize'],
            N_iter=N_iter,
            gamma_daily=True)
        return epid_curves, epid_metrics, best_params, best_values

    else:
        raise ValueError(f"modeling_type must be 'Oracle' or 'Calibration', got '{modeling_type}'")


###################
##### OTHERS  #####
###################

def read_folder_files(folder_path,
                      f_name=None,
                      Cols_select=None,
                      FILES_select=None,
                      parse_dates_list=None,
                      index_col=None,
                      dtype = None,
                      Date_range = None,
                      n_workers = 1,
                      chunksize = 2000):
    '''
    Read a collection of CSV files and returns a single dataframe

    folder_path : path of the folder collecting the CSV dataframes
    f_name : adds a new column indicating the name of each CSV file
    Cols_select: list of selected columns for each csv
    FILES_select : list of files to read (if None reads all files in the folder)
    parse_dates_list : list of dates in the csv columns to be processed as datetimes
    Date_range : filters the dataset based on the datetime columns
    n_workers : number of threads for parallel reading (default: 1 = sequential)
    chunksize : rows per chunk when Date_range filtering is active (default: 2000)
    '''
    FILES = os.listdir(folder_path)
    if FILES_select is not None:
        FILES = FILES_select
    FILES = [f for f in FILES if f.endswith('.csv')]

    n_files = len(FILES)
    print(f'  reading {n_files} files (n_workers={n_workers})...')
    completed = [0]

    def _read_one(file_name):
        if Date_range is not None:
            chunks = pd.read_csv(os.path.join(folder_path, file_name),
                                 usecols=Cols_select,
                                 index_col=index_col,
                                 parse_dates=parse_dates_list,
                                 dtype=dtype,
                                 chunksize=chunksize)
            df = pd.concat(
                chunk[pd.to_datetime(chunk.datetime, utc=True).dt.tz_convert(None).between(Date_range[0], Date_range[-1], inclusive='left')]
                for chunk in chunks
            )
        else:
            df = pd.read_csv(os.path.join(folder_path, file_name),
                             usecols=Cols_select,
                             index_col=index_col,
                             parse_dates=parse_dates_list,
                             dtype=dtype)
        if f_name is not None:
            df[f_name] = file_name.split('.csv')[0]
        completed[0] += 1
        print(f'  [{completed[0]}/{n_files}] {file_name} ({len(df)} rows)', flush=True)
        return df

    if n_workers > 1:
        from multiprocessing.pool import ThreadPool
        with ThreadPool(n_workers) as pool:
            frames = pool.map(_read_one, FILES)
    else:
        frames = [_read_one(f) for f in FILES]

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

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

def sort_df_columns(df, ascending = False):
    '''
    reorder a dataframe columns by the sum over the row-dimension
    ''' 
    Cols_ordered = df.sum(axis=0).sort_values(ascending = ascending).index
    return df[Cols_ordered]

def sort_df_rows(df, ascending = False):
    # Calculate the average of each row (excluding the index or non-numeric columns)
    row_sums = df.sum(axis=1)
    # Sort the DataFrame rows based on the row averages
    df_sorted = df.loc[row_sums.sort_values(ascending = ascending).index]
    return df_sorted

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

##############
#LATEX TABLES#
##############

def _sanitize_str(x: str) -> str:
    x = x.replace("±", r"$\pm$")
    x = x.replace(" ± ", r" $\pm$ ")
    x = x.replace("%", r"\%")
    return x

def _sanitize_labels(labels, names=None):
    """Sanitize a (Multi)Index's labels and names."""
    if isinstance(labels, pd.MultiIndex):
        new_tuples = []
        for tpl in labels.tolist():
            new_tuples.append(tuple(_sanitize_str(str(v)) for v in tpl))
        new_names = [(_sanitize_str(str(n)) if n is not None else None) for n in labels.names]
        return pd.MultiIndex.from_tuples(new_tuples, names=new_names)
    else:
        new_vals = [_sanitize_str(str(v)) for v in labels.tolist()]
        new_name = _sanitize_str(str(labels.name)) if labels.name else None
        return pd.Index(new_vals, name=new_name)

def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # remove None/NaN -> ""
    df = df.replace({None: ""})
    df = df.replace({np.nan: ""})
    # sanitize cell strings
    df = df.map(lambda v: _sanitize_str(v) if isinstance(v, str) else v)
    # sanitize columns and index (labels + names)
    df.columns = _sanitize_labels(df.columns)
    df.index   = _sanitize_labels(df.index)
    return df

def to_tabular(df: pd.DataFrame, column_format: str) -> str:
    df = _sanitize_df(df)
    # escape=False because we’ve already escaped/sanitized
    return df.to_latex(escape=False,
                       multicolumn=True,
                       multicolumn_format='c',
                       column_format=column_format,
                       index = True)


#####################################
### COMPUTING REPRODUCTIVE NUMBER ###
#####################################


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

def visual_ax_dates(ax, 
                     Dates, 
                     visual_saturday = False, 
                     tick_step = 1, 
                     rot = 90, 
                     axis = 'x', 
                     date0 = False):
    
    Dates_md = pd.to_datetime(Dates).strftime("%-d %b")

    Dates_ticks = Dates_md
    if date0:
        Dates_ticks = [Dates[0].date()] + list(Dates_md[1:])

    if visual_saturday:
        Dates_ticks = ['Sat ' + dt if d.weekday() == 5 else dt for dt, d in zip(Dates_ticks, Dates)]
        for i,d in enumerate(Dates):
            if d.weekday()==5:
                ax.axvline(i, linewidth = .5, linestyle = '--', color = 'black')

    xt = range(len(Dates))
    ax_visual_ticklabel(ax, {'t': xt[::tick_step], 
                             'tl': Dates_ticks[::tick_step],
                             'rot': rot,
                             'size': 10}, axis = axis)

def _viz_sim_labels(ax, Date_range, title = '', ylabel = '', tick_step = 7):
    '''
    change datas and labels of simulation plot
    '''
    visual_ax_dates(ax, Date_range, tick_step = tick_step, rot = 45)
    ax_visual_labeltitles(ax, 
                          {'xlabel': 'Date', 
                           'ylabel': ylabel,
                           'title': title,
                           'label_size': 15,
                           'title_size': 20})

def viz_single_sim(ax, 
                   Single_sim, 
                   linewidth = 2, 
                   viz= 'CI', 
                   color = 'grey', 
                   scatter = 'false'):
    '''
    visualize single simulation
    '''
    
    #cumulative infected
    N_users = np.sum(Single_sim[0])
    S = Single_sim[:,0]
    CI = N_users - S
    
    #daily infected
    I = Single_sim[:,1]   
    
    if viz =='CI':
        ax.plot(CI, linestyle = '-', color = color, label = 'Cumulative Infected (CI)', 
                linewidth = linewidth)
    if viz =='I':
        ax.plot(I, linestyle = '-', color = color, label = 'Infected (I)', 
                linewidth= linewidth)

def get_em_vals(DICT_EMO_metrics, k, em, N_users, norm = False):
    '''
    Import epidemic metric values
    k  : scenario
    em : epidemic metric
    '''
    ind_epid_metric = config.Emo_metrics_names.index(em)
    if norm and em in ['peak_size', 'final_size']:
        return 100*DICT_EMO_metrics[k][:,ind_epid_metric]/N_users
    else:
        return DICT_EMO_metrics[k][:,ind_epid_metric]

def cumulative_infected(Single_sim):
    '''
    given Single_sim : daily time-series of Susceptibles and Infected
    this function returns the Cumulative number of infected
    '''
    #cumulative infected
    N_users = np.sum(Single_sim[0])
    S = Single_sim[:,0]
    CI = N_users - S
    return CI 

def get_max_indmax(Single_sim, curve = 'CI'):
    '''
    get maximum and index maximum for an epidemiologic curve
    '''
    if curve=='I':
        #daily infected
        I = Single_sim[:,1]
        I_max = np.max(I)
        I_max_ind = np.argwhere(I==I_max)[0,0]
        return I_max, I_max_ind
    if curve=='CI':
        CI = cumulative_infected(Single_sim)
        CI_max = np.max(CI)
        CI_max_ind = np.argwhere(CI==CI_max)[0,0]
        return CI_max, CI_max_ind

def simulations_infected_v1(ax, 
                            DICT_EMO,
                            DICT_EMO_metrics,
                            Date_range,
                            k = 'Complete', 
                            color = config.COLOR_GT, 
                            box_x = 28, 
                            _viz_single_sim = True, 
                            viz = 'I',
                            em = 'peak_size'):
    
    DICT_EMO_metrics_reduced = {k:v[:100] for k, v in DICT_EMO_metrics.items()}
        
    #simulation ensemble
    Sims = DICT_EMO[k][1][:100]
    #single simulation
    Single_sim = Sims[0]
    N_users = np.sum(Single_sim[0])
    
    #descriptive statistics of epidemiological outcome
    CI_max, CI_max_ind = get_max_indmax(Single_sim , 'CI')
    #ax.scatter(CI_max_ind, CI_max, marker = 'x', s = 30, color = 'blue', zorder = 10)
    I_max, I_max_ind = get_max_indmax(Single_sim , 'I')
    #ax.scatter(I_max_ind, I_max, marker = 'x', s = 30, color = 'blue', zorder = 10)

    
    for sim in Sims:
        #Daily Infected
        if _viz_single_sim:
            viz_single_sim(ax, sim, linewidth=.1, color = color, viz = viz)
            I_max, I_max_ind = get_max_indmax(sim , viz)
            ax.scatter(I_max_ind, I_max, marker = 'x', s= 2, color = color)  
            
    _viz_sim_labels(ax, Date_range, ylabel = '')
    
    ax.set_ylabel('Infected', size = 15)
    #ax.set_ylabel('Count', size = 15)
    ax.set_xlim(0,30)
    ax.axvline(27, color = 'black', linewidth = .5)
    Is_max = get_em_vals(DICT_EMO_metrics_reduced, k, em, N_users)
    Ms_ind = np.array([box_x]) 
    
    viz_scatter_boxplot(ax, 
                        [Is_max], 
                        Ms_ind,
                        Colors = ['black'], 
                        Colors_scatter = [color], 
                        cbar = False)
    
    visual_ax_dates(ax, Date_range, tick_step = 7, rot = 45)
    ax.set_xlabel('')
    xticks = list(ax.get_xticks()) + [box_x]
    xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()] 
    
    if viz == 'I':
        xtick_labels += ['Peak']
    if viz == 'CI':
        xtick_labels += ['Total']
        
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

def axis_add_tick(ax,i):
    # get existing ticks and labels
    ticks = list(ax.get_yticks())
    labels = [str(t) for t in ticks]

    # add 0 if missing
    if i not in ticks:
        ticks.append(i)
        labels.append(str(i))

    # sort them together
    ticks, labels = zip(*sorted(zip(ticks, labels)))

    # apply
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

def remove_axis_ticktext(ax, axis='x'):
    if axis == 'x':
        ax.xaxis.set_major_formatter(NullFormatter())
    elif axis == 'y':
        ax.yaxis.set_major_formatter(NullFormatter())

def set_leg_bbox(ax, 
                 bx = 1.05, 
                 by = 1):
    
    leg = ax.get_legend()
    
    if leg is not None:
        leg.set_bbox_to_anchor((bx, by)) 

def set_percent_yticks(ax, decimals=2, factor=100):
    """Format current y-ticks as percentage strings."""
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y*factor:.{decimals}f}" for y in yticks])

def set_percent_xticks(ax, decimals=2, factor=100):
    """Format current x-ticks as percentage strings."""
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x*factor:.{decimals}f}" for x in xticks])


#FUNCTION FOR CREATING DATA FOR VISUALIZATION
def ax_visual_label(ax, DICT_label, legend_pars): 
    Patches = [mpatches.Patch(color=l, label=c)  for c,l in DICT_label.items()]       
    ax.legend(handles= Patches, 
              loc = legend_pars['loc'], 
              fontsize = legend_pars['fontsize'],
              title_fontsize = legend_pars['title_fontsize'])

def ax_visual_xticklabel(ax, DICT_xtl): 
    xt, xtl, rot, size = (DICT_xtl[c] for c in ['xt', 'xtl','rot','size'])
    ax.set_xticks(xt)
    ax.set_xticklabels(xtl, rotation = rot, size = size)

def ax_visual_yticklabel(ax, DICT_ytl): 
    yt, ytl, rot, size = (DICT_ytl[c] for c in ['yt', 'ytl','rot','size'])
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl, rotation = rot, size = size)

def ax_visual_ticklabel(ax, DICT_xtl, axis='cx'): 
    xt, xtl, rot, size = (DICT_xtl[c] for c in ['t', 'tl','rot','size'])
    if axis=='x':
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl, rotation = rot, size = size)
    if axis=='y':
        ax.set_yticks(xt)
        ax.set_yticklabels(xtl, rotation = rot, size = size)   

def ax_visual_labeltitles(ax, DICT_lt): 
    xl = DICT_lt['xlabel']
    yl = DICT_lt['ylabel']
    title = DICT_lt['title']
    size_l = DICT_lt['label_size']
    size_t = DICT_lt['title_size']
    ax.set_xlabel(xl, size = size_l)
    ax.set_ylabel(yl, size = size_l)
    ax.set_title(title, size = size_t)

def ax_visual_legend(ax, 
                     DICT_legend, 
                     List_fc = None): 
    
    Colors = DICT_legend['colors']
    Indicators = DICT_legend['classes']    
    Patches = [mpatches.Patch(edgecolor = c, 
                              facecolor= c, 
                              label=l)  for c,l in zip(Colors, Indicators)]
    if List_fc is not None:
        Patches = [mpatches.Patch(label=l, 
                                  edgecolor = c, 
                                  facecolor=f)  for c,l,f in zip(Colors, Indicators, List_fc)]

    ax.legend(handles = Patches, 
              title   = DICT_legend['title'], 
              loc            = DICT_legend['loc'], 
              fontsize       = DICT_legend['fontsize'],
              title_fontsize = DICT_legend['title_fontsize'])

def gen_DICT_ax_visual(ax_viz = 'label_titles', 
                       return_all = False):

    DICT_ax_visual = {'label_ticks': {'t': '', 
                                      'tl': '',
                                      'rot': 90,
                                      'size': 20},
                      
                      'label_titles': {'xlabel': '', 
                                       'ylabel': '',
                                       'title': '',
                                       'label_size': 20,
                                       'title_size': 25}, 
                    
                      'legend' : {'classes': [],
                                  'colors': [], 
                                  'title': '', 
                                  'loc': 'upper left', 
                                  'fontsize': 15, 
                                  'title_fontsize': 15}}
    
    if return_all:
        return DICT_ax_visual
    
    return DICT_ax_visual[ax_viz]

def visual_imshow(ax, 
                  df_tbr, 
                  Colors, 
                  legend_pars = None):
    '''
    legend_pars has keys: 'loc', 'fontsize', 'title_fontsize'
    '''
    
    cmap = ListedColormap(Colors)  # 0 -> blue, 1 -> yellow
    boundaries = np.arange(0,len(Colors)+1)-0.5   # Define boundaries that map values 1 -> blue, 2 -> red
    norm = BoundaryNorm(boundaries, cmap.N)

    ax.imshow(df_tbr, 
               aspect='auto', 
               cmap=cmap, 
               norm = norm,
               origin = 'lower', 
               interpolation = None,
               interpolation_stage='rgba')

def get_temporal_ticks(Inds, 
                       only_minute = True, 
                       and_first_day_hour = False, 
                       and_first_month_day = False):

    #show bool of indexes
    Inds_sb = Inds.minute == 0
    if and_first_day_hour:
        Inds_sb = Inds_sb & (Inds.hour == 0)
    if and_first_month_day:
        Inds_sb = Inds_sb & (Inds.day==1) 

    return Inds, Inds_sb
    
def set_temporal_xticks(ax, 
                        Inds, 
                        Inds_sb,
                        axis = 'x',
                        str_format = '%Y-%m-%d %H'):

    #tick's index
    x_ti = np.arange(0,len(Inds))[Inds_sb]
    #tick's vals 
    x_tv = np.array(Inds)[Inds_sb]
    if str_format is not None:
        x_tv = pd.to_datetime(x_tv).strftime(str_format)
    if axis=='y':
        ax.set_yticks(x_ti,x_tv , rotation = 90, size = 15)
    else:     
        ax.set_xticks(x_ti,x_tv , rotation = 90, size = 15)   

def plot_stacked_bar(df_count_, 
                     Colors, 
                     Labels, 
                     xs = None, 
                     x_nu = 0, 
                     bar_width = 1, 
                     legend=  False, 
                     legend_loc= 'upper left', log_scale=  False): 
    '''
    plot a count-dataset as a stacked bar
    '''
    df_count = df_count_.copy()
    
    #getting the x-range 
    Cols = df_count.columns 
    x = range(len(Cols))
    if xs is not None: 
        x = xs 

    x = [x_i + x_nu for x_i in x]
    
    #plotting the first count row 
    r = 0
    y_bottom = df_count.iloc[r].values
    plt.bar(x, 
            y_bottom,
            color = Colors[r], 
            label = Labels[r], 
            width = bar_width)
    
    #plotting in order the subsequent count rows
    for r in range(len(df_count))[1:]:
        
        y_up = df_count.iloc[r]
        
        plt.bar(x, 
                y_up, 
                bottom = y_bottom, 
                color  = Colors[r],
                label  = Labels[r], 
                width  = bar_width)

        y_bottom += y_up

    if legend: 
        plt.legend(loc = legend_loc, fontsize = 15)
    if log_scale:
        plt.yscale('log')

def convert_num_colors(numbers):
    # Choose a matplotlib colormap (e.g., 'viridis', 'plasma', 'inferno', etc.)
    cmap = plt.get_cmap('viridis')
    # Normalize the data to range between 0 and 1
    from matplotlib.colors import LogNorm
    norm = Normalize(vmin=min(numbers), vmax=max(numbers))
    # Map the normalized data to colors using the colormap
    colors = cmap(norm(numbers))

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    return colors, sm

def gen_colors(num_colors):
    # Generate a color map with the specified number of colors
    cmap = plt.get_cmap('tab20')  # You can use 'tab10', 'tab20', 'Set3', or any other distinct colormap
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    
    return colors

def map_nc(numbers, log = False, mm_set = None): 
    
    import matplotlib.colors as mcolors
    # Create a colormap (e.g., 'viridis')
    cmap = plt.get_cmap('viridis')
    # Normalize the numbers to a range between 0 and 1
    mm = (min(numbers), max(numbers))
    if mm_set is not None:
        mm = mm_set
                    
    norm = plt.Normalize(mm[0], mm[1])
    if log: 
        norm = mcolors.LogNorm(vmin = mm[0], 
                               vmax = mm[1])
        
    # Map the numbers to colors using the colormap
    colors = cmap(norm(numbers))
    # Create a colorbar with the original range of values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable
    
    return colors, sm


############################################
##### CONTACT -EPIDEMIC VISUALIZATION ######
############################################


#Define the function for visualizing each daily contact using ax object
def visual_contact_daily(ax, df_cm0_, USERS_select, cbar=True):
    df_cm0 = df_cm0_.pivot(index = 'u1', columns = 'u2', values = 'n_minutes').fillna(0).reindex(USERS_select,columns = USERS_select).fillna(0)
    #df_cm0+=1e-6
    cax = ax.matshow(df_cm0.values, 
                     cmap=cm.jet, 
                     #vmin=0, vmax=3600, 
                     norm=LogNorm(vmin=1, vmax=3600),
                     origin='lower', alpha=1.0)
    if cbar:
        cbar = plt.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.1)
        #cbar.set_ticks(np.arange(1, 3601, 120))
        cbar.set_ticks([1, 60, 300, 600, 1800, 3600])  # Adjust tick values as needed
        cbar.set_ticklabels([1, 60, 300, 600, 1800, 3600]) 
        cbar.set_label('n_minutes', fontsize=  15)


def visual_epid_count(ax, ts_SI, Dates, USERS_select):
    ax.plot(Dates, ts_SI[:, 0][:-1], label='S', color='green')
    ax.plot(Dates, ts_SI[:, 1][:-1], label='I', color='red')
    ax.set_xticks(Dates)
    ax.set_xticklabels(Dates, rotation=90)
    ax.set_ylim(0, len(USERS_select))
    ax.legend()

def visual_R0(ax,ts_SI, Dates):
    ax.plot(Dates, ts_SI[:, 2][:-1], color='blue')
    ax.set_xticks(Dates)
    ax.set_xticklabels(Dates, rotation=90)

def visual_shaded_area(ax, x, x_minus, x_plus, color = 'blue', label = '', fill_btw = True, visual_line = True):
    if visual_line:
        ax.plot(x, color=color, label = label)
    if fill_btw:
        ax.fill_between(range(len(x)), x_minus, x_plus, color=color, alpha=0.2)

def visual_curves_SI(ax, COLLECT_simulations, USERS_select, Dates):
    q25, q50, q75 = np.percentile(COLLECT_simulations, [25, 50, 75], axis=0, method = 'nearest')
    visual_shaded_area(ax, q50[:,0], q25[:,0], q75[:,0], color = 'green', label ='S')
    visual_shaded_area(ax, q50[:,1], q25[:,1], q75[:,1], color = 'red', label ='I')
    ax.set_xticks(range(len(Dates)), Dates)
    ax.set_xticklabels(Dates, rotation=90)
    ax.set_ylim(0, len(USERS_select))    
    ax.set_ylabel('user count')
    ax.legend()

def visual_curves_SI_spectus(ax, 
                             C_sims, 
                             class_ = 'I',
                             color = 'red', 
                             legend = False, 
                             fill_btw = True, 
                             visual_median = True,
                             visual_mean = False,
                             visual_all_sims = False, 
                             visual_saturday = False,
                             alpha = 0.2):
    '''
    visualize the SI curves from spectus data
    '''
    
    q25, q50, q75 = np.percentile(C_sims, 
                                  [25, 50, 75], 
                                  axis=0, 
                                  method = 'nearest')
    
    if class_ == 'I':
        visual_shaded_area(ax, 
                           q50[:,1], q25[:,1], q75[:,1], 
                           color = color, 
                           fill_btw = fill_btw, visual_line = visual_median, alpha = alpha)
    if class_ == 'S':
        visual_shaded_area(ax, 
                           q50[:,0], q25[:,0], q75[:,0], 
                           color = color, 
                           fill_btw = fill_btw, 
                           visual_line = visual_median, alpha = alpha)

    if visual_all_sims:
        #print(f'yes {class_}')
        for c in C_sims:
            if class_ == 'I':
                ax.plot(c[:,1], color = color, linewidth = .1)
            if class_ == 'S':
                ax.plot(c[:,0], color = color, linewidth = .1)
            if class_ == 'I_tot':
                N_tot = np.sum(c[0])
                ax.plot(N_tot - c[:,0], color = color, linewidth = .1)


    if visual_mean:
        if class_ == 'I':
            ax.plot(np.mean(C_sims[:,:,1], axis = 0), color = color, linewidth = 2, linestyle = '--')
        if class_ == 'S':
            ax.plot(np.mean(C_sims[:,:,0], axis = 0), color = color, linewidth = 2, linestyle = '--')
        
    #ax.set_ylabel(f'user count', size = 20)
    if legend:
        ax.legend()

def visual_curves_R0(ax, COLLECT_simulations, Dates):
    q25, q50, q75 = np.percentile(COLLECT_simulations, [25, 50, 75], axis=0, method = 'nearest')
    visual_shaded_area(ax, q50[:,2], q25[:,2], q75[:,2], color = 'blue', label ='R0')
    ax.set_xticks(range(len(Dates)), Dates)
    ax.set_xticklabels(Dates, rotation=90)
    ax.legend()

def visual_epid_simulation(axes, COLLECT_simulations, Dates):
    '''
    visualize iterated epidemiological simulation
    '''    
    visual_curves_SI(axes[0], COLLECT_simulations, Dates)
    visual_curves_R0(axes[1], COLLECT_simulations, Dates)

def ax_legend_level(ax, 
                    loc = 'lower left', 
                    fontsize = 10,
                    include_complete = True):
    
    Classes = ['ground truth'] + [convert_to_percent_range(str(l)) for l in Levels]       
    Colors  = ['blue']     + [DICT_colors_level[l] for l in Levels]

    if not include_complete:
        Classes = Classes[1:]
        Colors = Colors[1:]    
    
    DICT_legend = gen_DICT_ax_visual('legend')
    DICT_legend.update({'classes': Classes, 
                        'colors': Colors, 
                        'loc': loc, 
                        'fontsize':10})
    
    ax_visual_legend(ax, DICT_legend)

def ax_legend_sparsification(ax, 
                             loc = 'upper right', 
                             fontsize = 10, 
                             include_complete = True):

    Classes = ['ground truth'] + [DICT_rename_ss_brief[s] for s in List_ss] 
    Colors  = ['blue']     + [DICT_colors_ss[s] for s in List_ss]      
    if not include_complete:
        Classes = Classes[1:]
        Colors = Colors[1:]  
        
    DICT_legend = gen_DICT_ax_visual('legend')
    DICT_legend.update({'classes': Classes,
                        'colors': Colors, 
                        'title': '', 
                        'loc': loc, 
                        'fontsize':fontsize})
    ax_visual_legend(ax, DICT_legend)

def ax_legend_emv(ax, 
                  EMVs, 
                  loc = 'upper right', 
                  fontsize = 10, 
                  title_fontsize = 10,
                  title= '',
                  include_complete = True):

    Classes = ['ground truth'] +  [DICT_rename_EMVs[s] for s in EMVs]
    Colors  = ['blue'] +      [DICT_colors_emv[s] for s in EMVs]  
    if not include_complete:
        Classes = Classes[1:]
        Colors = Colors[1:]  
                                 
    DICT_legend = gen_DICT_ax_visual('legend')
    DICT_legend.update({'classes': Classes,
                        'colors': Colors, 
                        'title': title, 
                        'loc': loc, 
                        'fontsize':fontsize,
                        'title_fontsize': title_fontsize})
    ax_visual_legend(ax, DICT_legend)

def ax_visual_line_legend(ax, DICT_legend):
    
    Colors = DICT_legend['colors']
    Indicators = DICT_legend['classes']
    LineStyles = DICT_legend.get('linestyles', ['solid'] * len(Colors))  # Default to solid if not specified

    Patches = [mlines.Line2D([], [], color=c, linestyle=ls, label=l, linewidth=2) 
               for c, l, ls in zip(Colors, Indicators, LineStyles)]

    ax.legend(handles=Patches, 
              title=DICT_legend['title'],
              loc=DICT_legend['loc'], 
              fontsize=DICT_legend['fontsize'],
              title_fontsize=DICT_legend['title_fontsize'])

def scatter_df(ax, 
               df, x, y, 
               title = '', 
               s = 1,
               c = 'blue',
               x_rename = None, 
               y_rename = None, 
               label_size =  15,
               title_size = 20,
               cmap = None,
               vmin = None,
               vmax = None, 
               colorbar = False, 
               colorbar_label = ''): 
    
    X = df[x].values
    Y = df[y].values
    
    sc = ax.scatter(X, Y,  
                    c = c, 
                    s = s, 
                    cmap = cmap,
                    vmin = vmin, 
                    vmax = vmax)

    if colorbar:
        cbar = plt.colorbar(sc, ax = ax)
        cbar.set_label(colorbar_label)
        
    if x_rename is None:
        x_rename = x
    if y_rename is None:
        y_rename = y

    ax_visual_labeltitles(ax, {'xlabel': x_rename, 
                               'ylabel': y_rename,
                               'title': title, 
                               'label_size': label_size, 
                               'title_size': title_size})
    

def rescale_ax_ticks(ax, 
                     int_scale = 4, 
                     digit_round = 0, 
                     scient_not_drop = False, 
                     axis = 'x'):
    '''
    scient_not_drop: drop scientific notation if already rescaled and put it into 
    else rescale the ticks accordingly
    '''

    if scient_not_drop:
        if axis=='y':
            ax.yaxis.offsetText.set_visible(False)
            ylabel = ax.get_ylabel()
            ax.set_ylabel(f"{ylabel} $(x10^{int_scale})$")
        if axis=='x':
            ax.xaxis.offsetText.set_visible(False)
            xlabel = ax.get_xlabel()
            ax.set_xlabel(f"{xlabel} $(x10^{int_scale})$")
        
    else:
        if axis=='y':
            yticks = ax.get_yticks()
            yticks_rescaled = [f'{np.round(y/(10**int_scale),digit_round)}' for y in yticks]
            ax.set_yticklabels(yticks_rescaled)
            ylabel = ax.get_ylabel()
            ax.set_ylabel(f"{ylabel} $(x10^{int_scale})$")
        if axis=='x': 
            xticks = ax.get_xticks()
            xticks_rescaled = [f'{np.round(x/(10**int_scale),digit_round)}' for x in xticks]
            ax.set_xticklabels(xticks_rescaled)
            xlabel = ax.get_xlabel()
            ax.set_xlabel(f"{xlabel} $(x10^{int_scale})$")


#Visualize scatterplot and boxplot together
def viz_scatter_boxplot(ax,
                        List_Ms, 
                        Ms_ind,
                        Colors, 
                        Colors_scatter = 'black',
                        vmin = 0, 
                        vmax = 15, 
                        cmap = 'viridis',
                        cbar = True,
                        cbar_label = '',
                        scatter_size = 1):
    '''
    Visualize the boxplots of a list of values in List_Ms corresponding to indexes Ms_ind,
    Colors are the colors of the boxplots
    Colors_scatter are the colors of the scatter points (can be also a list)
        cmap : of the Colors_scatter
    '''

    for i, c, Ms in zip(Ms_ind, Colors, List_Ms):
        
        ax.boxplot(Ms, 
                   positions = [i], 
                   widths = 0.5, 
                   showfliers = False, 
                   patch_artist=True,
                   boxprops=dict(facecolor= 'none', color= c),
                   capprops=dict(color=c),
                   whiskerprops=dict(color=c),
                   flierprops=dict(color=c, markeredgecolor=c),
                   medianprops=dict(color=c))

        X_vals = [i + np.random.normal(scale = 0.05) for j in range(len(Ms))]
        if len(Colors_scatter)>1:
            IND = np.argwhere(Ms_ind==i)[0][0]
            sc = ax.scatter(X_vals, 
                            Ms, 
                            c = Colors_scatter[IND],
                            vmin = vmin, vmax = vmax, cmap = cmap, 
                            s = scatter_size)
            if cbar and IND==0:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(cbar_label, size=14)
            
        else:
            ax.scatter(X_vals, 
                       Ms, 
                       color = Colors_scatter, 
                       s = scatter_size)

def change_first_xtick(ax, newlabel='newlabel'):
    xticks = ax.get_xticks()  # Get current x-tick positions
    xticklabels = ax.get_xticklabels()  # Get current x-tick labels
    ax.set_xticks(xticks)  # Update x-ticks
    xticklabels[0] = newlabel  # Modify first tick label
    ax.set_xticklabels(xticklabels)  # Update x-tick labels

#Visualize scatterplot and boxplot together
def viz_scatter_boxplot_new(ax,
                            List_Ms, 
                            Ms_ind,
                            Colors = 'black', 
                            Colors_scatter = 'black',
                            scatter = True,
                            scatter_size = 1, 
                            x_Labels = None, 
                            x_label_size = 0,
                            x_label_rot = 0):
    '''
    Visualize the boxplots of a list of values in List_Ms corresponding to indexes Ms_ind
    Colors are the colors of the boxplots
    Colors_scatter are the colors of the scatter points (can be also a list)
        cmap : of the Colors_scatter
    '''
    
    if not isinstance(Colors, list):
        Colors = [Colors] * len(List_Ms)
    if not isinstance(Colors_scatter, list):
        Colors_scatter = [Colors_scatter] * len(List_Ms)

    for i, Ms, c, c_scatter in zip(Ms_ind, 
                                   List_Ms, 
                                   Colors, 
                                   Colors_scatter):
        
        ax.boxplot(Ms, 
                   positions = [i], 
                   widths = 0.5, 
                   showfliers = False, 
                   patch_artist = True,
                   boxprops = dict(facecolor = 'none', color = c),
                   capprops = dict(color=c),
                   whiskerprops = dict(color=c),
                   flierprops = dict(color=c, markeredgecolor=c),
                   medianprops = dict(color=c))

        #add scatter points within the boxplot
        if scatter:
            X_vals = [i + np.random.normal(scale = 0.05) for j in range(len(Ms))]
            ax.scatter(X_vals, 
                       Ms, 
                       color = c_scatter, 
                       s = scatter_size)
            
    if x_Labels is not None: 
        ax_visual_ticklabel(ax, {'t': Ms_ind, 
                                 'tl': x_Labels,
                                 'rot': x_label_rot,
                                 'size': x_label_size}, axis = 'x')

def viz_bar_series(ax, 
                   s, 
                   s_ind = None, 
                   Colors = 'black',
                   x_Labels= None, 
                   x_label_rot=0,
                   x_label_size=10):
    '''
    visual a series values with an ax.bar plot with customized colors and labeling
    s: series
    s_ind : x-ticks of bar plots
    Colors : can be also a list equal to the length of the series
    '''

    if not isinstance(Colors, list):
        Colors = [Colors] * len(s)
        
    if x_Labels is not None:
        s = s.loc[x_Labels]
    else:
        x_Labels = s.index
        
    if s_ind is None:
        s_ind = range(len(s))
        
    for i,val,c in zip(s_ind, s.values, Colors):
        ax.bar([i], val, color = c)
        
    ax_visual_ticklabel(ax, {'t': s_ind, 
                             'tl': x_Labels,
                             'rot': x_label_rot,
                             'size': x_label_size}, axis = 'x')

def custom_boxplot_from_stats(ax, 
                              stats, 
                              positions=None, 
                              color='black', 
                              width=.8, 
                              linewidth=1,
                              face_alpha=1,
                              median_color='black',
                              bar_as_median=False,
                              bar_errorbar = True, 
                              capsize = 1):
    """
    Draw a boxplot (default) or a barplot-with-errorbars (if bar_as_median=True)
    given precomputed stats (dicts with q1, q3, med, whislo, whishi).

    color: str or list[str] — single color for all boxes or one per box.
    median_color: str — color for median lines (default 'black').
    bar_as_median: bool — if True, plot median as bar height with CI error bars instead of a box.
    """
    
    n = len(stats)
    if positions is None:
        positions = list(range(1, n+1))

    # normalize color input
    if isinstance(color, (list, tuple)):
        if len(color) != n:
            raise ValueError("If 'color' is a list, its length must equal len(stats).")
        colors = list(color)
    else:
        colors = [color] * n

    if bar_as_median:
        # Draw bars at median with error bars from whislo/whishi
        medians = [s['med'] for s in stats]
        err_low = [s['med'] - s['whislo'] for s in stats]
        err_high = [s['whishi'] - s['med'] for s in stats]
        ax.bar(positions, medians,
               color=colors,
               alpha=face_alpha,
               width=width,
               edgecolor='none',
               zorder=2)
        if bar_errorbar:
            ax.errorbar(positions, medians,
                        yerr=[err_low, err_high],
                        fmt='none',
                        ecolor=median_color,
                        elinewidth=linewidth,
                        capsize=capsize,
                        zorder=3)
        
        return None  # nothing to return in this mode

    # --- default boxplot branch ---
    bp = ax.bxp(stats,
                positions=positions,
                widths=width,
                patch_artist=True,
                showfliers=False)

    for i in range(n):
        c = colors[i]
        bp['boxes'][i].set(facecolor=c, edgecolor=c, linewidth=linewidth, alpha=face_alpha)
        bp['medians'][i].set(color=median_color, linewidth=linewidth)
        #if i < len(bp['fliers']):
        #    bp['fliers'][i].set(marker='o', color=c, alpha=0.7)

        wi0, wi1 = 2*i, 2*i + 1
        bp['whiskers'][wi0].set(color=c, linewidth=linewidth)
        bp['whiskers'][wi1].set(color=c, linewidth=linewidth)
        bp['caps'][wi0].set(color=c, linewidth=linewidth)
        bp['caps'][wi1].set(color=c, linewidth=linewidth)

    return bp

def caret_marker(direction = "up", 
                 width = 1.0, 
                 height = 1.0):
    """
    Symmetric triangular caret centered at (0,0).
    direction: "up" or "down"
    width, height: relative shape proportions.
    """
    w = 0.9 * width
    h = 0.9 * height
    verts = np.array([
        [ 0.0,  h/2],   # tip
        [-w/2, -h/2],   # left base
        [ w/2, -h/2],   # right base
        [ 0.0,  0.0],   # ignored for CLOSEPOLY
    ])
    if direction == "down":
        verts[:, 1] *= -1
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    return MarkerStyle(Path(verts, codes))

def visual_metric(ax, 
                  df_stats, 
                  metric, 
                  List_ss,
                  X,
                  *,
                  gt_tick = [0],
                  DICT_colors_ss = config.DICT_colors_ss,
                  COLOR_GT_std = 'black', #config.COLOR_GT,
                  col_gt = 'Complete',
                  visual_bxp_groundtruth=True,
                  visual_std_groundtruth=True,
                  visual_mean=True,
                  visual_mean_std=False,
                  visual_bxp=True,
                  set_yax_percent=False,
                  # --- NEW ---
                  color_mean_std='black',          # can be str or list per-level
                  mean_marker_size=5,
                  cap_std_marker_size=8,
                  err_linewidth=1.2, 
                  marker_std_up = caret_marker('up'),
                  marker_std_down = caret_marker('down'),
                  xtl_noperc = False,
                  face_alpha = 1, 
                  linewidth =1,
                  capsize=1,
                  bar_as_median = False,
                  bar_errorbar_gt = True):
    """
    df_stats must have as index [s, l]; sparsity and sparsity level 
    when analyzing debiasing it should be [emv, l] : debiasing approach and sparsity level
    
    Visualize boxplots and/or mean (+/- std) for each sparsity model in List_ss.

    Colors:
      - DICT_colors_ss[s] can be a single color or a list of colors (len == len(Levels)).
      - color_mean_std can also be a single color or a list per-level. If None, it follows DICT_colors_ss[s].
    """
    # standard matplotlib naming of stats for IQR and CI
    bxp_stats = ['whislo', 'q1', 'med', 'q3', 'whishi']
    
    # groundtruth stats
    s_bxp_gt = [{stat: df_stats.loc[col_gt, col_gt][f'{metric}_{stat}']
                 for stat in bxp_stats}]
    s_bxp_gt[0].update({'fliers': []})
    
    s_mean_gt = [df_stats.loc[col_gt, col_gt][f'{metric}_mean']]
    s_std_gt  = [df_stats.loc[col_gt, col_gt][f'{metric}_std']]

    Levels = config.Levels
    n_levels = len(Levels)

    def _normalize_colors(c):
        """Return a list of length n_levels from a color or color list."""
        if isinstance(c, (list, tuple)):
            if len(c) != n_levels:
                raise ValueError(f"Color list must match len(Levels)={n_levels}")
            return list(c)
        else:
            return [c] * n_levels

    for j, s in enumerate(List_ss):
        print(s)
        # base color(s) for this sparsity model
        base_color = DICT_colors_ss[s]
        print(base_color)
        colors_levels = _normalize_colors(base_color)

        # mean/std colors per level (follow base if None; else normalize)
        if color_mean_std is None:
            ms_colors = colors_levels
        else:
            ms_colors = _normalize_colors(color_mean_std)

        # [0] gather stats per level
        s_bxp = [{k: df_stats.loc[s, str(l)][f'{metric}_{k}'] for k in bxp_stats}
                 for l in Levels]
        for d in s_bxp:
            d.update({'fliers': []})

        # [1] boxplots (support per-level colors)
        if visual_bxp:
            custom_boxplot_from_stats(ax,
                                      s_bxp,
                                      positions=X + j,
                                      color=colors_levels, 
                                      face_alpha = face_alpha, 
                                      bar_as_median = bar_as_median,
                                      capsize=capsize, 
                                      linewidth = linewidth) # <-- list or single OK
            
            # median ground-truth line
            gt_median = df_stats.loc[col_gt, col_gt][f'{metric}_med']
            ax.axhline(gt_median, color=config.COLOR_GT, linewidth=.5)

        if visual_bxp_groundtruth:
            custom_boxplot_from_stats(ax,
                                      s_bxp_gt,
                                      positions=[0],
                                      color=config.COLOR_GT,
                                      capsize=capsize, 
                                      linewidth = linewidth,
                                      bar_as_median = bar_as_median,
                                      bar_errorbar = bar_errorbar_gt)

        # [2] mean and std
        if visual_mean or visual_mean_std:
            s_mean = [df_stats.loc[s, str(l)][f'{metric}_mean'] for l in Levels]
            s_std  = [df_stats.loc[s, str(l)][f'{metric}_std']  for l in Levels]
            x_pos  = np.full(n_levels, X + j, dtype=float)

        
        
        if visual_mean:
            # per-level mean markers
            for xi, yi, c in zip(x_pos, s_mean, ms_colors):
                ax.scatter(xi, yi, color = c, s = mean_marker_size, marker='o', zorder=3)

            # ground truth mean
            ax.scatter(gt_tick, s_mean_gt, color=COLOR_GT_std, s=mean_marker_size, marker='o', zorder=3)

            if visual_std_groundtruth:
                # GT std line + caret caps
                ax.errorbar(gt_tick,
                            s_mean_gt,
                            yerr=s_std_gt,
                            fmt='none', ecolor=COLOR_GT_std,
                            elinewidth = err_linewidth, capsize=0, zorder=2)
                
                ax.scatter(gt_tick, [s_mean_gt[0] + s_std_gt[0]],
                           marker= marker_std_up,   color=COLOR_GT_std,
                           s=cap_std_marker_size, zorder=4)
                ax.scatter(gt_tick, [s_mean_gt[0] - s_std_gt[0]],
                           marker= marker_std_down, color=COLOR_GT_std,
                           s=cap_std_marker_size, zorder=4)

        if visual_mean_std:
            # loop per level so each errorbar uses its own color
            for xi, yi, si, c in zip(x_pos, s_mean, s_std, ms_colors):
                ax.errorbar([xi], [yi], yerr=[si],
                            fmt='none', ecolor=c,
                            elinewidth=err_linewidth, capsize=0, zorder=2)
                ax.scatter(xi, yi + si, marker = marker_std_up,
                           color=c, s=cap_std_marker_size, zorder=4)
                ax.scatter(xi, yi - si, marker = marker_std_down,
                           color=c, s=cap_std_marker_size, zorder=4)

    # x-ticks
    DICT_xtl = gen_DICT_ax_visual('label_ticks')
    DICT_xtl['t']  = [0] + list(X)
    xtl = ['0-5'] + config.Levels_str
    
    if xtl_noperc:
        xtl = [x[:-1] for x in xtl]
        
    DICT_xtl['tl'] = xtl
    DICT_xtl['rot'] = 45
    ax_visual_ticklabel(ax, DICT_xtl, axis='x')

    # y ticks as percents
    if set_yax_percent:
        DICT_ytl = gen_DICT_ax_visual('label_ticks')
        yticks = ax.get_yticks()
        yticks = yticks[(yticks >= 0) & (yticks <= 1)]
        DICT_ytl['t']  = yticks
        DICT_ytl['tl'] = (yticks * 100).astype(int)
        DICT_ytl['rot'] = 0
        ax_visual_ticklabel(ax, DICT_ytl, axis='y')

def restyle_ax(ax,
               title_size =        config.ax_title_size,
               label_size =        config.ax_label_size,
               text_size  =        config.ax_text_size,
               tick_size  =        config.ax_tick_size,
               legend_font_size  = config.ax_legend_font_size,
               legend_title_size = config.ax_legend_title_size,               
               font_family='Liberation Sans', 
               other_text = True):
    """
    Apply consistent font sizes and font family to ticks, labels, title, legend, and texts in an Axes.
    """

    # Axis labels
    ax.xaxis.label.set_size(label_size)
    ax.xaxis.label.set_family(font_family)
    ax.yaxis.label.set_size(label_size)
    ax.yaxis.label.set_family(font_family)

    # Title
    ax.title.set_size(title_size)
    ax.title.set_family(font_family)

    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(tick_size)
        tick.set_family(font_family)

    # Legend
    legend = ax.get_legend()
    if legend:
        legend.get_title().set_fontsize(legend_title_size)
        legend.get_title().set_family(font_family)
        for text in legend.get_texts():
            text.set_fontsize(legend_font_size)
            text.set_family(font_family)
    if other_text:
        # Other text
        for text in ax.texts:
            text.set_fontsize(text_size)
            text.set_family(font_family)

#####################
#### RIDGEPLOTS #####
#####################

#histogram creation function
def compute_discrete_ridge_data(
    df,
    by,
    column,
    bins,
    normalize = 'max'):
    
    groups = list(df[by].dropna().unique())
    H = []
    for g in groups:
        h, _ = np.histogram(df.loc[df[by] == g, column], bins=bins)
        H.append(h.astype(float))
        
    H = np.vstack(H)
    widths = np.diff(bins)
    
    if normalize == 'density':
        H = (H / H.sum(1, keepdims=True)) / widths
    elif normalize == 'max':
        m = H.max()
        if m > 0:
            H = H / m
    x = (bins[:-1] + bins[1:]) / 2
    
    return x, H, groups

def plot_discrete_ridges(ax,
                         x,
                         H,
                         groups = None,
                         overlap = 1.4,
                         alpha = 0.55,
                         color = cm.Blues,        # can be str, list, or colormap
                         linewidth = 1.0,
                         show_labels = True,
                         mask = None,             # boolean list/array of len(groups)
                         x_label = 0,
                         labels = None,           # NEW: list/array of custom labels per group
                         label_rotation = 0,      # NEW: rotation (deg)
                         label_size = 10):        # NEW: fontsize

    spacing = 1.0 / overlap

    if groups is None:
        groups = list(range(len(H)))

    if mask is None:
        mask = [True] * len(groups)
    elif len(mask) != len(groups):
        raise ValueError("`mask` must have same length as groups")

    if isinstance(color, str):
        colors = [color] * len(groups)
    elif hasattr(color, "__call__"):
        colors = [color(i / max(1, len(groups)-1)) for i in range(len(groups))]
    elif isinstance(color, (list, tuple)) and len(color) == len(groups):
        colors = color
    else:
        raise ValueError("`color` must be a string, colormap, or list of len(groups).")

    n = len(groups)
    for idx in range(n - 1, -1, -1):
        g    = groups[idx]
        h    = H[idx]
        c    = colors[idx]
        keep = mask[idx]
        y0 = idx * spacing
        ridge_alpha = alpha if keep else 0.0

        x = np.asarray(x)
        w = np.median(np.diff(x)) if len(x) > 1 else 1.0
        edges = np.concatenate([x - w/2, [x[-1] + w/2]])
        h_ext = np.concatenate([h, [h[-1]]])

        if keep:
            ax.axhline(y0, color='black', lw=0.5, zorder=0)

        ax.fill_between(
            edges, y0, y0 + h_ext,
            step='post', alpha=ridge_alpha,
            facecolor=c, edgecolor='black', zorder=1
        )

        ax.plot(
            edges, y0 + h_ext,
            drawstyle='steps-post', color='black',
            lw=linewidth, alpha=ridge_alpha, zorder=2
        )

        # --- generalized label block ---
        if keep and show_labels:
            txt = (labels[idx] if (labels is not None and len(labels) == n)
                   else str(g))
            ax.text(
                x_label, y0, txt,
                va='center', ha='right',
                rotation=label_rotation,
                fontsize=label_size,
                zorder=3
            )

def discrete_ridge_hist(
    ax,
    df,
    by,
    column,
    bins,
    overlap=1.4,
    alpha=0.55,
    color=cm.Blues,
    normalize='max',
    linewidth=1.0,
    show_labels = False,
    x_label = 0,
    labels = None,           # NEW: list/array of custom labels per group
    label_rotation = 0,      # NEW: rotation (deg)
    label_size = 10):        # NEW: fontsize

    x, H, groups = compute_discrete_ridge_data(
        df=df,
        by=by,
        column=column,
        bins=bins,
        normalize=normalize
    )
    plot_discrete_ridges(
        ax=ax,
        x=x,
        H=H,
        groups=groups,
        overlap=overlap,
        alpha=alpha,
        color=color,
        linewidth=linewidth,
        show_labels=show_labels,
        x_label = x_label,
        labels = labels,           # NEW: list/array of custom labels per group
        label_rotation = label_rotation,      # NEW: rotation (deg)
        label_size = label_size)        # NEW: fontsize
    
def plot_level_ridges(
    df_freq,
    ax,
    l,
    ss=None,                    # list of scenario keys (defaults to global _ss if present)
    x=None,                     # x positions; defaults to range over df_freq columns
    overlap=4,
    alpha=0.8,
    color=None,
    linewidth=1,
    show_labels=False,
    x_label=0,
    mask=None,
    labels=None,                # custom labels per ridge (len == len(ss))
    label_rotation=0,
    label_size=10):
    if ss is None:
        ss = _ss  # fallback to existing global if user keeps it
    if x is None:
        x = range(df_freq.shape[1])

    H_l = [df_freq.loc[(s, str(l))].values for s in ss]

    plot_discrete_ridges(
        ax=ax,
        x=x,
        H=H_l,
        groups=ss,
        overlap=overlap,
        alpha=alpha,
        color=color,
        linewidth=linewidth,
        show_labels=show_labels,
        x_label=x_label,
        mask=mask,
        labels=labels,
        label_rotation=label_rotation,
        label_size=label_size
    )

def plot_missingness_ridges(
    df_freq,
    ax,
    s,
    levels=None,                # list/iterable of Levels (defaults to global Levels if present)
    x=None,
    overlap=4,
    alpha=0.8,
    color=None,
    linewidth=1,
    show_labels=False,
    x_label=0,
    mask=None,
    labels=None,                # custom labels per level (len == len(levels))
    label_rotation=0,
    label_size=10):
    
    if levels is None:
        levels = config.Levels  # fallback to existing global if user keeps it
    if x is None:
        x = range(df_freq.shape[1])

    H_l = [df_freq.loc[(s, str(l))].values for l in levels]

    plot_discrete_ridges(
        ax=ax,
        x=x,
        H=H_l,
        groups=[str(l) for l in levels],
        overlap=overlap,
        alpha=alpha,
        color=color,
        linewidth=linewidth,
        show_labels=show_labels,
        x_label=x_label,
        mask=mask,
        labels=labels,
        label_rotation=label_rotation,
        label_size=label_size
    )

def plot_groundtruth_ridges(
    df_freq,
    ax,
    groundtruth=('Complete', 'Complete'),
    k=3,                        # number of repeated ridges (ignored if mask is given)
    x=None,
    overlap=4,
    alpha=0.8,
    color=None,
    linewidth=1,
    show_labels=False,
    x_label=0,
    mask=None,                  # e.g., [True, False, False]
    labels=None,                # custom labels per repeated ridge (len == len(mask or k))
    label_rotation=0,
    label_size=10):
    if x is None:
        x = range(df_freq.shape[1])

    v = df_freq.loc[groundtruth].values

    # Determine how many layers to plot
    if mask is not None:
        k_eff = len(mask)
    else:
        k_eff = k
        mask = [True] + [False] * (k_eff - 1)  # default: show only first ridge

    H_gt = [v] * k_eff
    groups = [f"GT{i+1}" for i in range(k_eff)]  # placeholder group names unless labels provided

    plot_discrete_ridges(
        ax=ax,
        x=x,
        H=H_gt,
        groups=groups,
        overlap=overlap,
        alpha=alpha,
        color=color,
        linewidth=linewidth,
        show_labels=show_labels,
        x_label=x_label,
        mask=mask,
        labels=labels,
        label_rotation=label_rotation,
        label_size=label_size
    )

def annotate_axes(
    axes,
    texts=None,
    dates=None,
    xs=None,
    tick_size = .7,
    y_tick_height = -0.1,
    y_tick_height_last_axis = -0.1,
    y_date_labels = 0):
    """
    Annotate axes with optional category labels on the left
    and date ticks/labels on the bottom axis.

    Parameters
    ----------
    axes : list of Axes
        Axes objects to annotate.
    texts : list of str or None, optional
        Labels for each subplot (reversed order is used).
        If None, no text is drawn.
    dates : sequence of datetime, optional
        Full sequence of dates for labeling.
    xs : list of int, optional
        Positions in the date sequence to mark and label.
    """
    
    if xs is None:
        xs = []
    xs_dates = []
    if dates is not None and xs:
        xs_dates = [dates[i-1].strftime("%-d %b") for i in xs]

    # category labels on the left
    if texts is not None:
        for ax, t in zip(axes, texts[::-1]):
            ax.text(
                1, .6, t,
                transform=ax.get_xaxis_transform()
            )

    #xlimits
    for ax in axes:
        ax.set_xlim(1, 27.5)

    # tick marks
    for ax in axes[:-1]:
        if xs:
            ax.plot(
                xs, [y_tick_height]*len(xs),
                linestyle='none', marker='|',
                markersize= tick_size, color='black',
                zorder = 999,
                clip_on=False
            )

    # date labels only on the bottom axis
    if xs_dates:
        
        ax = axes[-1]
        ax.plot(xs, 
                [y_tick_height_last_axis]*len(xs),
                linestyle='none', 
                marker = '|',
                markersize = tick_size, 
                color = 'black',
                zorder = 999,
                clip_on = False)
        
        for x, label in zip(xs, xs_dates):
            ax.text(x, 
                    y_date_labels, 
                    label,
                    rotation = 45, 
                    fontsize = 10,
                    ha = 'right', 
                    va = 'top',
                    transform=ax.get_xaxis_transform(),
                    clip_on=False)

def panel_metric_dynamic(axes, 
                         c_freq, 
                         c,
                         _ss = config.List_ss_rename,
                         DICT_colors = config.DICT_colors_ss):
                         
    '''
    axes: axes stacked vertically
    c_freq: fequency of c
        - must have index (sparsity, sparsity_level)
    c : dynamic metric ('day_peak', 'day_last_case', 'day_last_recovery'])
    _ss : list of sparsity approaches or debiasing approaches
    '''

    overlap   = 1.5
    alpha     = 0.8
    linewidth = 1
    colors = [DICT_colors[s] for s in _ss]
    
    for s, ax in zip(_ss, axes[:-1]):
        plot_missingness_ridges(
            df_freq=c_freq,
            ax=ax,
            s=s,
            overlap=overlap,
            alpha=alpha,
            color= DICT_colors[s],
            linewidth=.01,
            show_labels = True, 
            label_rotation = 0,
            labels = ['10-20', '','','', '50-60'])
        
        ax.axis('off')
    
    ax = axes[-1]
    plot_groundtruth_ridges(
        df_freq=c_freq,
        ax=ax,
        groundtruth=('ground truth', 'ground truth'),
        overlap=overlap,
        alpha=alpha,
        color=config.COLOR_GT,
        linewidth=.01,
        mask = [False,False,False,False,True], 
        show_labels = True, 
        labels = ['','','','','0-5'])
    ax.axis('off')
    
    _xs = [1, 8, 15, 22, 27]
    annotate_axes(axes, 
                  dates = config.Dates_plus1, 
                  xs= _xs,
                  tick_size = 5,
                  y_tick_height = -.1, 
                  y_tick_height_last_axis = 2.6, 
                  y_date_labels = .7)
    
    for ax, color in zip(axes, colors):
    
        rect = patches.Rectangle((0, .05),          # bottom-left in axes coords
                                 1, .95,            # full width & height
                                 transform = ax.transAxes,  # use axes coordinates
                                 facecolor = color,
                                 alpha = .4,
                                 zorder = -1)
        ax.add_patch(rect)
    
    
    for ax in axes:
        restyle_ax(ax,
                   text_size  = config.ax_tick_size, 
                   title_size = config.ax_label_size)

####################################    
##### CONTACT VISUALIZATION ########
####################################

def draw_filled_line(
    ax,
    x,
    y,
    color,
    base=0.03,
    alpha=0.2,
    lw=1.5,
    s=10,
    marker='o',
    facecolor='none'):
    """Fill under curve to a base line, then draw line and markers."""
    
    ax.fill_between(
        x, y, base,
        color=color,
        alpha=alpha
    )

    ax.plot(
        x, y,
        color=color,
        linewidth=lw
    )

    ax.scatter(
        x, y,
        edgecolor=color,
        facecolor=facecolor,  # use facecolor argument
        s=s,
        marker=marker
        
    )

def to_12h_label(h):
    """Return '12 am/pm' style label for hour h in [0..23]."""
    if h == 0:
        return "12 am"
    if h < 12:
        return f"{h} am"
    if h == 12:
        return "12 pm"
    return f"{h-12} pm"

def make_xtick_labels(ticks, start_hour):
    """Shift ticks by start_hour and format as 12-hour labels."""
    shifted = [ (t + start_hour) % 24 for t in ticks ]
    return [to_12h_label(h) for h in shifted]

def legend_weekend_weekday(ax, 
                           loc='upper left'):
    legend_elements = [
        Line2D([0], [0],
               marker='o',
               color='black',       # line color (unused here)
               markerfacecolor='black',
               markeredgecolor='black',
               linestyle='None',
               label='weekday'),
        Line2D([0], [0],
               marker='D',
               color='black',
               markerfacecolor='none',   # hollow marker
               markeredgecolor='black',
               linestyle='None',
               label='weekend')]
    
    ax.legend(handles=legend_elements, loc = loc)

def panel_csa_level(ax, 
                    df_csa, 
                    level, 
                    contact_metric= 'share', 
                    percent_yticks = True):
    '''
    compare the csa (contact share average)
    over different sparsity approaches for a given sparsity-level
    '''

    def get_csa(df_csa, 
                l,
                wp = 'weekday',
                s = 'Data driven',
                hour_order = range(24)):
        '''
        get csa values for a specific (wp, s, l)
        hour_order : ordering of the hour probabilities
        '''
        
        df_wp = df_csa[df_csa['weekperiod']==wp]
        df_sl = df_wp[(df_wp['sparsity']==s) & (df_wp['sparsity_level'] == str(l))]
    
        df_sl = df_sl.set_index('hourofday').loc[hour_order]
        return df_sl[contact_metric].values
    
    start_hour = 8
    hour_order = list(range(start_hour,24)) + list(range(0,start_hour))
    
    for marker, wp in zip(['o','D'], 
                          ['weekday','weekend']):
    
        x = range(24)
        for s in config.List_ss_rename:
            y_csa = get_csa(df_csa, 
                            l = level,
                            wp = wp, 
                            s = s,
                            hour_order= hour_order)
    
            c = config.DICT_colors_ss[s]
            draw_filled_line(ax, 
                             x, y_csa, 
                             color = c ,
                             base=0, alpha= 0.1, lw=.5, s=20, 
                             marker= marker, 
                             facecolor = c if (wp =='weekday') else 'none')
    
    _xticks = [0, 6, 12, 18, 23]
    _xticks_labels = make_xtick_labels(_xticks, start_hour)
    ax.set_xticks(_xticks)
    ax.set_xticklabels(_xticks_labels, rotation=45)
    
    # titles / labels
    ax.set_xlabel("hour of day")
    if percent_yticks: 
        set_percent_yticks(ax, decimals=0)

    ax.grid(axis="x", linestyle="-", color="gray", alpha=0.7)

def rsq(df, col1, col2, corr_type = 'pearson_r2'):
    if corr_type =='pearson_r2':
        return (df[col1].corr(df[col2]))
    else:
        return r2_score(df[col1], df[col2])

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
    restyle_ax(ax, label_size=config.ax_tick_size)
    
    # ensure horizontal alignment
    for label in cbar.ax.get_xticklabels():
        label.set_rotation(0)

def _binned_percentile_yvals(df, 
                             x_col, y_col, 
                             x_bins, 
                             percentile,
                             use_weight=True, 
                             weight_col="weight"):
    """
    Return a 1D float array of length len(x_bins) with NaN at the left edge.
    """
    df = df[[x_col, y_col] + ([weight_col] if use_weight else [])].dropna()
    x_idx = np.digitize(df[x_col].to_numpy(), x_bins)

    def _to_scalar(q):
        # coerce np.array / np.ndarray / pandas scalar to Python float or np.nan
        if q is None:
            return np.nan
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        return q_arr[0] if q_arr.size else np.nan

    q_by_xbin = {}
    p = float(percentile) / 100.0

    for i in range(1, len(x_bins)):
        mask = (x_idx == i)
        if not mask.any():
            q_by_xbin[i] = np.nan
            continue

        vals = df.loc[mask, y_col].to_numpy()
        if use_weight:
            w = df.loc[mask, weight_col].to_numpy()
            q = DescrStatsW(data=vals, weights=w).quantile(probs=p, return_pandas=False)
        else:
            q = np.nanquantile(vals, p)

        q_by_xbin[i] = _to_scalar(q)

    # prepend NaN for the left edge so indexing aligns with bins
    y_vals = [np.nan] + [q_by_xbin.get(i, np.nan) for i in range(1, len(x_bins))]
    
    return np.asarray(y_vals, dtype=float)

def plot_binned_percentile(ax,
                           _df,
                           x_col, y_col,
                           x_bins, y_bins,
                           percentile=50,
                           color='cyan',
                           linestyle='-',
                           linewidth=2,
                           scatter=False,
                           scatter_size=1,
                           use_weight = True, 
                           weight_col = "weight",
                           bin_non_linear=False,   # <--- NEW
                           no_binning = False,
                           _plot = True,
                           **kwargs):
    """
    Plot a percentile line over a binned heatmap.
    If bin_non_linear=True, y is mapped to bin coordinates using the actual (possibly non-uniform) y_bins.
    Otherwise, a linear rescale (uniform bins) is applied as before.
    """
    # 1) Get percentile y at each x-bin (in DATA units)
    y_vals = _binned_percentile_yvals(_df, x_col, y_col, x_bins, 
                                      percentile, 
                                      use_weight = use_weight, 
                                      weight_col = weight_col)
    
    if no_binning:
        ax.scatter(x_bins, y_vals, color=color, s=scatter_size)
    else:
        # 2) Map y to heatmap coordinates
        if bin_non_linear:
            y_rescaled = map_values_to_bincoords(y_vals, y_bins, fractional=True)
        else:
            y_rescaled = rescale_to_bins(y_vals, y_bins)
    
        # 3) x mapping (keep your original)
        x_rescaled = (x_bins - x_bins[0]) / (np.diff(x_bins)[1]) - 0.5
    
        # 4) draw
        if _plot:
            ax.plot(x_rescaled, y_rescaled, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)
        if scatter:
            ax.scatter(x_rescaled, y_rescaled, color=color, s=scatter_size)

def panels_missing_users_detected_contacts(axes, 
                                           df_merged, 
                                           l):
    '''
    visualize detected contacts vs. missing users
    for a given range under different sparsity approaches
    '''
    _LS = config.List_ss_rename
    x_metric= 'missing_users_perc'

    for ax, s in zip(axes, _LS):
        
        df_s = df_merged[df_merged['sparsity'] == s]
        dict_level_df_s = subset_df_feature(df_s, 'sparsity_level').copy()
    
        #setting the colorbar for the hour
        start_hour = 8
        df_sl = dict_level_df_s[str(l)].copy()
        df_sl['hourofday'] += (24 - start_hour)
        df_sl['hourofday'] %= 24
    
        scatter_df(ax, 
                   df_sl, 
                   x = x_metric, 
                   y = 'count_contacts',
                   y_rename = '', 
                   x_rename = '',
                   s = .1,
                   #c = DICT_colors_ss[s])
                   c = df_sl['hourofday'],
                   cmap = 'Blues', 
                   colorbar=False)
    
        bin_users = np.arange(.0,.9,.05) 
        bin_contacts = np.linspace(10,1e3, 20)

        plot_binned_percentile(ax,
                               df_sl,
                               x_metric, 
                               'count_contacts', 
                               bin_users, 
                               bin_contacts, 
                               percentile=50,
                               color='black',
                               linestyle = '-',
                               linewidth = 2,
                               scatter=True,
                               scatter_size = 5,
                               use_weight = False, 
                               weight_col = "weight",
                               bin_non_linear=True,   # <--- NEW
                               _plot = False, 
                               no_binning = True)
    
        
        ax.set_yscale('log')
        
        #ax.set_xlim(0.2,0.75)
        ax.set_ylim(2,1.2e3)
    
        R2 = rsq(df_sl, 
                 x_metric, 
                 'count_contacts')
        
        ax.text(0.98, 0.02,                      # (x,y) bottom-right
                fr"$\rho = {R2:.2f}$",         # latex-style rho^2
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=10,
            )
        ax.set_title(s)
    
        if s != _LS[-1]:
            ax.set_xlabel('')
            #remove_axis_ticktext(ax, axis= 'x')

    axes[0].set_ylabel('detected contacts')
    axes[1].set_xlabel('missing users (%)')

#########################################
#### GAP DISTRIBUTION AND ENTROPIES #####
#########################################

def panel_gap_distribution(ax, 
                           df_gap_count,
                           level= '40-50',
                           fill_alpha = 0.18,
                           hours_select = range(1,13), 
                           ax_ticks = True):
    '''
    visualize the gap duration distribution for a sample of sequences withi a range of missing hours
    df_gap_count: df of gap durations
    level: level of missing hours (from 0-10 to 90-100)
    '''

    
    
    #subset the gap count dataframe by the level of missing hours                    
    df_gap_count_level = df_gap_count[df_gap_count['missing_hours'] == level]
    
    df_gcl = df_gap_count_level.pivot(index = 'gap_duration_hours', 
                                      columns = 'sparsity', 
                                      values = 'count').fillna(0)
    
    #normalize column counts to obtain the probability for each gap duration
    df_gcl = df_gcl.div(df_gcl.sum(axis=0), axis=1)
    
    
    
    for s in ['Data driven', 'Random uniform']:
        y = df_gcl.loc[hours_select, s].values
        x = np.arange(len(y))  # 0..11
        
        # area
        ax.fill_between(
            x, 0, y,
            color = config.DICT_colors_ss[s],
            alpha = fill_alpha,
            zorder = 0
        )
    
        # line
        ax.plot(
            x, y,
            color=config.DICT_colors_ss[s])
        
        # points
        ax.scatter(
            x, y,
            color=config.DICT_colors_ss[s],
            s=10)

    if ax_ticks: 
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y*100:.0f}" for y in yticks])
        ax.set_ylim(0,.8)
        ax.set_ylim(1e-5,1.1)
        ax.set_yscale('log')   # for y-axis
        
        # choose the major tick positions
        major_ticks = [0, 5, 11]#, 17, 23]
        # set axis limits that include all major ticks
        ax.set_xlim(-.25, 11)
        # major ticks and labels
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([str(t+1) for t in major_ticks])
        # minor ticks (optional): every hour
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        # style
        ax.tick_params(axis="x", which="major", length=6, width=1)
        ax.tick_params(axis="x", which="minor", length=3, width=0.8)

def plot_kde(ax, x, color, label=None, fill_alpha=0.35, ls='-', lw=1.5, z=1):
    '''
    plot gaussian kernel density estimation for a given array
    '''
    from scipy.stats import gaussian_kde
    x = np.asarray(x.dropna())
    kde = gaussian_kde(x)
    xx = np.linspace(x.min(), x.max(), 500)
    yy = kde(xx)
    if fill_alpha > 0:
        ax.fill_between(xx, yy, 0, color=color, alpha=fill_alpha, zorder=z)
    ax.plot(xx, yy, color=color, lw=lw, ls=ls, zorder=z+1, label=label)
    return xx, yy

def panel_sequence_entropies(ax,
                             df_sequence_entropies,
                             level_missing_hours = '40-50',
                             alpha = 0.35):

    df_sequence_entropies_level = df_sequence_entropies[df_sequence_entropies['missing_hours'] == level_missing_hours]

    for s in ['Data driven', 'Random uniform']:

        plot_kde(ax,
                 df_sequence_entropies_level[s],
                 color= config.DICT_colors_ss[s],
                 fill_alpha=alpha)


################################    
##### CALIBRATION OUTCOMES #####
################################

def visual_grid_R0_global(ax, 
                          df_GRID_sub, 
                          x = 'beta', 
                          y = 'gamma',
                          cmap = 'Greys', 
                          R0_column = 'R0_global_mean',
                          levels = 20, 
                          R0_min =0, 
                          R0_max =7,
                          cbar = True, 
                          cbar_title = 'mean R0',
                          contour_lines=None,  # pass a list of R0 values for contour lines
                          contour_line_color='k',
                          contour_line_style='--', 
                          manual_clabel = False, 
                          alpha = 1, 
                          inset_kw_arg = None, 
                          symmetric_cbar = False,
                          sym_linthresh = 1e-3):

    from matplotlib.colors import Normalize, SymLogNorm
    from matplotlib.cm import ScalarMappable
    
    # ---- choose normalization ----
    if symmetric_cbar:
        vmax_sym = max(abs(R0_min), abs(R0_max))
        norm = SymLogNorm(linthresh=sym_linthresh, vmin=-vmax_sym, vmax=vmax_sym, base=10)
        max_pow = int(np.floor(np.log10(vmax_sym)))
        pos_ticks = [10**k for k in range(0, max_pow + 1)]
        neg_ticks = [-t for t in reversed(pos_ticks)]
        ticks = neg_ticks + [0] + pos_ticks
    else:
        ticks = np.arange(R0_min, R0_max + 1, 1)
        norm = Normalize(vmin=R0_min, vmax=R0_max)
        
    # Filled contour plot
    tcf = ax.tricontourf(
        df_GRID_sub[x],
        df_GRID_sub[y],
        df_GRID_sub[R0_column],
        levels=levels,
        cmap=cmap, 
        norm = norm,
        vmin=0, 
        vmax=R0_max + 1,
        alpha=alpha)
    
    # Add contour lines if requested
    if contour_lines is not None:
        contours = ax.tricontour(
            df_GRID_sub[x],
            df_GRID_sub[y],
            df_GRID_sub[R0_column],
            levels=contour_lines,
            colors=contour_line_color,
            linestyles=contour_line_style,
            linewidths=1, 
            alpha = alpha)
        
        offset_x = 0
        offset_y = 0
        if manual_clabel is not False:
            labels = ax.clabel(contours, inline=True, fontsize=10, 
                               manual = manual_clabel)
            for txt in labels:
                x, y = txt.get_position()
                txt.set_position((x + offset_x, y + offset_y))  # offset_x > 0 moves to the rightc
                #txt.set_color('white')
    
    if cbar:
        inset_kw = dict(
            width="40%",
            height="4%",
            loc='lower center',
            bbox_to_anchor=(0.1, 0.7, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=1
        )
        if inset_kw_arg is not None:
            inset_kw.update(inset_kw_arg)
        cax = inset_axes(ax, **inset_kw)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        # Create horizontal colorbar using dummy ScalarMappable
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')

            
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:g}" for t in ticks])
    
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.set_label(
            cbar_title,
            size=12,
            labelpad=5,
            bbox=dict(facecolor='white', edgecolor='none', pad=1)
        )
    
        for tick in cbar.ax.xaxis.get_ticklabels():
            tick.set_color('white')

def visual_GRID_R0(ax, 
                  df_GRID, 
                  x = 'gamma',
                  y = 'beta',
                  R0_column = 'R0_global_mean',
                  beta_range = None,
                  gamma_range = None, 
                  grid_levels = 4, 
                  contour_lines = None, 
                  manual_clabel = False,
                  cmap = 'gist_yarg',
                  cbar = False, 
                  cbar_title = '',
                  scatter_size =1, 
                  alpha =1, 
                  R0_min = 1,
                  R0_max = 7,
                  inset_kw_arg = None, 
                  symmetric_cbar = False): 

    if beta_range is not None:
        df_GRID = df_GRID[df_GRID['beta'].between(beta_range[0], beta_range[1])]
    if gamma_range is not None:
        df_GRID = df_GRID[df_GRID['gamma'].between(gamma_range[0], gamma_range[1])]

    visual_grid_R0_global(ax, 
                          df_GRID, 
                          R0_column = R0_column,
                          levels = grid_levels,
                          R0_min = R0_min,
                          R0_max = R0_max,
                          contour_lines = contour_lines,
                          cbar = cbar,
                          cbar_title = cbar_title,
                          x = x, 
                          y = y, 
                          manual_clabel = manual_clabel,
                          cmap = cmap, 
                          alpha =alpha, 
                          inset_kw_arg = inset_kw_arg,
                          symmetric_cbar = symmetric_cbar)
    
    scatter_df(ax, 
            df_GRID, 
            x, 
            y, 
            s = scatter_size,
            c = df_GRID[R0_column].values, 
            cmap = plt.cm.gist_yarg, 
            vmin =0,
            vmax=4,
            colorbar = False,
            colorbar_label = '$\overline{R}_0$')#fontsize_cbar = 20)

def visual_pars_med_ci(ax,
                       df_pars_stats, 
                       x = 'beta',
                       y = 'gamma',
                       capsize=4,
                       elinewidth=1.2,
                       markeredgewidth=0.80):
    '''
    visualize median and 95%CI 
    of pars (x,y) for different levels of sparsity    
    '''

    df = df_pars_stats.copy()
    df = df.loc[[str(l) for l in config.Levels]]
       
    #Median values
    X = df[f'{x}_med'].values
    Y = df[f'{y}_med'].values

    #Confidence interval values
    X_wl = df[f'{x}_whislo'].values
    Y_wl = df[f'{y}_whislo'].values
    X_wh = df[f'{x}_whishi'].values
    Y_wh = df[f'{y}_whishi'].values
    
    for i, col in enumerate(config.COLORS_LEVEL):
        ax.errorbar(X[i], Y[i],
                    xerr=[[X[i] - X_wl[i]], [X_wh[i] - X[i]]],
                    yerr=[[Y[i] - Y_wl[i]], [Y_wh[i] - Y[i]]],
                    fmt= 'o',
                    color=col,
                    ecolor=col,
                    capsize= capsize,
                    elinewidth= elinewidth,
                    markeredgewidth= markeredgewidth)

def visual_gt_par(ax, 
                  groundtruth_pars, 
                  x = 'beta', 
                  y = 'gamma', 
                  s = 30):
    '''
    visualization of the ground truth parameters 
    '''
    ax.scatter(
        groundtruth_pars[x],
        groundtruth_pars[y],
        facecolors=config.COLOR_GT,
        s=s,
        marker='D',
        edgecolors=config.COLOR_GT,
        zorder=999)

def visual_fitted_params(ax, 
                         df_fpc_stats, 
                         df_R0_grid_estimates, 
                         beta_grid_range, 
                         gamma_grid_range,
                         R0_min, 
                         R0_max,
                         groundtruth_pars,
                         grid_levels,
                         List_mc): 

    '''
    df_fpc : stats of fitted parameters
    df_R0_grid_estimates: grid of esitimated R0 value
    (beta_grid_range, gamma_grid_range): select grid for the range
    (R0_min, R0_max): min, max R0 for the colorbar 
    groundtruth_pars: dictionary containing the ground truth simulation parameters
    grid_levels : number of plotted levels over the contour plot of the R0 heatmap
    List_mc : list of ax coordinates for labeling of the contour lines (has same len of grid_levels)
    '''
    #visualize median and confidence interval of the estimated parameters
    visual_pars_med_ci(ax,
                       df_fpc_stats, 
                       x = 'beta',
                       y = 'gamma',
                       capsize=4,
                       elinewidth=1.2,
                       markeredgewidth=0.80)
    
    #visualize the underlying grid of R0 values 
    visual_GRID_R0(ax, 
                  df_R0_grid_estimates, 
                  y = 'gamma',
                  x = 'beta',
                  cmap = 'Blues',
                  R0_column = 'R0_mean',
                  beta_range = beta_grid_range,
                  gamma_range = gamma_grid_range, 
                  R0_min = R0_min,
                  R0_max= R0_max,
                  grid_levels = grid_levels, 
                  contour_lines = grid_levels, 
                  manual_clabel = List_mc,
                  cbar = False, 
                  #cbar_title = 'ground truth $\overline{R}_0$',
                  scatter_size=.001)

    #visualize the ground truth parameters 
    visual_gt_par(ax, 
                  groundtruth_pars, 
                  x = 'beta', 
                  y = 'gamma', 
                  s = 30)


################################    
##### OUTCOMES FROM CUEBIQ #####
################################

def process_cov_share_data(cov_share_sw, 
                           sws = 30):
    '''
    process the data of coverage share from cuebiq for a given sliding window renaming consistently the variables
    sws: sliding window size (possible values are 7, 14, 21, 30, 50)
    '''

    cov_share_sw['DATE'] = pd.to_datetime(cov_share_sw['DATE']) 
    
    cols_old = ['DATE', 
                '0.9–1.0', '0.8–0.9', '0.7–0.8', '0.6–0.7', '0.5–0.6','0.4–0.5', '0.3–0.4', '0.2–0.3', '0.1–0.2', '0.0–0.1', 
                'WINDOW_DAYS']
    
    cols_new = ['DATE', 
                '0-10', '10-20', '20-30', '30-40', '40-50','50-60', '60-70', '70-80', '80-90', '90-100', 
                'WINDOW_DAYS']
    
    dict_rename_cols = dict(zip(cols_old, cols_new))
    cov_share_sw.columns = [dict_rename_cols[c] for c in cov_share_sw.columns]
    
    sw_sizes  = cov_share_sw['WINDOW_DAYS'].unique()
    sws_share = cov_share_sw[cov_share_sw['WINDOW_DAYS'] == sws].set_index('DATE').drop('WINDOW_DAYS', 
                                                                                        axis = 1)
    
    return sws_share

def viz_coverage(ax,
                 cov_share_sw, 
                 sws = 7, 
                 date_range = None, 
                 date_step =10):

    sws_share = cov_share_sw[cov_share_sw['WINDOW_DAYS'] == sws].set_index('DATE').drop('WINDOW_DAYS', axis = 1)
    df = sws_share.copy()
    if date_range is not None:
        df = df.loc[date_range]
    
    cmap = cm.get_cmap("coolwarm")     # blue→red
    colors = [cmap(i) for i in np.linspace(0, 1, df.shape[1])]
    
    df.plot(kind='bar',
            stacked=True,
            ax=ax,
            width=1.1,
            color=colors)
        
    df.index = pd.to_datetime(df.index)
    
    ax.set_xticks(range(0, len(df), date_step))
    ax.set_xticklabels(df.index[::date_step].strftime("%m-%d"), rotation=90)
    ax.set_title(f'{sws} days')

def convert_mmdd_to_ddmon(ax):
    """Convert xticklabels from 'MM-DD' to 'DD Mon' format."""
    labels = ax.get_xticklabels()
    new_labels = []
    
    for lbl in labels:
        txt = lbl.get_text()
        if "-" in txt:
            try:
                m, d = txt.split("-")
                date = dt.datetime(2000, int(m), int(d))   # dummy year
                new_labels.append(date.strftime("%d %b"))
            except:
                new_labels.append(txt)
        else:
            new_labels.append(txt)

    ax.set_xticklabels(new_labels)

def panel_spectus_curves(ax_top, 
                         ax_left, 
                         ax_right, 
                         curves, 
                         Curve_ref):
    
    #[0] TOP LEGEND
    legend_elements = [
        Patch(facecolor= config.COLOR_GT, label = 'Reference'),
        Patch(facecolor= config.COLOR_CALIB_BS ,label = 'Calib. on biased contacts'),
        Patch(facecolor= config.COLOR_CALIB_CC ,label = 'Calib. on rescaled contacts')]
    
    ax_top.legend(handles = legend_elements,
                  title = '',
                  loc = 'center',
                  ncol = 3,                # two columns
                  columnspacing = 1.5,     # space between columns
                  handletextpad = 0.5,
                  fontsize = 9,
                  framealpha=1,         # fully opaque
                  facecolor="white",    # white background
                  edgecolor="black")     # optional: black border

    ax_top.axis('off')
    
    #[1] CURVES AND BOXPLOT OF TOTAL INFECTED
    axes = [ax_left, ax_right]
    alpha = 0.4

    ax = axes[0]
    _m = 'percentage'
    t  = 'calib_sparse'
    c_tm = curves[(curves['TYPE'] == t) & (curves['REFERENCE']== _m)].copy()
    c_tm['CI'] /= 855
    c_tm_list = [k[['S','CI']].values for c,k in subset_df_feature(c_tm, 'N_iter').items()]
    
    visual_curves_SI_spectus(ax, c_tm_list, visual_all_sims= True, color = config.COLOR_CALIB_BS, alpha = alpha)
    
    t  = 'calib_cc'
    c_tm = curves[(curves['TYPE'] == t) & (curves['REFERENCE']== _m)].copy()
    c_tm['CI'] /= 855
    c_tm_list = [k[['S','CI']].values for c,k in subset_df_feature(c_tm, 'N_iter').items()]
    visual_curves_SI_spectus(ax, c_tm_list, visual_all_sims= True, color = config.COLOR_CALIB_CC, alpha = alpha)
    remove_axis_ticktext(ax, axis='x')
    ax.set_xticks(range(0,90,20))
    ax.set_xticklabels(range(0,90,20), rotation =0)
    ax.plot(Curve_ref.values, c = config.COLOR_GT)
    ax.set_xlabel('simulation day')
    ax.set_xlim(0,89)

    ax = axes[1]
    
    _m = 'percentage'
    
    t  = 'calib_sparse'
    c_tm = curves[(curves['TYPE'] == t) & (curves['REFERENCE']== _m)].copy()
    c_tm['CI'] /= 855
    CI_sparse = c_tm.groupby('N_iter').agg({'CI':max})
    CI_sparse_stats = {f:c(CI_sparse['CI']) for f,c in funcs.items()}
    
    t  = 'calib_cc'
    c_tm = curves[(curves['TYPE'] == t) & (curves['REFERENCE']== _m)].copy()
    c_tm['CI'] /= 855
    CI_cc = c_tm.groupby('N_iter').agg({'CI':max})
    CI_cc_stats = {f:c(CI_cc['CI']) for f,c in funcs.items()}
    
    custom_boxplot_from_stats(ax, 
                              [CI_sparse_stats], 
                              positions=[0], 
                              color = config.COLOR_CALIB_BS, face_alpha = .8)
    
    ax.scatter(0, CI_sparse_stats['mean'], color = 'black', s= 20, zorder= 100)
    
    custom_boxplot_from_stats(ax, 
                              [CI_cc_stats], 
                              positions=[1], 
                              color = config.COLOR_CALIB_CC, face_alpha = .8)
    
    ax.scatter(1, CI_sparse_stats['mean'], color = 'black', s= 20, zorder = 100)
    remove_axis_ticktext(ax, axis='x')
    ax.set_ylabel('total infected (%)')
    
    axes[0].set_ylabel('cumulative infected (%)')
    set_percent_yticks(axes[0], decimals= 0)
    set_percent_yticks(axes[1], decimals= 0)
    
    ax.axhline(Curve_ref.values[-1], color = config.COLOR_GT)


