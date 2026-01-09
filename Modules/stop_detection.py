import pandas as pd
from scipy.spatial.distance import pdist, cdist
import numpy as np
import math
import datetime as dt 
import matplotlib.pyplot as plt
import itertools

##########################################
### COORDINATE CONVERSION FUNCTIONS ######
##########################################

def ConvertCoordinate_3587(lat, lon):
    '''
    conversion from crs:4326 to crs:3587
    '''   
    latInEPSG3857 = (math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)) * (20037508.34 / 180)
    lonInEPSG3857 = (lon* 20037508.34 / 180)
    
    return latInEPSG3857, lonInEPSG3857

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
        #j_star is the candidate 'stop end'
        timestamp_i = traj['unix_timestamp'].iat[i]
        j_star = next((j for j in range(i, len(traj)) if traj['unix_timestamp'].iat[j] - timestamp_i >= dur_min * 60), -1) 
        
        #initial diameter
        D_start = diameter(coords[i:j_star+1])

        #CONDITIONS BLOCKING THE SEARCH
        #conditions to block the j_star search
        Cond_exhausted_traj = j_star==-1 
        #condition that diameter is over the delta_roam threshold
        Cond_diam_OT = D_start > delta_roam 
        #condition that there is at least a consecutive ping pair that has a time-separation greater than dt_max
        Cond_cc_diff_OT = (traj['unix_timestamp'][i:j_star+1].diff().dropna() >= dt_max*60).any()

        #[STEP2] - decide whether index i is a 'stop' or 'trip' ping 
        if Cond_exhausted_traj or Cond_diam_OT or Cond_cc_diff_OT:
            #DISCARD i and j_star as candidate 'stop start' and 'stop end' pings 
            #move forward and update i, starting from scratch
            i += 1
        else:
            #SELECT i as a 'stop start' ping AND 
            #[STEP3] proceed with the iterative search of 'stop end'
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
