import sys
from Modules.utils import * 
from Modules.analysis import *

path = sys.argv[0]
FOLD_save = sys.argv[1]

Files = np.sort(os.listdir(path))

#[0] Import location trajectories for each user
DICT_dtypes = {'id':'int32', 
               'datetime':'string', 
               'lat': 'float32',
               'lon': 'float32'}

#dataframe containing all location trajectories
df_LOC = read_folder_files(path, 
                           Cols_select = ['id','datetime', 'lat','lng'],
                           dtype = DICT_dtypes)

#[1] Preprocessing for launching Lachesis stop detection algorithm 
#Input dataset must have columns:
    # unix_timestamp : time in utc format (seconds)
    # (x,y) : (lon,lat) with 4326CRS are converted to 3587CRS (meters)
    # id: unique identifier of the user 

df_LOC.columns = ['id', 'datetime', 'lat', 'lon']
df_LOC['datetime'] = pd.to_datetime(df_LOC['datetime']).astype('int64') // 10**9
time, lat, lon = 'datetime', 'lat','lon'
c_dict = {time: 'unix_timestamp', lat: 'y', lon: 'x'}
df_LOC = df_LOC[['id',time,lat,lon]].rename(columns = c_dict)
convert_df_coord_3587(df_LOC, lat = 'y', lon = 'x')

#[2] Apply Lachesis stop detection for each user trajectory

#set of Lachesis parameters 
# dur_min    : minimum stop duration (minutes), 
# dt_max     : max consectuive ping time difference (minutes), 
# delta_roam : max diameter (meters)
S = (10,360, 50)     

dur_min, dt_max, delta_roam = S
df_stops = df_LOC.groupby('id').apply(lambda x: print(f"id: {x.name}") or lachesis(x, dur_min, dt_max, delta_roam))

#save the dataframe of stop tables
par_str = f'mintime_{dur_min}_maxdtime_{dt_max}_maxdiam_{delta_roam}'
df_stops.reset_index().rename(columns = {'level_1': 'stop_label'}).to_csv(FOLD_save + 'df_stops_' + par_str + '.csv')
