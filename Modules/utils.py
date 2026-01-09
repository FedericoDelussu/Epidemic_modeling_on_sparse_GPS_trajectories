import pandas as pd
import numpy as np
import os 
import pickle
import datetime as dt

#TMP folder where results are saved
DICT_paths = {'TMP': '/home/fedde/work/Project_Penn/TMP/', 
              'Data': '/home/fedde/work/Project_Penn/Data/'}

def get_table_count(df,x,y):
    return df.groupby([x,y]).size().reset_index().pivot(index= x, columns = y, values = 0).fillna(0)

def get_weekday(d):
    w_days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    return w_days[d.weekday()]

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

def subset_df_feature(df,f):
    '''
    df : dataframe
    f  : feature name 
    '''
    #feature unique values 
    f_vals = df[f].unique()
    #subset dataframe according to feature f records 
    return {v: df[df[f]==v] for v in f_vals}

def gen_fold(FOLD):
    if os.path.exists(FOLD):
        print(f"Directory '{FOLD}' already exists. No action taken.")
        return
    try:
        os.mkdir(FOLD)
        print(f"Directory '{FOLD}' created successfully.")
    except Exception as e:
        print(f"Failed to create directory '{FOLD}': {e}")
        
def stack_dict_to_df(DICT_contacts, f_name = 'level'):
    rows = []
    for key, df in DICT_contacts.items():
        df = df.copy()
        df[f_name] = str(key)
        rows.append(df)
        
    stacked_df = pd.concat(rows, axis=0)#, ignore_index=True)
    
    return stacked_df


def read_folder_files(folder_path, 
                      f_name=None,
                      Cols_select=None, 
                      FILES_select=None,
                      parse_dates_list=None, 
                      index_col=None, 
                      dtype = None, 
                      Date_range = None):
    '''
    Read a collection of CSV files into a single dataframe, concatenating as they are read
    folder_path : path of the folder collecting the CSV dataframes

    Date_range : filters the dataset based on the datetime columns
    '''
    
    combined_df = pd.DataFrame()  # Initialize an empty dataframe

    # List all files in the folder
    FILES = os.listdir(folder_path)
    # If specific files are provided, select them
    if FILES_select is not None:
        FILES = FILES_select

    # Iterate over each file in the folder
    for file_name in FILES:

        # Check if the file is a CSV file
        if file_name.endswith('.csv'):  
            
            # Read the CSV file into a dataframe
            df = pd.read_csv(os.path.join(folder_path, file_name), 
                             usecols=Cols_select, 
                             index_col=index_col, 
                             parse_dates=parse_dates_list, 
                             dtype = dtype)

            if Date_range is not None:
                df = df[df.datetime.between(Date_range[0], Date_range[-1], inclusive = 'left')]
                
            # Add a new column with the file name if f_name is specified
            if f_name is not None: 
                df[f_name] = file_name.split('.csv')[0]

            # Concatenate the new dataframe to the combined dataframe
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

def convert_to_percent_range(input_str):
    # Remove parentheses and split the string
    numbers = input_str.strip('()').split(',')
    # Convert the numbers to percentage format
    lower = int(float(numbers[0]) * 100)
    upper = int(float(numbers[1]) * 100)
    # Return the formatted string
    return f'{lower}-{upper}%'


def gen_str_exp(par_lach, par_search):
    '''
    string of experiment with:
    - lachesis parameters for stop detection
    - search parameters for identifying completed users
    '''

    ghr, SW_width_days = par_search
    dur_min, dt_max, delta_roam = par_lach
    str_lach  = f'mintime_{dur_min}_maxdtime_{dt_max}_maxdiam_{delta_roam}'
    str_search = f"ghr{ghr}_days{SW_width_days}"
    str_exp = f"LACH_{str_lach}_SEARCH_{str_search}"
    return str_exp


def gen_str_exp_sparse(P, t_res): 
    '''
    P : sparsity level for empirical gap sampling
    t_res : temporal resolution

    sparsity experiment string identifier
    '''
    
    str_exp_sparse = f'P_{P}_{t_res}'
    
    return str_exp_sparse

#############################
##### IMPORT FUNCTIONS ######
#############################

def import_loc_1min_data(USERS, Cols_select = None, DICT_dtypes = None, Date_range = None):
    '''
    import original location data at 1 minute resolution
    '''

    #IMPORT DATA FOR COMPLETED USERS
    path = DICT_paths['TMP'] + 'f_005_D1_filtering/filtered_traj_1min_downsample/'
    USERS_select_files = [f"{u}.csv" for u in USERS]

    Cols_select_default = ['id', 'datetime','lat','lng']
    DICT_dtypes_default = {'id':'int32', 
                           'lat': 'float32',
                            'lon': 'float32'}

    if Cols_select is not None: 
        DICT_dtypes = {v:k for v,k in DICT_dtypes_default.items() if v in Cols_select}
        
    else:
        Cols_select = Cols_select_default
        DICT_dtypes = DICT_dtypes_default
            
    df_US = read_folder_files(path, 
                              f_name = None, 
                              Cols_select = Cols_select,
                              FILES_select = USERS_select_files, 
                              parse_dates_list= ['datetime'],
                              index_col = None, 
                              Date_range = Date_range,
                              dtype = DICT_dtypes)
    
    return df_US

def import_country_shp(ISO3): 
    '''
    given the GID_2,
    import the associated shapefile data 
    '''
    
    #path with country shapefiles
    path_data_fede = '/data/work/fedde/Project_PM25/'
    path_results = path_data_fede + "Results/"
    path = path_results + 'Data/23_03_01_data_loc_phy_poll_1h/'
    path_shp_country = path + 'Info/Country_GID_2_shapefiles/'    
    df_shp_ISO3 = gpd.read_file(path_shp_country + ISO3 + '/' + ISO3 + '.shp')
    
    return df_shp_ISO3

def import_data_LACH_stops(par_lach):
    '''
    par_lach : (dur_min, dt_max, delta_roam)
    '''
    dur_min, dt_max, delta_roam = par_lach
    str_lach  = f'mintime_{dur_min}_maxdtime_{dt_max}_maxdiam_{delta_roam}'
    PATH_save = DICT_paths['TMP'] + 'f_007_D1_lachesis_stops_AND_loc-1hdi/'

    df_stops = pd.read_csv(PATH_save + 'df_stops_' + str_lach + '.csv',
                           index_col = 0, 
                           parse_dates = ['start_time', 'end_time'])
    return df_stops 

def import_data_LACH_LOC_1hdi(par_lach):
    '''
    par_lach : (dur_min, dt_max, delta_roam)
    '''
    
    
    dur_min, dt_max, delta_roam = par_lach
    str_lach  = f'mintime_{dur_min}_maxdtime_{dt_max}_maxdiam_{delta_roam}'
    PATH_save = DICT_paths['TMP'] + 'f_007_D1_lachesis_stops_AND_loc-1hdi/'

    #1HOUR-INTERPOLATED LOCATION 
    df_LOC_1hdi = pd.read_csv(PATH_save + 'df_LOC_1hdi_' + str_lach + '.csv', 
                              index_col = 0, 
                              parse_dates = ['date_hour'])
    return df_LOC_1hdi


#def import_data_CONTACT(par_lach, 
#                        par_search): 
#    '''
#    par_lach : (dur_min, dt_max, delta_roam)
#    par_search: (ghr, SW_width_days)
#    '''
#    
#    ghr, SW_width_days = par_search
#    dur_min, dt_max, delta_roam = par_lach
#    str_lach  = f'mintime_{dur_min}_maxdtime_{dt_max}_maxdiam_{delta_roam}'
#    str_search = f"ghr{ghr}_days{SW_width_days}"
#    str_exp = f"LACH_{str_lach}_SEARCH_{str_search}"
#
#    PATH_save = DICT_paths['TMP'] + 'f_008_D1_contact_estimation/'
#
#    df_ch = pd.read_csv(PATH_save + 'df_contacts_' + str_exp +'.csv',
#                        index_col = 0,
#                        parse_dates = ['date_hour'])
#    
#    return df_ch


def import_data_USERS_SELECT(par_lach, 
                             par_search): 
    '''
    par_lach : (dur_min, dt_max, delta_roam)
    par_search: (ghr, SW_width_days)
    '''

    ghr, SW_width_days = par_search
    dur_min, dt_max, delta_roam = par_lach
    str_lach  = f'mintime_{dur_min}_maxdtime_{dt_max}_maxdiam_{delta_roam}'
    str_search = f"ghr{ghr}_days{SW_width_days}"
    str_exp = f"LACH_{str_lach}_SEARCH_{str_search}"

    PATH_save = DICT_paths['TMP'] + 'f_008_D1_contact_estimation/'

    df_select = pd.read_csv(PATH_save + 'df_users_select_' + str_exp + '.csv', 
                            index_col = 0, 
                            parse_dates = ['date_start','date_end'])
    
    return df_select


def import_gap_data(par_lach, par_search, 
                    Metrics_ranges, 
                    metric = 'perc-missing-hours', 
                    t_res = '1hour', 
                    type = 'mask'):
    '''
    import gap masks for empirical sparsification
    '''

    Types = ['mask','stops', 'contact', 
             'loc_filtered', 'contact_marginal', 'contact_within']
    
    if type not in Types:
        print('wrong type required')
        return 
    
    FOLD_emp_sparse =  DICT_paths['TMP'] + 'f_010_D1_empirical_sparsification/'
    str_exp = gen_str_exp(par_lach, par_search)
    PATH_exp = FOLD_emp_sparse + str_exp

    if type!= 'contact':
        DICT_df= {}
        for range in Metrics_ranges[metric]:
            FOLD_save = f'{PATH_exp}/{t_res}_{metric}_{range[0]}_{range[1]}/'
            df = pd.read_csv(FOLD_save + f'df_{type}.csv', index_col = 0)
            DICT_df[range] = df
        return DICT_df

    else: 
        DICT_df= {}
        for range in Metrics_ranges[metric]:
            FOLD_save = f'{PATH_exp}/{t_res}_{metric}_{range[0]}_{range[1]}/'
            #join marginal and within contacts
            df_cm = pd.read_csv(FOLD_save + 'df_contact_marginal.csv', index_col = 0)
            df_cw = pd.read_csv(FOLD_save + 'df_contact_within.csv', index_col = 0)         
            Cs = ['u1','u2', 'date_hour', 'n_minutes']
            df_ctot = pd.concat([df_cm[Cs], df_cw[Cs]], axis=0)
            df_ctot = df_ctot.groupby(['u1','u2','date_hour'])['n_minutes'].sum().reset_index()
            DICT_df[range] = df_ctot 
            
        return DICT_df


def import_original_stops(par_lach, par_search): 
    '''
    import original stops within the study period identified by par_search
    '''

    def filter_stops(df_stops, 
                 USERS_select, 
                 Date_range, 
                 Cols_select = None, 
                 reset_ghr =None):
    
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
        df_sb1.loc[:,'end_time']   = pd.to_datetime(Date_range[1] - dt.timedelta(minutes=1))
        
        df_stops_DR = pd.concat([df_sb0, df_stops_DR, df_sb1], axis = 0)
    
        if reset_ghr is not None:
            df_stops_DR = df_stops_DR.rename(columns = {'geohash9': 'geohash'})
            df_stops_DR['geohash'] = df_stops_DR['geohash'].str[:reset_ghr]
    
        return df_stops_DR
        
    #import original stop data 
    PATH_save = DICT_paths['TMP'] + 'f_007_D1_lachesis_stops_AND_loc-1hdi/'
    dur_min, dt_max, delta_roam = par_lach
    str_lach  = f'mintime_{dur_min}_maxdtime_{dt_max}_maxdiam_{delta_roam}'
    df_stops = pd.read_csv(PATH_save + 'df_stops_' + str_lach + '.csv',
                               index_col =0, 
                               parse_dates = ['start_time', 'end_time'])
    
    df_select = import_data_USERS_SELECT(par_lach, par_search)
    USERS_select = df_select['user'].values
    Date_range = pd.DatetimeIndex([df_select.loc[0,'date_start'], df_select.loc[0,'date_end']])
    Cols_select = ['id', 'start_time', 'end_time', 'geohash9', 
                   'medoid_x', 'medoid_y', 
                   'diameter_m', 'n_pings', 'duration_s']
    df_stops_DR = filter_stops(df_stops, USERS_select, Date_range, Cols_select, reset_ghr = par_search[0])     

    return df_stops_DR

def import_original_contacts(par_lach, par_search):
    
    def compute_tot_contacts(df_cm,df_cw):
        '''
        df_cw : within-cell contacts
        df_cm : marginal contacts
        '''
        Cs = ['u1','u2', 'date_hour', 'n_minutes']
        df_ctot = pd.concat([df_cm[Cs], df_cw[Cs]], axis=0)
        df_ctot = df_ctot.groupby(['u1','u2','date_hour'])['n_minutes'].sum().reset_index()
        return df_ctot

    #import original contact data
    PATH_save = DICT_paths['TMP'] + 'f_008_D1_contact_estimation/'
    str_exp = gen_str_exp(par_lach, par_search)
    df_cwithin  = pd.read_csv(PATH_save + 'df_contacts_within_cell_' + str_exp +'.csv', index_col =0)
    df_chm_hour = pd.read_csv(PATH_save + 'df_contacts_marginal_' + str_exp +'.csv', index_col =0)
    df_contact = compute_tot_contacts(df_cwithin, df_chm_hour)

    return df_contact

def import_EXP_SPARSE_epid_data(DICT_pars):
    '''
    import data from epidemiological simulation for complete and sparse trajectories
    DICT_pars = {'search': (par_lach, par_search), 
                 'sparse': (metric, t_res), 
                 'epid'  : (scenario, p_init,N_iter)}
    '''
    
    par_lach, par_search     = DICT_pars['search']
    metric, t_res            = DICT_pars['sparse']
    scenario, p_init,N_iter  = DICT_pars['epid']

    FOLD_epid_model =  DICT_paths['TMP'] + 'f_011_D1_variation_epid-model_contacts/'
    str_exp = gen_str_exp(par_lach, par_search)
    PATH_exp = FOLD_epid_model + str_exp
    FOLD_save = f'{PATH_exp}/{t_res}_{metric}/'
    str_epid = f'{scenario}_pinit_{p_init}_Niter_{N_iter}'
    df_epid_stats = pd.read_csv(f'{FOLD_save}/epid-stats_{str_epid}.csv', index_col = 0)
    pickle_file = f'{FOLD_save}SI-curves_{str_epid}.pkl' 
    with open(pickle_file, "rb") as f:
        DICT_SI_curves = pickle.load(f)

    return df_epid_stats, DICT_SI_curves

def import_EXP_SPARSE_contact_data(DICT_pars, Metrics_ranges): 
    '''
    import data from contact estimation for complete and sparse trajectories
    DICT_pars = {'search': (par_lach, par_search), 
                 'sparse': (metric, t_res), 
                 'epid'  : (scenario, p_init,N_iter)}
    '''

    par_lach, par_search     = DICT_pars['search']
    metric, t_res            = DICT_pars['sparse']

    #IMPORT EMPIRICAL GAPS FOR SPARSIFICATION

    
    #IMPORT CONTACTS DATA (before and after sparsification)
    DICT_contacts = import_gap_data(par_lach, par_search, Metrics_ranges, 
                                    metric = metric, t_res = t_res,  
                                    type = 'contact')
    DICT_contacts = {str(k):v for k,v in DICT_contacts.items()}
    df_contacts_original = import_original_contacts(par_lach, par_search)
    DICT_contacts.update({'Complete' : df_contacts_original})
    
    df_contacts = stack_dict_to_df(DICT_contacts)
    
    return df_contacts
    
def import_all_contacts(FOLD_contacts, Period, inclusive = 'left'):
    '''
    Period : tuple including 2 date objects, 
    by default right border is not included

    Import, selecting from all contacts (estimated from LACH and ghr setup) between (2014,2,1) and (2015,2,1)
    '''

    def compute_tot_contacts(df_cm,df_cw):
        '''
        df_cw : within-cell contacts
        df_cm : marginal contacts
        '''
        Cs = ['u1','u2', 'date_hour', 'n_minutes']
        df_ctot = pd.concat([df_cm[Cs], df_cw[Cs]], axis=0)
        df_ctot = df_ctot.groupby(['u1','u2','date_hour'])['n_minutes'].sum().reset_index()
        
        return df_ctot
        
    Days = pd.date_range(Period[0], Period[-1], inclusive = 'left').astype('str')
    FILES_cwithin = [f'df_cwithin_{w}.csv' for w in Days]
    FILES_cmargin = [f'df_cmargin_{w}.csv' for w in Days]
    
    df_cwithin = read_folder_files(FOLD_contacts, 
                                   FILES_select = FILES_cwithin, 
                                   parse_dates_list = ['date_hour'],
                                   index_col = 0)
    
    df_cmargin = read_folder_files(FOLD_contacts, 
                                   FILES_select = FILES_cmargin, 
                                   parse_dates_list = ['date_hour'], 
                                   index_col = 0)
    df_contacts = compute_tot_contacts(df_cmargin, df_cwithin)

    return df_contacts

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

def _to_tabular(df: pd.DataFrame, column_format: str) -> str:
    df = _sanitize_df(df)
    # escape=False because we’ve already escaped/sanitized
    return df.to_latex(escape=False,
                       multicolumn=True,
                       multicolumn_format='c',
                       column_format=column_format,
                       index = True)






