import pandas as pd
import numpy as np
import datetime
import pickle
from tqdm import tqdm
from estimations import Estimator
import os

"""
- Sort df by 'hack_license', 'pickup_datetime'
- Convert columns into datetime objects
- Create `sub_index` for later sorting
@Param
cleaned_trip_df: trip_cleaned.csv on the drive
"""
def preprocess_df(cleaned_trip_df):
    episode_data = cleaned_trip_df.copy()
    episode_data.sort_values(['hack_license', 'pickup_datetime'], inplace=True)
    episode_data['pickup_datetime'] = pd.to_datetime(episode_data['pickup_datetime'])
    episode_data['dropoff_datetime'] = pd.to_datetime(episode_data['dropoff_datetime'])
    episode_data['pickup_time'] = pd.to_datetime(episode_data['pickup_time']).dt.time
    episode_data['dropoff_time'] = pd.to_datetime(episode_data['dropoff_time']).dt.time
    episode_data['sub_index'] = episode_data.index 
    return episode_data


def _get_shift(time, weekday=True, pickup=True):
    if weekday:
        if pickup:
            if (time > datetime.time(23, 30)) | (time <= datetime.time(11, 30)):
                return 'A'
            else:
                return 'B'
        else:
            if (time > datetime.time(8, 30)) & (time <= datetime.time(20, 30)):
                return 'A'
            else:
                return 'B'            
    else: ## weekend
        if pickup:
            if (time > datetime.time(2, 0)) & (time <= datetime.time(14, 0)):
                return 'A'
            else:
                return 'B'
        else:
            if (time > datetime.time(10, 0)) & (time <= datetime.time(22, 0)):
                return 'A'
            else:
                return 'B'        
        return None
    
def assign_shift(episode_data):
    ## determine shift for weekdays
    episode_data.loc[(episode_data['pickup_weekday']==1) &
                          (episode_data['first_pickup']==1) , '_PU_shift'] = episode_data['pickup_time']\
                                    .apply(lambda x: _get_shift(x, weekday=True, pickup=True))
    episode_data.loc[(episode_data['dropoff_weekday']==1) &
                          (episode_data['last_dropoff']==1), '_DO_shift'] = episode_data['dropoff_time']\
                                    .apply(lambda x: _get_shift(x, weekday=True, pickup=False))


    ## determine shift for weekends
    episode_data.loc[(episode_data['pickup_weekday']==0) &
                          (episode_data['first_pickup']==1) , '_PU_shift'] = episode_data['pickup_time']\
                                    .apply(lambda x: _get_shift(x, weekday=False, pickup=True))
    episode_data.loc[(episode_data['dropoff_weekday']==0) &
                          (episode_data['last_dropoff']==1), '_DO_shift'] = episode_data['dropoff_time']\
                                    .apply(lambda x: _get_shift(x, weekday=False, pickup=False))

    ## fill shifts for other rows
    episode_data['_PU_shift'] = episode_data.groupby('hack_license')['_PU_shift'].ffill()
    episode_data['_DO_shift'] = episode_data.groupby('hack_license')['_DO_shift'].bfill()
    return episode_data

def select_shift(episode_data, selected_shift):
    ## keep only consistent shift
    episode_data = episode_data.loc[episode_data['_PU_shift'] == episode_data['_DO_shift']]
    episode_data = episode_data.loc[episode_data['_PU_shift'] == selected_shift]
    episode_data = episode_data[['sub_index', 'hack_license', 'pickup_datetime', 'dropoff_datetime', 
                                'pickup_taxizone_id', 'dropoff_taxizone_id', 
                                'total_amount', 'first_pickup', 'last_dropoff']].copy()
    return episode_data

def get_complete_shifts(episode_data):
    episode_data['mask_start'] = np.where(episode_data["first_pickup"] == 1, 1, np.nan)
    episode_data['mask_start'] = episode_data.groupby('hack_license')['mask_start'].ffill()
    episode_data['mask_end'] = np.where(episode_data["last_dropoff"] == 1, 1, np.nan)
    episode_data['mask_end'] = episode_data.groupby('hack_license')['mask_end'].bfill()
    episode_data['mask'] = np.where((episode_data["mask_start"] == 1) & (episode_data["mask_end"] == 1), 1, np.nan)
    episode_data.dropna(subset=['mask'], inplace=True)
    episode_data.drop(columns=['mask_start', 'mask_end', 'mask'], inplace=True)
    return episode_data
    
def assign_ep_id(episode_data):
    episode_data['episode'] = np.where(episode_data["first_pickup"] == 1, episode_data["first_pickup"].index, np.nan)
    episode_data['episode'] = episode_data.groupby('hack_license')['episode'].ffill()
    return episode_data

def convert_to_time_index(episode_data, interval_index_table, delta_t):
    
    round_by = '{}min'.format(delta_t)
    episode_data['pickup_datetime_interval'] = episode_data['pickup_datetime'].dt.round(round_by).dt.time
    episode_data['dropoff_datetime_interval'] = episode_data['dropoff_datetime'].dt.round(round_by).dt.time
    
    ## convert DO interval to time index
    current_conversion = dict(zip(interval_index_table.interval, interval_index_table[f'time_index_{delta_t}m']))
    episode_data['pickup_datetime_index'] = [current_conversion[t] for t in episode_data['pickup_datetime_interval']]
    episode_data['dropoff_datetime_index'] = [current_conversion[t] for t in episode_data['dropoff_datetime_interval']]
    return episode_data

def _merge_immediate_trips(trip_df):

    trip_df.sort_values(['episode', 'sub_index'], inplace=True)
    
    trip_df['pickup_datetime_index_next'] = trip_df.groupby('episode')['pickup_datetime_index'].shift(-1)
    trip_df['pickup_taxizone_id_next'] = trip_df.groupby('episode')['pickup_taxizone_id'].shift(-1)

    trip_df['dropoff_datetime_index_prev'] = trip_df.groupby('episode')['dropoff_datetime_index'].shift(1)
    trip_df['dropoff_taxizone_id_prev'] = trip_df.groupby('episode')['dropoff_taxizone_id'].shift(1)

    
    trip_df['mask_bfill'] = np.where((trip_df['dropoff_taxizone_id']==trip_df['pickup_taxizone_id_next']) &
                                     (trip_df['dropoff_datetime_index']==trip_df['pickup_datetime_index_next']), 1, np.nan)
    trip_df['mask_ffill'] = np.where((trip_df['pickup_taxizone_id']==trip_df['dropoff_taxizone_id_prev']) &
                                     (trip_df['pickup_datetime_index']==trip_df['dropoff_datetime_index_prev']), 1, np.nan)

    ffill_cols = ['sub_index', 'pickup_taxizone_id', 'pickup_datetime_index']
    for col_names in ffill_cols:
        trip_df[col_names] = np.where(trip_df['mask_ffill'] == 1, np.nan, trip_df[col_names])
        trip_df[col_names] = trip_df.groupby('episode')[col_names].ffill()
    bfill_cols = ['dropoff_taxizone_id', 'dropoff_datetime_index']
    for col_names in bfill_cols:
        trip_df[col_names] = np.where(trip_df['mask_bfill'] == 1, np.nan, trip_df[col_names])
        trip_df[col_names] = trip_df.groupby('episode')[col_names].bfill()
        
    trip_df = trip_df[['episode', 'sub_index', 'pickup_taxizone_id', 'dropoff_taxizone_id', 'pickup_datetime_index', 
                   'dropoff_datetime_index', 'total_amount']]
    trip_df = trip_df.groupby(['episode', 'sub_index', 'pickup_taxizone_id', 'dropoff_taxizone_id', 
                               'pickup_datetime_index', 'dropoff_datetime_index'], as_index=False).agg('sum')
    return trip_df
    
    
def get_trip_df(episode_data):
    trip_df = episode_data[[ 'episode', 'sub_index', 'pickup_taxizone_id', 'dropoff_taxizone_id', 'pickup_datetime_index',
                        'dropoff_datetime_index', 'total_amount']].copy()
    
    trip_df = _merge_immediate_trips(trip_df)

    trip_df.sort_values(['episode', 'sub_index'], inplace=True)
    trip_df.rename(columns={'pickup_taxizone_id': 'loc',
                            'pickup_datetime_index': 'time',
                            'dropoff_taxizone_id': 'loc_next',
                            'dropoff_datetime_index': 'time_next',
                            'total_amount': 'reward'}, inplace=True)
    trip_df['type'] = 0
    return trip_df

def get_repo_df(trip_df):
    reposition_df = trip_df.copy()
    reposition_df['loc_next_next'] = reposition_df.groupby('episode')['loc'].shift(-1) ## will be next location
    reposition_df['time_next_next'] = reposition_df.groupby('episode')['time'].shift(-1) ## will be next time
    reposition_df.drop(columns=['loc', 'time'], inplace=True)
    reposition_df.rename(columns={'loc_next': 'loc',
                            'time_next': 'time',
                            'loc_next_next': 'loc_next',
                            'time_next_next': 'time_next'}, inplace=True)
    reposition_df['reward'] = 0
    ## 1 = cruise, 2 = repo
    reposition_df['type'] = np.where(reposition_df['loc'] == reposition_df['loc_next'], 1, 2)
    return reposition_df

## insert rows to dataframe if cruising takes more than 1 interval
def _expand_by_time_step_helper(selected_df):
    df = selected_df.copy()
    assert df['time'].unique().shape[0] == 1
    assert df['time_next'].unique().shape[0] == 1
    
    lower_time = int(df['time'].unique())
    upper_time = int(df['time_next'].unique())
    
    max_time_index = 60/delta_t*24 
    
    t = df.shape[0]
    ## go over midnight
    time_list_len = t + 1
    if upper_time < lower_time:
        upper_time = upper_time + max_time_index
        new_time_list = np.linspace(lower_time, upper_time, time_list_len, endpoint=True)
        new_time_list = np.where(new_time_list < max_time_index, new_time_list, new_time_list%max_time_index)
        new_time_list = new_time_list.astype(int)
#         new_time_list = [int(i) for i in new_time_list if i < max_time_index else int(i%max_time_index)]
    else:
        new_time_list = np.linspace(lower_time, upper_time, time_list_len, endpoint=True).astype(int)
        
    df['time'] = new_time_list[:t]
    df['time_next'] = new_time_list[-t:]
    df['expanded_index'] = np.linspace(0, upper_time-lower_time, t, endpoint=False).astype(int) ##for sorting within expanded rows
    
    
#     if upper_time == max_time_index - 1:
#         adjusted_time = [i+max_time_index for i in df['time'] if i < 20 else i]
#         lower_time = int(min(adjusted_time))
#         upper_time = int(max(adjusted_time))
#         current_new_time = np.arange(lower_time, upper_time, 1) / (max_time_index)
#         next_new_time = np.arange(lower_time+1, upper_time+1, 1) / (max_time_index)
#     else:
#         current_new_time = np.arange(lower_time, upper_time, 1)
#         next_new_time = np.arange(lower_time+1, upper_time+1, 1)
        
#     df['time'] = current_new_time
#     df['time_next'] = next_new_time
#     df['expanded_index'] = range(upper_time-lower_time) ##for sorting within expanded rows
    return df

"""
Return a list of time index. It accounts for the case when the index goes over midnight.
The length is t+1 for t is #rows of df. Current time takes the first t indices and the
next_time takes the last t indices from the list. 
"""
# def _get_time_list(original_time_list):
#     max_time_index = 60/delta_t*24
#     start_time = original_time_list[0]
 
#     if (max_time_index-1) in original_time_list:
#         new_time_list = [i+max_time_index for i in original_time_list if i < 20 else i]
#          end_time = original_time_list[-1]
#     start_time = 
    
## extract rows that will be expanded from the rest. Return the expanded part and a list of which rows in repo_df to be kept. 
def expand_by_time_step(reposition_df):
    expand_list = []
    repeat_size = []
    kept_list = []

    # for cruise rows that are longer than 1 time interval
    for row in reposition_df.itertuples(index=False):
        if (getattr(row, 'type') == 1) & (getattr(row, 'time_next') - getattr(row, 'time') > 1):
            expand_list.append(tuple(row))
            repeat_size.append(int(getattr(row, 'time_next') - getattr(row, 'time')))
            kept_list.append(np.nan)
        else:
            kept_list.append(1)

    # if no rows to be expand, return None
    if expand_list == []:
        return None, None
    
    expand_df_list = [pd.DataFrame(np.repeat(np.array([np.array(current_df)]), 
                                             np.array(current_repeatsize), axis=0),
                                   columns=reposition_df.columns) \
                      for current_df, current_repeatsize in zip(expand_list, repeat_size)]
    expand_df_list_1 = [_expand_by_time_step_helper(df) for df in expand_df_list]
    expand_df = pd.concat(expand_df_list_1, axis=0)  
    expand_df = expand_df.astype({'episode': 'float',
                                  'sub_index': 'float',
                                  'loc': 'float',
                                  'time': 'float',
                                  'loc_next': 'float',
                                  'time_next': 'float',
                                  'type': 'float'})
    expand_df = expand_df.astype({'episode': 'Int64',
                                  'sub_index': 'Int64',
                                  'loc': 'Int64',
                                  'time': 'Int64',
                                  'loc_next': 'Int64',
                                  'time_next': 'Int64',
                                  'type': 'Int64'})        
    return expand_df, kept_list

def _expand_by_zones_helper(df, path):
    # location
    t = len(path)-1
    df['loc'] = path[:t]
    df['loc_next'] = path[-t:]
    
    # time
    df = _expand_by_time_step_helper(df)
#     lower_time = min(df['time'])
#     upper_time = max(df['time_next'])
#     time_list = np.linspace(lower_time, upper_time, t+1, endpoint=True).astype(int)
#     df['time'] = time_list[:t]
#     df['time_next'] = time_list[-t:]
#     df['expanded_index'] = np.linspace(lower_time, upper_time, t, endpoint=True) ##for sorting within expanded rows
    return df

def expand_by_zones(repo_df):
    
    expand_list = []
    repeat_size = []
    kept_list = []
    path_list = []
    remove_log = [] ## keep track of episode to be removed
    # for cruise rows that are longer than 1 time interval
    for row in repo_df.itertuples(index=False):
        origin = getattr(row, 'loc')
        destination = getattr(row, 'loc_next')
        if np.isnan(destination):
            kept_list.append(1)
            continue

        try:
            shortest_path = est.shortest_path(origin,destination)
        except:
            ## zone 1 is out of reach. An episode with zone 1 will be removed.
            remove_log.append(getattr(row, 'episode'))
            kept_list.append(1)
            continue

        if len(shortest_path) > 2:
            expand_list.append(tuple(row))
            repeat_size.append(int(len(shortest_path)-1))
            kept_list.append(np.nan) ## to be removed from repo_df
            path_list.append(shortest_path)
        else:
            kept_list.append(1)
            
    ## if no row need expanding, return None and keep al rows of repo_df
    if expand_list == []:
        return None, None, None
    
    expand_df_list = [pd.DataFrame(np.repeat(np.array([np.array(current_df)]), 
                                             np.array(current_repeatsize), axis=0),
                                   columns=repo_df.columns) \
                      for current_df, current_repeatsize in zip(expand_list, repeat_size)]
    expand_df_list_1 = [_expand_by_zones_helper(df, path) for df, path in zip(expand_df_list, path_list)]
    expand_df = pd.concat(expand_df_list_1, axis=0)  
    expand_df = expand_df.astype({'episode': 'float',
                                  'sub_index': 'float',
                                  'loc': 'float',
                                  'time': 'float',
                                  'loc_next': 'float',
                                  'time_next': 'float',
                                  'type': 'float'})
    expand_df = expand_df.astype({'episode': 'Int64',
                                  'sub_index': 'Int64',
                                  'loc': 'Int64',
                                  'time': 'Int64',
                                  'loc_next': 'Int64',
                                  'time_next': 'Int64',
                                  'type': 'Int64'})
    return expand_df, kept_list, remove_log

# expand same zone cruising at every time step 
def combine_expand_and_repo_df(reposition_df, expand_df, kept_list):
    
    ## if no expand df, return original reposition_df
    if expand_df is None:
        reposition_df['expanded_index'] = 0
        return reposition_df
    
    ## drop rows in repo to prevent duplicates when concat to cruise
    reposition_df['kept'] = kept_list
    reposition_df.dropna(subset=['kept'], inplace=True)
    reposition_df.drop(columns=['kept'], inplace=True)
    
    # Concatenate back expand_df and repo_df
    reposition_df = pd.concat([reposition_df, expand_df], sort=True).sort_values(['episode', 'sub_index'])
    reposition_df['expanded_index'] = reposition_df['expanded_index'].fillna(0)
        
    return reposition_df

def to_SARSA_format(trip_df, reposition_df, remove_log=None):
    sarsa_df = pd.concat([trip_df, reposition_df], sort=True)
    sarsa_df['expanded_index'] = sarsa_df['expanded_index'].fillna(0)
    sarsa_df.sort_values(['episode', 'sub_index', 'type', 'expanded_index'], inplace=True)

    sarsa_df['action'] = np.where(sarsa_df['type'] == 0, np.nan,  sarsa_df['loc_next'])
    sarsa_df['action'] = sarsa_df.groupby(['episode'])['action'].ffill()

    sarsa_df.dropna(subset=['action'], inplace=True)
    sarsa_df.drop(columns=['type'], inplace=True)
    sarsa_df = sarsa_df.astype('float64')
    # sarsa_df = sarsa_df.groupby(['episode', 'sub_index', 'expanded_index', 'loc', 
    #                              'loc_next', 'time', 'time_next', 'action'], as_index=False).agg('sum')
    sarsa_df.sort_values(['episode', 'sub_index', 'expanded_index'], inplace=True)
    sarsa_df = sarsa_df.astype({'episode': 'Int64',
                                  'loc': 'Int64',
                                  'time': 'Int64',
                                  'loc_next': 'Int64',
                                  'action': 'Int64',
                                  'time_next': 'Int64'})
    # # sarsa_df['action'] = sarsa_df['loc_next']
    sarsa_df['action_next'] = sarsa_df.groupby('episode')['action'].shift(-1)
    sarsa_df['state'] = [(loc, time) for loc, time in zip(sarsa_df['loc'], sarsa_df['time'])]
    sarsa_df['state_next'] = [(loc, time) for loc, time in zip(sarsa_df['loc_next'], sarsa_df['time_next'])]
    if (remove_log is not None) & (remove_log != []):
        sarsa_df = sarsa_df.loc[~sarsa_df['episode'].isin(remove_log)]
    sarsa_df.reset_index(inplace=True)
    sarsa_df = sarsa_df[['episode', 'state', 'action', 'reward', 'state_next', 'action_next']]
    return sarsa_df

def generate_SARSA_samples(cleaned_trip_df, shift, interval_index_table, delta_t, save_path, version):
    
    sample_df = preprocess_df(cleaned_trip_df)
    sample_df = assign_shift(sample_df)
    sample_df = select_shift(sample_df, shift)
    sample_df = get_complete_shifts(sample_df)
    sample_df = assign_ep_id(sample_df)
    sample_df = convert_to_time_index(sample_df, interval_index_table, delta_t)
    trip_df = get_trip_df(sample_df)
    repo_df = get_repo_df(trip_df)
    
    if version == 3:
        ## expand zones
        expand_df, kept_list, remove_log = expand_by_zones(repo_df)
        repo_df = combine_expand_and_repo_df(repo_df, expand_df, kept_list)
    elif version == 2:
        remove_log = None
    else:
        raise ValueError('version must be 2 or 3')

#     ## expand time
    expand_df, kept_list = expand_by_time_step(repo_df)
    repo_df_1 = combine_expand_and_repo_df(repo_df, expand_df, kept_list)
    sarsa_df = to_SARSA_format(trip_df, repo_df_1)

    test_dataset(sarsa_df)
    with open(save_path, 'wb') as handle:
        pickle.dump(sarsa_df, handle)
    print('    saved at', save_path)

def test_dataset(df):
    
    temp = df.copy()
    temp['cur_zone'] = [i[0] for i in temp['state']]
    temp['next_zone'] = [i[0] for i in temp['state_next']]

    temp['cur_time'] = [i[1] for i in temp['state']]
    temp['next_time'] = [i[1] for i in temp['state_next']]
    
    temp['state'] = temp['state'].astype('str')
    temp['state_next'] = temp['state_next'].astype('str')
    
    temp = temp.astype({'episode': 'Int64',
                          'action': 'Int64',
                          'action_next': 'Int64',
                          'cur_zone': 'Int64',
                          'next_zone': 'Int64',
                          'cur_time': 'Int64',
                          'next_time': 'Int64'})
    

    
    test = temp.loc[temp['state'].isnull()]
    assert test.shape[0] == 0, 'current states cannot be NaN'
    print('.')
    
    test = temp.loc[temp['state_next'].isnull()]
    assert test.shape[0] == 0, 'current states cannot be NaN'
    print('.')
    
    test = temp.loc[(temp['next_time'] - temp['cur_time'] > 1) & 
                         (temp['reward'] == 0) & 
                         (temp['cur_zone'] == temp['next_zone'])]
    assert test.shape[0] == 0, 'same zone cruising over more than 1 time interval not expanded'
    print('.')
    
    test = temp.loc[(temp['state'] == temp['state_next']) & (temp['reward'] == 0)]
    assert test.shape[0] == 0, 'immediate trips are not merged'
    print('.')
    
    test = temp.loc[(temp['reward'] == 0) & (temp['action'] != temp['next_zone']) & (~temp['action_next'].isnull())]
    assert test.shape[0] == 0, 'next zone must equal to action when reposition'
    print('.')
    
    max_time_index = 60/delta_t*24
    test = temp.loc[(temp['cur_time'] >= max_time_index) | (temp['next_time'] >= max_time_index)]
    assert test.shape[0] == 0, 'time index exceeds possible value'
    print('.')
    

    
    
    
    
    

# ++++++++++++++++++++++++
## must set the following. Possible overwriting if input uncarefully
delta_t = 15
interval_index_table_file_path = 'data/interval_index_table_0.csv'
cleaned_trip_df_file_path = 'data/trip_cleaned.csv'
shift = 'B'
CHUNK_SIZE = 1000000
VERSION = 3
dir_name = f'data/sh{shift}_v{VERSION}'
save_path = dir_name + '/sarsa_{}.pickle'
# ++++++++++++++++++++++++

interval_index_table = pd.read_csv(interval_index_table_file_path)
interval_index_table['interval'] = pd.to_datetime(interval_index_table['interval']).dt.time
print('interval_index_table read')

est = Estimator(dir_path='data/', delta_t = delta_t)
print('Estimator created for shortest paths')

if not os.path.isdir(dir_name):
    print(f'{dir_name} does not exist. Create dir')
    os.makedirs(dir_name)

cnter = 0
result_list = []
for chunk in tqdm(pd.read_csv(cleaned_trip_df_file_path, chunksize=CHUNK_SIZE), desc='Chunk'):
    cnter += 1
#     print('\nChunk ', cnter)
    generate_SARSA_samples(chunk, shift, interval_index_table, delta_t,
                           save_path.format(cnter), version=3)
    
    
    
