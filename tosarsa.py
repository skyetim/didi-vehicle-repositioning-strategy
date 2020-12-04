import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle
from functools import reduce
from tqdm import tqdm


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

def get_trip_df(episode_data):
    trip_df = episode_data[[ 'episode', 'sub_index', 'pickup_taxizone_id', 'dropoff_taxizone_id', 'pickup_datetime_index',
                        'dropoff_datetime_index', 'total_amount']].copy()

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
    lower_time = min(df['time'])
    upper_time = max(df['time_next'])
    df['time'] = range(lower_time, upper_time)
    df['time_next'] = range(lower_time+1, upper_time+1)
    df['expanded_index'] = range(upper_time-lower_time) ##for sorting within expanded rows
    return df

## extract rows that will be expanded from the rest. Return the expanded part and a list of which rows in repo_df to be kept. 
def expand_by_time_step(reposition_df):
    cruise_list = []
    repeat_size = []
    kept_list = []
    repo_df = reposition_df.copy()

    # for cruise rows that are longer than 1 time interval
    for row in repo_df.itertuples(index=False):
        if (getattr(row, 'type') == 1) & (getattr(row, 'time_next') - getattr(row, 'time') > 1):
            cruise_list.append(tuple(row))
            repeat_size.append(getattr(row, 'time_next') - getattr(row, 'time'))
            kept_list.append(np.nan)
        else:
            kept_list.append(1)
#     repo_df['kept'] = kept_list
#     repo_df.dropna(subset=['kept'], inplace=True)
#     repo_df.drop(columns=['kept'], inplace=True)

    if cruise_list == []:  ## i.e. no cruise needs expanding 
        return None
    
    cruise_df = pd.DataFrame(np.repeat(np.array(cruise_list), repeat_size, axis=0))
    cruise_df.columns = repo_df.columns

    ## Cast type back to int
    cruise_df = cruise_df.astype({'episode': 'float',
                                  'sub_index': 'float',
                                  'loc': 'float',
                                  'time': 'float',
                                  'loc_next': 'float',
                                  'time_next': 'float',
                                  'type': 'float'})
    cruise_df = cruise_df.astype({'episode': 'Int64',
                                  'sub_index': 'Int64',
                                  'loc': 'Int64',
                                  'time': 'Int64',
                                  'loc_next': 'Int64',
                                  'time_next': 'Int64',
                                  'type': 'Int64'})
    
    # Expand cruise_df at every time step
    if cruise_df is not None:
        cruise_df = cruise_df.groupby(['episode', 'sub_index', 'loc', 'reward', 'type', 'loc_next']).apply(_expand_by_time_step_helper)
        
    return cruise_df, kept_list

# expand same zone cruising at every time step 
def process_repo_df(reposition_df):
    # Separate into 1. transitions of cruise within the zone (cruise_df) 
    # and 2. transitions of repositioning to other zone (repo_df)
    cruise_df, kept_list = expand_by_time_step(reposition_df)
    
    ## drop rows in repo to prevent duplicates when concat to cruise
    repo_df = reposition_df.copy()
    repo_df['kept'] = kept_list
    repo_df.dropna(subset=['kept'], inplace=True)
    repo_df.drop(columns=['kept'], inplace=True)
    
    # Concatenate back cruise_df and repo_df
    if cruise_df is not None:
        repo_df = pd.concat([repo_df, cruise_df], sort=True).sort_values(['episode', 'sub_index'])
        repo_df['expanded_index'] = repo_df['expanded_index'].fillna(0)
    else:
        repo_df['expanded_index'] = 0
        
    return repo_df
      
# def _combine_repo_and_cruise_df(reposition_df, cruise_df):
#     if cruise_df is not None:
#         cruise_df_2 = cruise_df.groupby(['episode', 'sub_index', 'loc', 'reward', 'type', 'loc_next']).apply(_expand_cruise)
#         reposition_df = pd.concat([reposition_df, cruise_df_2], sort=True).sort_values(['episode', 'sub_index'])
#         reposition_df['expanded_index'] = reposition_df['expanded_index'].fillna(0)
#     else:
#         reposition_df['expanded_index'] = 0

#     return reposition_df

def to_SARSA_format(trip_df, reposition_df):
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
    sarsa_df = sarsa_df[['episode', 'state', 'action', 'reward', 'state_next', 'action_next']]
    return sarsa_df

def save_sarsa(cleaned_trip_df, shift, interval_index_table, delta_t, save_path):
    
#     output = preprocess_df(cleaned_trip_df)
#     output = assign_shift(output)
#     output = select_shift(output, shift)
#     output = get_complete_shifts(output)
#     output = assign_ep_id(output)
#     output = convert_to_time_index(output, interval_index_table, delta_t)
#     trip_df = get_trip_df(output)
#     repo_df = get_repo_df(trip_df)
#     cruise_df = get_cruise_df(repo_df)
#     repo_df = combine_repo_and_cruise_df(repo_df, cruise_df)
#     sarsa_df = to_SARSA_format(trip_df, repo_df)
    
    ## expand only bt time step
    output = tosarsa.preprocess_df(cleaned_trip_df)
    output = tosarsa.assign_shift(output)
    output = tosarsa.select_shift(output, shift)
    output = tosarsa.get_complete_shifts(output)
    output = tosarsa.assign_ep_id(output)
    output = tosarsa.convert_to_time_index(output, interval_index_table, delta_t)
    trip_df = tosarsa.get_trip_df(output)
    repo_df = tosarsa.get_repo_df(trip_df)
    repo_df = tosarsa.process_repo_df(repo_df)
    sarsa_df = tosarsa.to_SARSA_format(trip_df, repo_df)
    
    with open(save_path, 'wb') as handle:
        pickle.dump(sarsa_df, handle)
    print('    saved at', save_path)
    
# def process(cleaned_trip_df, shift, interval_index_table, delta_t):
#     print('.')
#     output = convert_to_datetime(cleaned_trip_df)
#     output = assign_shift(output)
#     output = select_shift(output, shift)
#     output = get_full_shift_trips(output)
#     output = assign_unique_ep_id(output)
#     output = convert_to_time_index(output, interval_index_table, delta_t)
#     trip_df = get_trip_df(output)
#     repo_df = get_repo_df(trip_df)
#     cruise_df = get_cruise_df(repo_df)
#     repo_df = combine_repo_and_cruise_df(repo_df, cruise_df)
#     sarsa_df = to_SARSA_format(trip_df, repo_df)
#     return sarsa_df

def concat(prev_result, current_df):
    return pd.concat([prev_result, pd.DataFrame(current_df)], axis=0)
    
## Processing

# delta_t = 15
# interval_index_table_file_path = 'data/interval_index_table_0.csv'
# interval_index_table = pd.read_csv(interval_index_table_file_path)
# interval_index_table['interval'] = pd.to_datetime(interval_index_table['interval']).dt.time
# print('interval_index_table read')

# cleaned_trip_df_file_path = 'data/trip_cleaned.csv'
# shift = 'B'
# CHUNK_SIZE = 1000000
# save_path = 'data/SARSA_sample_sh{}_chunk{}.pickle'
# cnter = 0
# result_list = []
# for chunk in tqdm(pd.read_csv(cleaned_trip_df_file_path, chunksize=CHUNK_SIZE), desc='Chunk'):
#     cnter += 1
# #     print('\nChunk ', cnter)
#     save_sarsa(chunk, shift, interval_index_table, delta_t, save_path.format(shift, cnter))
    
    
    
    
    
    
# MapReduce structure:
# chunks = pd.read_csv(cleaned_trip_df_file_path, chunksize=1000)
# print('mspping...')
# processed_chunks = map(lambda c: process(c, shift=shift, interval_index_table=interval_index_table, 
#                                          delta_t=delta_t), chunks)
# print('reducing...')
# result = reduce(concat, processed_chunks)

# result = pd.concat(result_list, axis=0)
# print(result.shape)
# with open(save_path, 'wb') as handle:
#     pickle.dump(result, handle)
# print('    saved at', save_path)