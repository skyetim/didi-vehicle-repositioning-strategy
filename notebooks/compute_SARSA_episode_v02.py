#!/usr/bin/env python
# coding: utf-8

# # Generate dataset for SARSA algorithm

# Two things to set
# 1. selected_shift: either A or B
# 2. delta_t: data must available in conversion table
# 
# New:
# 1. no immediate transition is collapsed

# In[90]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime


# In[91]:


cleaned_trip_df = pd.read_csv('../data/trip_cleaned.csv')


# In[92]:


pd.set_option('max_row', 10)


# ## Convert time to datetime

# In[93]:


episode_data = cleaned_trip_df.copy()
episode_data.sort_values(['hack_license', 'pickup_datetime'], inplace=True)
episode_data['pickup_datetime'] = pd.to_datetime(episode_data['pickup_datetime'])
episode_data['dropoff_datetime'] = pd.to_datetime(episode_data['dropoff_datetime'])
episode_data['pickup_time'] = pd.to_datetime(episode_data['pickup_time']).dt.time
episode_data['dropoff_time'] = pd.to_datetime(episode_data['dropoff_time']).dt.time
episode_data['sub_index'] = episode_data.index ##preserve order for the rest of the process


# ## Get selected shift
# * start in the morning
# * end in the afternoon

# In[94]:


selected_shift = 'A'


# In[95]:


def get_shift(time, weekday=True, pickup=True):
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


# ### Label trips with either shift A or B

# In[96]:


## determine shift for weekdays
episode_data.loc[(episode_data['pickup_weekday']==1) &
                      (episode_data['first_pickup']==1) , '_PU_shift'] = episode_data['pickup_time']\
                                .apply(lambda x: get_shift(x, weekday=True, pickup=True))
episode_data.loc[(episode_data['dropoff_weekday']==1) &
                      (episode_data['last_dropoff']==1), '_DO_shift'] = episode_data['dropoff_time']\
                                .apply(lambda x: get_shift(x, weekday=True, pickup=False))

## determine shift for weekends
episode_data.loc[(episode_data['pickup_weekday']==0) &
                      (episode_data['first_pickup']==1) , '_PU_shift'] = episode_data['pickup_time']\
                                .apply(lambda x: get_shift(x, weekday=False, pickup=True))
episode_data.loc[(episode_data['dropoff_weekday']==0) &
                      (episode_data['last_dropoff']==1), '_DO_shift'] = episode_data['dropoff_time']\
                                .apply(lambda x: get_shift(x, weekday=False, pickup=False))

## fill shifts for other rows
episode_data['_PU_shift'] = episode_data.groupby('hack_license')['_PU_shift'].ffill()
episode_data['_DO_shift'] = episode_data.groupby('hack_license')['_DO_shift'].bfill()


# In[97]:


print('# consistent shift trips: ', episode_data.loc[episode_data['_PU_shift'] == episode_data['_DO_shift']].shape)
print('total trips: ', episode_data.shape)


# ### Pick 1 shift and keep only trips with consistent start and end time

# In[98]:


## keep only consistent shift
episode_data = episode_data.loc[episode_data['_PU_shift'] == episode_data['_DO_shift']]
episode_data = episode_data.loc[episode_data['_PU_shift'] == selected_shift]
episode_data = episode_data[['sub_index', 'hack_license', 'pickup_datetime', 'dropoff_datetime', 
                            'pickup_taxizone_id', 'dropoff_taxizone_id', 
                            'total_amount', 'first_pickup', 'last_dropoff']].copy()


# ## Keep only data points of full shift 
# A complete shift must have both the first pickup and the last drop off

# In[99]:


episode_data['mask_start'] = np.where(episode_data["first_pickup"] == 1, 1, np.nan)
episode_data['mask_start'] = episode_data.groupby('hack_license')['mask_start'].ffill()
episode_data['mask_end'] = np.where(episode_data["last_dropoff"] == 1, 1, np.nan)
episode_data['mask_end'] = episode_data.groupby('hack_license')['mask_end'].bfill()
episode_data['mask'] = np.where((episode_data["mask_start"] == 1) & (episode_data["mask_end"] == 1), 1, np.nan)
episode_data.dropna(subset=['mask'], inplace=True)
episode_data.drop(columns=['mask_start', 'mask_end', 'mask'], inplace=True)


# ## Assign unique episode id
# One shift of a driver is assgined with a unique id

# In[100]:


episode_data['episode'] = np.where(episode_data["first_pickup"] == 1, episode_data["first_pickup"].index, np.nan)
episode_data['episode'] = episode_data.groupby('hack_license')['episode'].ffill()


# ## Convert all time columns to time index
# depending on `delta_t`. First round and convert using `interval_index_table`.

# In[101]:


interval_index_table = pd.read_csv('../data/interval_index_table_0.csv')
interval_index_table['interval'] = pd.to_datetime(interval_index_table['interval']).dt.time


# In[102]:


delta_t = 15
round_by = '{}min'.format(delta_t)
episode_data['pickup_datetime_interval'] = episode_data['pickup_datetime'].dt.round(round_by).dt.time
episode_data['dropoff_datetime_interval'] = episode_data['dropoff_datetime'].dt.round(round_by).dt.time

## convert DO interval to time index
current_conversion = dict(zip(interval_index_table.interval, interval_index_table[f'time_index_{delta_t}m']))
episode_data['pickup_datetime_index'] = [current_conversion[t] for t in episode_data['pickup_datetime_interval']]
episode_data['dropoff_datetime_index'] = [current_conversion[t] for t in episode_data['dropoff_datetime_interval']]


# ## label as delivery

# In[106]:


trip_df = episode_data[[ 'episode', 'sub_index', 'pickup_taxizone_id', 'dropoff_taxizone_id', 'pickup_datetime_index',
                        'dropoff_datetime_index', 'total_amount']].copy()

trip_df.sort_values(['episode', 'sub_index'], inplace=True)
trip_df.rename(columns={'pickup_taxizone_id': 'loc',
                        'pickup_datetime_index': 'time',
                        'dropoff_taxizone_id': 'loc_next',
                        'dropoff_datetime_index': 'time_next',
                        'total_amount': 'reward'}, inplace=True)
trip_df['type'] = 0


# ## Create reposition dataset
# 
# repostion from current DO loc to the next PU location

# In[107]:


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
reposition_df.head(10)


# ## Create cruise dataset 
# 
# * with expanded rows if cruising more than 1 time interval

# In[108]:


cruise_list = []
repeat_size = []
kept_list = []

# for cruise rows that are longer than 1 time interval
for row in reposition_df.itertuples(index=False):
    if (getattr(row, 'type') == 1) & (getattr(row, 'time_next') - getattr(row, 'time') > 1):
        cruise_list.append(tuple(row))
        repeat_size.append(getattr(row, 'time_next') - getattr(row, 'time'))
        kept_list.append(np.nan)
    else:
        kept_list.append(1)
reposition_df['kept'] = kept_list
reposition_df.dropna(subset=['kept'], inplace=True)
reposition_df.drop(columns=['kept'], inplace=True)

cruise_df = pd.DataFrame(np.repeat(np.array(cruise_list), repeat_size, axis=0))
cruise_df.columns = reposition_df.columns

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


# In[109]:


cruise_df.head(10)


# In[110]:


## insert rows to dataframe if cruising takes more than 1 interval
def expand_cruise(selected_df):
    df = selected_df.copy()
    lower_time = min(df['time'])
    upper_time = max(df['time_next'])
    df['time'] = range(lower_time, upper_time)
    df['time_next'] = range(lower_time+1, upper_time+1)
    df['expanded_index'] = range(upper_time-lower_time) ##for sorting within expanded rows
    return df


# ### Expand and combine cruise df to reposition df
# 
# * resort the df using episode-sub_index key

# In[111]:


cruise_df_2 = cruise_df.groupby(['episode', 'sub_index', 'loc', 'reward', 'type', 'loc_next']).apply(expand_cruise)
reposition_df = pd.concat([reposition_df, cruise_df_2], sort=True).sort_values(['episode', 'sub_index'])
reposition_df['expanded_index'] = reposition_df['expanded_index'].fillna(0)


# In[112]:


reposition_df.head(10)


# ## Combine reposition and trip dataset
# 
# ## Convert into SARSA format
# 
# * Two transitions: (228,68) -- 65 --> (65, 68) and (65, 68) -- 65 --> (13, 70) The first on is repo with reward 0 and the second is delivery with reward = trip fare

# In[121]:


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


# In[122]:


sarsa_df.head(10)


# In[94]:


import pickle
with open(f'../data/SARSA_eps_{delta_t}m_v02.pickle', 'wb') as handle:
    pickle.dump(sarsa_df, handle)

## For interpretable read. Suffering data loss.
sarsa_df.to_csv(f'../data/SARSA_eps_{delta_t}m_v02.csv', index=False)


# In[96]:


## load pickle
# with open('../data/historical_for_SARSA.pickle', 'rb') as handle:
#     episode_data = pickle.load(handle)

