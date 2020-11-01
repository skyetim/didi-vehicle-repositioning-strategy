import pandas as pd
import numpy as np
from random import choices


# TODO: ty & xx
def trip_fare(src, dst, t):
    """compute trip fare between two taxi zones at time t.

    Parameters
    ----------
    src : int or array-like
        source taxi zone id
    dst : int or array-like
        destination taxi zone id
    t : int or array-like
        environment time t

    Returns
    -------
    float or array-like
        fare between two taxi zones at time t.

    Notes
    -----
    Use average.
    """

    match = pd.read_csv('../data/fare_amount_scr_dst_t.csv')
    if (not isinstance(src, (list, tuple, np.ndarray))):
        src = [src]
        dst = [dst]
        t = [t]
    m = match.loc[(match['pickup_taxizone_id'].isin(src)) & (match['dropoff_taxizone_id'].isin(dst))]
    m = m.loc[(m['pickup_datetime_index'].isin(t)), 'mean']

    if m.shape[0] == 0:
        return -1
    else:
        return m.values


# TODO: ty & xx
def trip_distance(src, dst):
    """compute trip distance between two taxi zones.

    Parameters
    ----------
    src : int or array-like
        source taxi zone id
    dst : int or array-like
        destination taxi zone id

    Returns
    -------
    float or array-like
        distance between two taxi zones.

    Notes
    -----
    Use average.
    """
    match = pd.read_csv('../data/trip_distance_scr_dst.csv')
    if (not isinstance(src, (list, tuple, np.ndarray))):
        src = [src]
        dst = [dst]
    m = match.loc[(match['pickup_taxizone_id'].isin(src)) & (match['dropoff_taxizone_id'].isin(dst)), 'mean']

    if m.shape[0] == 0:
        return -1
    else:
        return m.values


# TODO: ty & xx
def trip_time(src, dst, t):
    """compute trip time between two taxi zones starting at time t.

    Parameters
    ----------
    src : int or array-like
        source taxi zone id
    dst : int or array-like
        destination taxi zone id
    t : int or array-like
        environment time t

    Returns
    -------
    float or array-like
        time between two taxi zones starting at time t.

    Notes
    -----
    Use average.
    """
    match = pd.read_csv('../data/trip_time_scr_dst_t.csv')
    if (not isinstance(src, (list, tuple, np.ndarray))):
        src = [src]
        dst = [dst]
        t = [t]
    m = match.loc[(match['pickup_taxizone_id'].isin(src)) & (match['dropoff_taxizone_id'].isin(dst))]
    m = m.loc[(m['pickup_datetime_index'].isin(t)), 'mean']

    if m.shape[0] == 0:
        return -1
    else:
        return m.values


# TODO: yy & bo
def cruise_time(zone, t):
    """compute cruise time in `zone` at time t before the next order comes in.

    Parameters
    ----------
    zone : int or array-like
        taxi zone id
    t : int or array-like
        environment time t

    Returns
    -------
    int or array-like
        time between two taxi zones starting at time t.

    Notes
    -----
    Use average.
    """
    lookup_table = pd.read_csv('../data/cruise_time_15m.csv')  # only implemented for 15 mins

    q = lookup_table.loc[(lookup_table['dropoff_datetime_index'] == t) & (lookup_table['taxizone_id'] == zone),
                         'med_cruise_time']
    if q.shape[0] == 0:
        return 1000
    else:
        return q.values


# placeholder
def cruise_distance(zone, t):
    """compute cruise distance in `zone` at time t before the next order comes in.

    Parameters
    ----------
    zone : int or array-like
        taxi zone id
    t : int or array-like
        environment time t

    Returns
    -------
    int or array-like
        distance between two taxi zones starting at time t.

    Notes
    -----
    Use average.
    """
    raise NotImplementedError()


# ty & xx
def generate_request(zone, t):
    """generate a random request in `zone` at time t

    Parameters
    ----------
    zone : int
        taxi zone id
    t : int
        time

    Returns
    -------
    int
        dst taxi zone
    """
    mc_mtx_2d = np.loadtxt("../data/mc_mtx.txt") 
    mc_mtx = mc_mtx_2d.reshape( 
        mc_mtx_2d.shape[0], mc_mtx_2d.shape[1] // 263, 263) #convert back to 3d array
    
    population = np.arange(1, 264)
    weights = mc_mtx[t,zone-1,:] #index starts from 0, zone starts from 1
    
    if (weights == np.zeros((1, 263))).all():
        uniform = np.full((263,1), 1/263)
        dst = choices(population, uniform)[0]
    else:
        dst = choices(population, weights)[0]

    return dst
