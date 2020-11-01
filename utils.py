# TODO: yy & bo
def timestamp_to_env_time(ts, delta_t=15, t0=0):
    """transform timestamp(s) to Environment time

    Parameters
    ----------
    ts : int or array-like
        timestamp as in raw dataset
    delta_t : int
        time interval of `delta_t` minutes
    t0: int
        when is the first environment time in real time, `t0` minutes after midnight

    Returns
    -------
    int or array-like
        transformed environment time
    """

    if (t0<0) | (t0>23):
        raise ValueError('t must be in between 0 and 23 inclusively')
        
    from datetime import datetime
    import pandas as pd
    import numpy as np
    ts = pd.Series(pd.to_datetime(ts))
    
    ## convert string to pd.datetime
    index_conversion_table = pd.read_csv('../data/interval_index_table_0.csv')
    index_conversion_table['interval'] = pd.to_datetime(index_conversion_table['interval']).dt.time
    
    if delta_t==15:
        rounded_q_time = ts.dt.round('15min').dt.time.values
        env_time = np.array([index_conversion_table.loc[index_conversion_table['interval'] == q]['index_15m'].values[0] \
                         for q in rounded_q_time])
    elif delta_t==60:
        rounded_q_time = pd.Series(pd.to_datetime(ts)).dt.round('60min').dt.time.values
        env_time = np.array([index_conversion_table.loc[index_conversion_table['interval'] == q]['index_60m'].values[0] \
                         for q in rounded_q_time])
    else:
        raise NotImplementedError('delta_t other 15 mins and 60 mins is not yet implemented.')
    
    if t0 != 0:
        env_time = np.array([i+24 if i<0 else i for i in (env_time - t0)])
    return env_time

# TODO: ty & xx
def is_adjacent(src, dst):
    """is two taxi zones adjacent.

    Parameters
    ----------
    src : int
        taxi zone 1
    dst : int
        taxi zone 2

    Returns
    -------
    bool
    """
    #raise NotImplementedError()
    
    ad = pd.read_csv('../data/adjacent_zone.csv')
    if dst<src:
        a = dst
        b = src
        r = ad.loc[(ad.zone1 == a)&(ad.zone2 == b)]
    else:
        r = ad.loc[(ad.zone1 == src)&(ad.zone2 == dst)]
    
    return (len(r)>0)
