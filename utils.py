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
    import pandas as pd

    index_conversion_table = pd.read_csv('data/interval_index_table_0.csv')
    if t=15:
        rounded_q_time = pd.Series(pd.to_datetime(ts)).dt.round('15min').dt.time.values
        return np.array([index_conversion_table.loc[index_conversion_table['interval'] == q]['index_15m'][0]
                        for q in rounded_q_time]) 
    elif t=60:
        rounded_q_time = pd.Series(pd.to_datetime(ts)).dt.round('60min').dt.time.values
        return np.array([index_conversion_table.loc[index_conversion_table['interval'] == q]['index_60m'][0]
                        for q in rounded_q_time])
    else:
        raise NotImplementedError('`delta_t` other 15 mins and 60 mins is not yet implemented.')

    raise NotImplementedError()
