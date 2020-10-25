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
    pass
