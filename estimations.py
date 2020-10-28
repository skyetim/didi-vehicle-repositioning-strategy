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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    lookup_table = pd.read_csv('../data/cruise_time_15m.csv')
    return lookup_table.loc[(lookup_table['dropoff_datetime_index'] == t) & 
                            (lookup_table['taxizone_id'] == zone), 'med_cruise_time']
#     raise NotImplementedError()


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
