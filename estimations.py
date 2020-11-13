import pandas as pd
import numpy as np
from random import choices


class Estimator:

    def __init__(self, dir_path=''):
        self.data_fare = pd.read_csv(dir_path + 'fare_amount_src_dst_t.csv')
        self.data_distance = pd.read_csv(dir_path + 'trip_distance_src_dst.csv')
        self.data_time = pd.read_csv(dir_path + 'trip_time_src_dst_t.csv')
        self.data_cruise15 = pd.read_csv(dir_path + 'cruise_time_15m.csv')
        self.data_mc2d = np.loadtxt(dir_path + "mc_mtx.txt")
        self.data_index = pd.read_csv(dir_path + 'interval_index_table_0.csv')
        self.data_index['interval'] = pd.to_datetime(
            self.data_index['interval']).dt.time
        self.data_adjacent = pd.read_csv(dir_path + 'adjacent_zone.csv')

    def trip_fare(self, src, dst, t):
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

        match = self.data_fare
        if (not isinstance(src, (list, tuple, np.ndarray))):
            src = [src]
            dst = [dst]
            t = [t]
            m = match.loc[(match['pickup_taxizone_id'].isin(src)) &
                      (match['dropoff_taxizone_id'].isin(dst))]
            m = m.loc[(m['pickup_datetime_index'].isin(t)), 'mean']
            if m.shape[0] == 0:
                result = -1
            else:
                result = m.values[0]
        else:
            m = match.loc[(match['pickup_taxizone_id'].isin(src)) &
                      (match['dropoff_taxizone_id'].isin(dst))]
            m = m.loc[(m['pickup_datetime_index'].isin(t)), 'mean']
            if m.shape[0] == 0:
                result = -1
            else:
                result = m.values
        return result

    # TODO: ty & xx

    def trip_distance(self, src, dst):
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
        match = self.data_distance
        if (not isinstance(src, (list, tuple, np.ndarray))):
            src = [src]
            dst = [dst]
            m = match.loc[(match['pickup_taxizone_id'].isin(src)) & (
                match['dropoff_taxizone_id'].isin(dst)), 'mean']
            if m.shape[0] == 0:
                result = -1
            else:
                result = m.values[0]
        else:
            m = match.loc[(match['pickup_taxizone_id'].isin(src)) & (
                match['dropoff_taxizone_id'].isin(dst)), 'mean']
            if m.shape[0] == 0:
                result = -1
            else:
                result = m.values
        return result
    

    # TODO: ty & xx

    def trip_time(self, src, dst, t):
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
        match = self.data_time
        if (not isinstance(src, (list, tuple, np.ndarray))):
            src = [src]
            dst = [dst]
            t = [t]
            m = match.loc[(match['pickup_taxizone_id'].isin(src)) &
                      (match['dropoff_taxizone_id'].isin(dst))]
            m = m.loc[(m['pickup_datetime_index'].isin(t)), 'mean']
            if m.shape[0] == 0:
                result = -1
            else:
                 result = int(round(1.0*m.values[0]/15)) 
        else:
            m = match.loc[(match['pickup_taxizone_id'].isin(src)) &
                      (match['dropoff_taxizone_id'].isin(dst))]
            m = m.loc[(m['pickup_datetime_index'].isin(t)), 'mean']
            if m.shape[0] == 0:
                result = -1
            else:
                 result = [int(round(1.0*z/15)) for z in m.values]
        return result

    # TODO: yy & bo

    def cruise_time(self, zone, t):
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
        # NOT VECTORIZED
        t_interval = 15
        lookup_table = self.data_cruise15  # only implemented for 15 mins

        q = lookup_table.loc[(lookup_table['dropoff_datetime_index'] == t) & (lookup_table['taxizone_id'] == zone),
                             'med_cruise_time_INT']
        if q.shape[0] == 0:
            return 1000//t_interval
        else:
            return q.values[0]

    # placeholder

    def cruise_distance(self, zone, t):
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

    def generate_request(self, zone, t):
        """generate a random request in `zone` at time t
        Parameters
        ----------
        data: np 3d array
            the Markov Chain Transition Matrix
        zone : int
            taxi zone id
        t : int
            time
        Returns
        -------
        int
            dst taxi zone
        """

        zone_num = 263
        mc_mtx_2d = self.data_mc2d
        # convert back to 3d array
        mc_mtx = mc_mtx_2d.reshape(
            mc_mtx_2d.shape[0], mc_mtx_2d.shape[1] // zone_num, zone_num)

        population = np.arange(1, zone_num + 1)
        # index starts from 0, zone starts from 1
        weights = mc_mtx[t, zone - 1, :]

        if (weights == np.zeros((1, zone_num))).all():
            uniform = np.full((zone_num, 1), 1 / zone_num)
            dst = choices(population, uniform)[0]
        else:
            dst = choices(population, weights)[0]

        return dst

    # TODO: yy & bo

    def timestamp_to_env_time(self, ts, delta_t=15, t0=0):
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

        if (t0 < 0) | (t0 > 23):
            raise ValueError('t must be in between 0 and 23 inclusively')

        ts = pd.Series(pd.to_datetime(ts))

        # convert string to pd.datetime [move to __init__]
        index_conversion_table = self.data_index
#         index_conversion_table['interval'] = pd.to_datetime(index_conversion_table['interval']).dt.time

        if delta_t == 15:
            rounded_q_time = ts.dt.round('15min').dt.time.values
            env_time = np.array([index_conversion_table.loc[index_conversion_table['interval'] == q]['index_15m'].values[0]
                                 for q in rounded_q_time])
        elif delta_t == 60:
            rounded_q_time = pd.Series(pd.to_datetime(
                ts)).dt.round('60min').dt.time.values
            env_time = np.array([index_conversion_table.loc[index_conversion_table['interval'] == q]['index_60m'].values[0]
                                 for q in rounded_q_time])
        else:
            raise NotImplementedError(
                'delta_t other 15 mins and 60 mins is not yet implemented.')

        # Handle when the time does not start at 12:00am
        if t0 != 0:
            env_time = np.array([i+24 if i < 0 else i for i in (env_time - t0)])

        return env_time

    # TODO: ty & xx

    def is_adjacent(self, src, dst):
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
        ad = self.data_adjacent
        if dst < src:
            a = dst
            b = src
            r = ad.loc[(ad.zone1 == a) & (ad.zone2 == b)]
        else:
            r = ad.loc[(ad.zone1 == src) & (ad.zone2 == dst)]

        return (len(r) > 0)
