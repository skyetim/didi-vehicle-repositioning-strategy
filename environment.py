import numpy as np
import gym
from gym import spaces
from estimations import (cruise_time, trip_time, generate_request, trip_fare, trip_distance, generate_request_load_data)
from utils import is_adjacent


class NYCEnv(gym.Env):
    def __init__(self):
        super(NYCEnv, self).__init__()
        self.NUM_TAXI_ZONES = 10
        self.SHIFT_START_TIME = 0
        self.SHIFT_DURATION = 10
        self.FUEL_UNIT_PRICE = .125
        self.action_space = spaces.Discrete(self.NUM_TAXI_ZONES + 1)
        self.observation_space = spaces.Box(
            low=np.array([1, self.SHIFT_START_TIME]),
            high=np.array([self.NUM_TAXI_ZONES, self.SHIFT_START_TIME + self.SHIFT_DURATION]),
            dtype=np.int8,
        )
        self._load_data()

    def step(self, action):
        if action == 0:
            return self._wait()
        if action == self.current_taxi_zone:
            return self._hunt()
        if not is_adjacent(action, self.current_taxi_zone):
            return self._fly()
        return self._cruise_to_adjacent_taxi_zone(action)

    def reset(self):
        self.total_rewards = 0
        self.current_taxi_zone = np.random.randint(1, self.NUM_TAXI_ZONES + 1)
        self.current_time = self.SHIFT_START_TIME
        return np.array([self.current_taxi_zone, self.current_time])

    def render(self, mode='console'):
        if mode != 'console':
            return NotImplementedError('Mode other than console is not yet implemented.')
        print(
            f'Current taxi zone: {self.current_taxi_zone}, current time: {self.current_time}, current reward: {self.total_rewards}'
        )

    def _check_done(self):
        return self.current_time == self.SHIFT_START_TIME + self.SHIFT_DURATION

    def _wait(self):
        self.current_time += 1
        info = {}
        return np.array([self.current_taxi_zone, self.current_time]), 0, self._check_done(), info

    def _hunt(self):
        self.current_time += cruise_time(self.current_taxi_zone, self.current_time)
        info = {}
        if self._check_done():
            return np.array([self.current_taxi_zone, self.current_time]), 0, True, info
        dst = generate_request(self.request_transition_matrix, sself.current_taxi_zone, self.current_time)
        self.current_time += trip_time(self.current_taxi_zone, dst, self.current_time)
        reward = trip_fare(self.current_taxi_zone, dst, self.current_time)
        reward -= self.FUEL_UNIT_PRICE * trip_distance(self.current_taxi_zone, dst, self.current_time)
        self.current_taxi_zone = dst
        self.total_rewards += reward
        return np.array([self.current_taxi_zone, self.current_time]), reward, self._check_done(), info

    def _fly(self):
        reward = -float('inf')
        info = {}
        self.total_rewards += reward
        return np.array([self.current_taxi_zone, self.SHIFT_START_TIME + self.SHIFT_DURATION]), reward, True, info

    def _cruise_to_adjacent_taxi_zone(self, action):
        reward = -self.FUEL_UNIT_PRICE * trip_distance(self.current_taxi_zone, action)
        info = {}
        self.current_taxi_zone = action
        self.current_time += trip_time(self.current_taxi_zone, action)
        return np.array([self.current_taxi_zone, self.current_time]), reward, self._check_done(), info

    def _load_data(self):
        self.request_transition_matrix = generate_request_load_data()
