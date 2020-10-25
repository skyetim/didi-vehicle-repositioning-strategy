import gym
from gym import spaces

# TODO: wt
class NYCEnv(gym.Env):
    def __init__(self):
        super(NYCEnv, self).__init__()
        self.action_space = None
        self.observation_space = None

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError