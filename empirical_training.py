import pickle
from algorithms import sarsa_empirical
from environment import NYCEnv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from snippet import notify

with open('data/SARSA_eps.pickle', 'rb') as f:
    samples = pickle.load(f)

env = NYCEnv(delta_t=10)

Q, history = sarsa_empirical(samples, env.action_space.n, len(samples))

plt.plot(range(len(history['mean_max_q'])), history['mean_max_q'])

plt.savefig('output/mean_max_Q.png', dpi=72)

plt.clf()

plt.plot(range(len(history['mean_td_delta'])), history['mean_td_delta'])

plt.savefig('output/mean_td_error.png', dpi=72)

with open('output/emp_Q.pkl', 'wb') as fq, open('output/emp_history.pkl', 'wb') as fs:
    pickle.dump(dict(Q), fq)
    pickle.dump(dict(history), fs)

notify()