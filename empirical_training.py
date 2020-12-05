import pickle
from algorithms import sarsa_empirical
from environment import NYCEnv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from snippet import notify
from utils import plot_td_error
import os

# with open('data/SARSA_eps.pickle', 'rb') as f:
#     samples = pickle.load(f)

## variables to set
CHUNK_NUM = 14
# sample_file_prefix = 'SARSA_eps_15m_v02_shA'
dir_name = 'shA_v3'
output_dir_name = f'output/{dir_name}_1204'

emp_Q_save_path = f'{output_dir_name}/emp_Q_v02_shA.pkl'
emp_history_save_path = f'{output_dir_name}/emp_history_shA.pkl'
average_Q_save_path = f'{output_dir_name}/mean_max_Q_shA.png'
td_error_save_path = f'{output_dir_name}/mean_td_error_shA.png'

if not os.path.isdir(output_dir_name):
    print(f'{output_dir_name} does not exist. Create dir')
    os.makedirs(output_dir_name)

sample_list = []
path = 'data/{}/sarsa_{}.pickle' #data path
 
for i in range(1, CHUNK_NUM+1):
    print(path.format(dir_name, i))
    with open(path.format(dir_name, i), 'rb') as f:
        sample_list.append(pickle.load(f))
samples = pd.concat(sample_list, axis=0)
print('sample shape: ', samples.shape)

#### need to set
TRAIN_ITERATION = len(samples)//2


env = NYCEnv(delta_t=10)
print(f'Number of Iterations: {TRAIN_ITERATION}')
Q, history = sarsa_empirical(samples, env.action_space.n, TRAIN_ITERATION, 
                             history_save_path=emp_history_save_path, 
                             Q_save_path=emp_Q_save_path)


with open(emp_Q_save_path, 'wb') as fq:
    pickle.dump(dict(Q), fq)
    
with open(emp_history_save_path, 'wb') as fs:
    pickle.dump(dict(history), fs)
    
print(f'saved at {emp_Q_save_path} and {emp_history_save_path}')


plt.plot(range(len(history['mean_max_q'])), history['mean_max_q'])
plt.title('Average Max Q over States')
plt.xlabel('Iterations (in a hundred)')
plt.savefig(average_Q_save_path, dpi=72)
plt.clf()
print(f'saved at {average_Q_save_path}')


plot_td_error(history['mean_td_delta'], n=5000, save_path=td_error_save_path)
# plt.plot(range(len(history['mean_td_delta'])), history['mean_td_delta'])
# plt.title('Average TD Error')
# plt.xlabel('Epsisode (in a hundred)')
# plt.savefig('output/mean_td_error.png', dpi=72)

plt.clf()

# notify()