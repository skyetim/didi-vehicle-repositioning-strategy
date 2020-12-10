# adapted from: https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA%20Solution.ipynb
from collections import defaultdict
import itertools
import numpy as np
import pandas as pd
import plotting
import sys
from tqdm.auto import trange
import pickle


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def hash_state(state):
    return (state[0], state[1])


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        hashed_state = hash_state(state)
        action_probs = policy(hashed_state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # One step in the environment
        for t in itertools.count():
            # Take a step
            next_state, reward, done, _ = env.step(action)
            hashed_next_state = hash_state(next_state)

            # Pick the next action
            next_action_probs = policy(hashed_next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            td_target = reward + discount_factor * Q[hashed_next_state][next_action]
            td_delta = td_target - Q[hashed_state][action]
            Q[hashed_state][action] += alpha * td_delta

            if done:
                break

            action = next_action
            state = next_state

    return Q, stats


def sarsa_empirical(samples, num_actions, num_episodes=None, discount_factor=1., alpha=0.5, batch_size=32, history_save_path=None,
                   Q_save_path=None, Q=None, history=None):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        history_save_path: path to periodically save history
        Q_save_path: path to periodically save Q
        Q: initial Q to start the training with. Initialized to all 0 if None
        history: initial history to start the training with. Initialized to all 0 if None
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    if Q is None:
        Q = defaultdict(lambda: np.zeros(num_actions))
    if history is None:
        history = defaultdict(list)

    total_samples = samples.shape[0]
    num_episodes = num_episodes or (total_samples // batch_size)

    for cur_episode in trange(num_episodes, desc='Episode'):
        batch_td_error = 0
        for cur_sample in np.random.choice(total_samples, batch_size):
            _, state, action, reward, next_state, next_action = samples.iloc[cur_sample]

            assert (action == int(action)) and (pd.isnull(next_action) or (next_action == int(next_action))), 'Action is a float'
            action = int(action)
            next_action = int(next_action) if not pd.isnull(next_action) else next_action

            # TD Update
            EQ = Q[next_state][next_action] if not pd.isnull(next_action) else 0
            td_target = reward + discount_factor * EQ
            td_delta = td_target - Q[state][action]
            batch_td_error += td_delta
            Q[state][action] += alpha * td_delta

        if cur_episode % 100 == 0:
            history['mean_max_q'].append(np.mean([i.max() for i in Q.values()]))
        history['mean_td_delta'].append(batch_td_error/batch_size)
        
        cp = 500000 ## save every `cp` iterations
        save_every_percent = 10
        if (cur_episode+1) % cp == 0:
            with open(Q_save_path, 'wb') as handle:
                pickle.dump(dict(Q), handle)
            with open(history_save_path, 'wb') as handle:
                pickle.dump(dict(history), handle)
            print(f'checkpoints saved at {Q_save_path} and {history_save_path}')

    return Q, history
