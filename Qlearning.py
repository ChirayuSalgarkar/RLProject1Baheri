import numpy as np
import Misc.util
import time


class DiscreteQLearner:
  """
  This class represents a Q-learning agent for discrete state and action spaces.
  It uses a Q-table to store estimated Q-values for each state-action pair.

  Key components:
  - Q-table initialization: Creates a Q-table with zeros.
  - Action selection: Epsilon-greedy strategy to balance exploration and exploitation.
  - Q-value update: Bellman equation to update Q-values based on experience.
  - Training: Main training loop to interact with the environment and learn.
  """

  def __init__(self, action_count, state_dimension, environment, discretization_bins, alpha=0.1, discount_factor=0.9, exploration_prob=0.1, exploration_decay=1.0):
    """
    Initializes the Q-learning agent with the specified parameters.

    Args:
      action_count: Number of possible actions.
      state_dimension: Dimensionality of the state space.
      environment: The environment in which the agent interacts.
      discretization_bins: Bins for discretizing the continuous state space.
      alpha: Learning rate for Q-value updates.
      discount_factor: Discount factor to prioritize future rewards.
      exploration_prob: Initial exploration probability for epsilon-greedy strategy.
      exploration_decay: Rate at which exploration probability decreases.
    """
    # Set up the agent's learning parameters
    self.action_count = action_count
    self.state_dimension = state_dimension
    self.environment = environment
    self.discretization_bins = discretization_bins
    self.alpha = alpha
    self.discount_factor = discount_factor
    self.exploration_prob = exploration_prob
    self.exploration_decay = exploration_decay

    # Initialize the Q-table
    self.q_table = self._initialize_q_table()

  def _initialize_q_table(self):
    """
    Initializes the Q-table with zeros.

    Returns:
      The initialized Q-table.
    """
    table_shape = tuple([self.discretization_bins.shape[0] + 1 for _ in range(self.state_dimension)] + [self.action_count])
    return np.zeros(table_shape)

  def select_action(self, discrete_state):
    """
    Selects an action based on the epsilon-greedy strategy.

    Args:
      discrete_state: The discretized state of the environment.

    Returns:
      The chosen action.
    """
    if np.random.random() < self.exploration_prob:
      return np.random.randint(0, self.action_count)  # Explore
    else:
      return np.argmax(self.q_table[discrete_state])  # Exploit

  def adjust_q_value(self, discrete_state, selected_action, reward, next_discrete_state, terminal):
    """
    Updates the Q-value for the given state-action pair using the Bellman equation.

    Args:
      discrete_state: The discretized state of the environment.
      selected_action: The chosen action.
      reward: The received reward.
      next_discrete_state: The discretized next state of the environment.
      terminal: Whether the episode has ended.
    """
    current_q = self.q_table[discrete_state + (selected_action,)]

    if terminal:
      max_future_q = 0  # No future Q-value if terminal state
    else:
      max_future_q = np.max(self.q_table[next_discrete_state])

    # Temporal difference (TD) update rule for Q-learning
    updated_q = current_q + self.alpha * (reward + (self.discount_factor * max_future_q) - current_q)
    self.q_table[discrete_state + (selected_action,)] = updated_q

  def train_agent(self, episode_count=1000, log_interval=1000, reward_log=None, time_log=None):
    """
    Trains the Q-learning agent over a specified number of episodes.

    Args:
      episode_count: The number of episodes to train for.
      log_interval: The interval at which to log average rewards.
      reward_log: A list to store episode rewards for logging.
      time_log: A list to store training times for logging.

    Returns:
      A list of episode rewards.
    """
    reward_per_episode = []

    if reward_log is not None:
      reward_log[0] = 0
    if time_log is not None:
      time_log[0] = time.time()

    for episode in range(episode_count):
      initial_state, _ = self.environment.reset()
      discrete_state = Misc.util.discretize_state(initial_state, self.discretization_bins)
      episode_reward = 0
      self.exploration_prob *= self.exploration_decay

      for step in range(300):  # Limit steps per episode
        action = self.select_action(discrete_state)
        next_state, reward, done, truncated, _ = self.environment.step(action)
        next_discrete_state = Misc.util.discretize_state(next_state, self.discretization_bins)
        episode_reward += reward

        self.adjust_q_value(discrete_state, action, reward, next_discrete_state, done or truncated)

        if done or truncated:
          break

        discrete_state = next_discrete_state

      reward_per_episode.append(episode_reward)

      if reward_log is not None:
        reward_log[episode + 1] = episode_reward
      if time_log is not None:
        time_log[episode + 1] = time.time()

      if episode % log_interval == 0:
        avg_reward = np.mean(reward_per_episode[-log_interval:])
        print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

    return reward_per_episode