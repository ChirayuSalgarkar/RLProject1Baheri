import numpy as np
import Misc.util 
import torch.optim as optim
import torch.nn as nn
import torch
from collections import deque
import random
import time

class DQN:
  """
  This class represents a Deep Q-Network (DQN), a popular reinforcement learning algorithm. 
  It trains an agent to make optimal decisions in an environment using deep neural networks.

  Key components of a DQN:
  - Policy network: Estimates Q-values for actions in a given state.
  - Target network: Used to compute target Q-values for training stability.
  - Replay memory: Stores past experiences (state, action, reward, next state, done) for learning.
  - Epsilon-greedy strategy: Balances exploration (trying new actions) and exploitation (choosing best action based on current knowledge).
  - Experience replay: Training on random samples from memory for better generalization.
  - Target network update: Periodically updates the target network with the policy network's weights to stabilize learning.
  """

  def __init__(self, policy_net, target_net, environment,
               learning_rate=0.1, discount_factor=0.9, epsilon=0.1, epsilon_decay=1.0,
               memory_size=10000, target_update_frequency=128):
    """
    Initializes the DQN agent with the specified parameters.

    Args:
      policy_net: The neural network used to estimate Q-values.
      target_net: The neural network used to compute target Q-values.
      environment: The environment in which the agent interacts.
      learning_rate: Learning rate for weight updates.
      discount_factor: Weight given to future rewards during learning.
      epsilon: Initial exploration rate for epsilon-greedy strategy.
      epsilon_decay: Rate at which exploration decreases.
      memory_size: Maximum number of experiences to store in replay memory.
      target_update_frequency: How often to update the target network with the policy network's weights.
    """
    # Set up the agent's learning parameters
    self.environment = environment
    self.memory = deque(maxlen=memory_size)  # Experience replay memory
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay

    # Networks and optimizer
    self.policy_net = policy_net
    self.target_net = target_net
    # Set target network to evaluation mode (no updates during training)
    self.target_net.eval()
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    self.criterion = nn.MSELoss()  # Loss function for training

    # Target network update counter
    self.update_counter = 0
    self.target_update_frequency = target_update_frequency

  def store_experience(self, state, action, reward, next_state, done, truncated):
    """
    Adds a new experience (state, action, reward, next state, done) to the replay memory.

    Args:
      state: The current state of the environment.
      action: The action taken by the agent.
      reward: The reward received for the action.
      next_state: The next state after taking the action.
      done: Whether the episode has ended.
      truncated: Whether the episode was terminated early.
    """
    self.memory.append((state, action, reward, next_state, done, truncated))

  def choose_action(self, state):
    """
    Selects an action based on the epsilon-greedy strategy.

    Args:
      state: The current state of the environment.

    Returns:
      The chosen action.
    """
    # Explore (try a random action) with some probability (epsilon)
    if np.random.random() < self.epsilon:
      return np.random.randint(0, self.environment.action_space.n)  # Explore
    else:
      # Exploit (choose best action based on current knowledge)
      # Ensure state has the correct shape and convert it to a tensor
      state = Misc.util.preprocess(state)
      state = torch.tensor(state, dtype=torch.float32)
      state = state.unsqueeze(0)

      # Evaluate the policy network in evaluation
