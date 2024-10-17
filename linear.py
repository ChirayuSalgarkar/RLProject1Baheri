import numpy as np
import Misc.util 
import time

class LinearFunctionApproximator:
  """
  This class represents a linear function approximator, which is a simple yet effective method for estimating Q-values in reinforcement learning. 
  It uses a linear relationship between state-action pairs and Q-values, making it computationally efficient and easier to understand than complex neural networks.

  The key components of this class are:
  - Initialization of parameters: learning rate, discount factor, exploration rate, and exploration decay.
  - Weight matrix initialization: a matrix representing the relationship between features of the state and actions.
  - Action selection: epsilon-greedy strategy to balance exploration and exploitation.
  - Q-value prediction: linear function approximation using the weight matrix and state vector.
  - Weight update: temporal difference (TD) learning to update weights based on experience.
  - Exploration decay: gradually reducing exploration rate over time.
  - Training: main training loop to interact with the environment and update weights.
  """

  def __init__(self, num_actions, state_size, environment, 
               learning_rate=0.1, discount_factor=0.9, 
               exploration_rate=0.1, exploration_decay=1.0):
    """
    Initializes the linear function approximator with the specified parameters.

    Args:
      num_actions: Number of possible actions.
      state_size: Dimensionality of the state space.
      environment: The environment in which the agent interacts.
      learning_rate: Learning rate for weight updates.
      discount_factor: Discount factor to prioritize future rewards.
      exploration_rate: Initial exploration rate for epsilon-greedy strategy.
      exploration_decay: Rate at which exploration decreases.
    """
    # Set up the agent's learning style
    self.num_actions = num_actions  
    self.state_size = state_size    
    self.environment = environment 
    self.learning_rate = learning_rate 
    self.discount_factor = discount_factor 
    self.exploration_rate = exploration_rate 
    self.exploration_decay = exploration_decay  
    
    # Initialize the agent's knowledge (weights) about best actions
    self.weights = np.zeros((state_size, num_actions))  # Weights for each state-action pair 
    self.steps_taken = 0  # Track steps to adjust exploration over time

  def choose_action(self, situation):
    """
    Selects an action based on the epsilon-greedy strategy.

    Args:
      situation: The current state of the environment.

    Returns:
      The chosen action.
    """
    # Should we explore (try something new)?
    if np.random.rand() < self.exploration_rate:
      return np.random.randint(self.num_actions)  # Pick a random action
    else:
      # Pick the action the agent thinks is best based on current knowledge (weights)
      q_values = self.predict(situation)
      return np.argmax(q_values)  # Return the action with the highest predicted Q-value

  def predict(self, situation):
    """
    Predicts the Q-values for all actions given a state using linear function approximation.

    Args:
      situation: The current state of the environment.

    Returns:
      A numpy array of Q-values for all actions.
    """
    # Calculate scores (dot product of state and weight matrix)
    return np.dot(situation, self.weights)

  def learn_from_experience(self, situation, action, reward, new_situation, is_done):
    """
    Updates the weights based on the temporal difference (TD) update rule.

    Args:
      situation: The current state of the environment.
      action: The chosen action.
      reward: The received reward.
      new_situation: The next state of the environment.
      is_done: Whether the episode is over.
    """
    # Remember the predicted goodness for the chosen action
    current_q = np.dot(situation, self.weights[:, action])

    # Calculate target Q-value based on next state
    if is_done:
      target_q = reward  # If the episode ends, target is just the reward
    else:
      next_q_values = self.predict(new_situation)
      target_q = reward + self.discount_factor * np.max(next_q_values)  # Discounted future reward

    # Compute TD error
    td_error = target_q - current_q

    # Update weights for the selected action using gradient descent
    self.weights[:, action] += self.learning_rate * td_error * situation

  def decay_exploration(self):
    """
    Gradually decays the exploration rate over time to favor exploitation over exploration.
    """
    self.exploration_rate *= self.exploration_decay

  def train(self, num_episodes):
    """
    The main training loop for the linear function approximator.

    Args:
      num_episodes: The number of episodes to train for.
    """
    for episode in range(num_episodes):
      situation = self.environment.reset()
      is_done = False

      while not is_done:
        action = self.choose_action(situation)
        new_situation, reward, is_done, _ = self.environment.step(action)
        self.learn_from_experience(situation, action, reward, new_situation, is_done)
        situation = new_situation

      # After each episode, decay the exploration rate
      self.decay_exploration()
