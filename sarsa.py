import numpy as np
import time as t
import Misc

class SarsaAlgorithm:
  """
  This class represents the Sarsa algorithm for reinforcement learning, which is a temporal difference (TD) method.
  It learns a Q-function that estimates the expected future reward for taking a specific action in a given state.

  Key components:
  - Q-table initialization: Creates a Q-table with zeros.
  - Action selection: Epsilon-greedy strategy to balance exploration and exploitation.
  - Q-value update: Sarsa update rule to update Q-values based on experience.
  - Training: Main training loop to interact with the environment and learn.
  """

  def __init__(self, nActs, stateDim, environment, binValues, alpha=0.1, discount=0.9, eps=0.1, epsDecay=1.0):
    """
    Initializes the Sarsa algorithm with the specified parameters.

    Args:
      nActs: Number of possible actions.
      stateDim: Dimensionality of the state space.
      environment: The environment in which the agent interacts.
      binValues: Bins for discretizing the continuous state space.
      alpha: Learning rate for Q-value updates.
      discount: Discount factor to prioritize future rewards.
      eps: Initial exploration probability for epsilon-greedy strategy.
      epsDecay: Rate at which exploration probability decreases.
    """
    # Set up the agent's learning parameters
    self.actions = nActs
    self.stateDimension = stateDim
    self.env = environment
    self.binDividers = binValues
    self.alpha = alpha
    self.discountFactor = discount
    self.eps = eps
    self.epsDecay = epsDecay

    # Initialize the Q-table
    self.Q_matrix = self._initialize_q_table()

  def _initialize_q_table(self):
    """
    Initializes the Q-table with zeros.

    Returns:
      The initialized Q-table.
    """
    gridShape = tuple([self.binDividers.shape[0] + 1 for _ in range(self.stateDimension)] + [self.actions])
    return np.zeros(gridShape)

  def _pick_action(self, curState):
    """
    Selects an action based on the epsilon-greedy strategy.

    Args:
      curState: The discretized state of the environment.

    Returns:
      The chosen action.
    """
    if np.random.rand() < self.eps:
      return np.random.choice(self.actions)  # Explore
    return np.argmax(self.Q_matrix[curState])  # Exploit

  def _update_Q_value(self, curState, curAct, rewardVal, nextState, terminationFlag):
    """
    Updates the Q-value for the given state-action pair using the Sarsa update rule.

    Args:
      curState: The discretized state of the environment.
      curAct: The chosen action.
      rewardVal: The received reward.
      nextState: The discretized next state of the environment.
      terminationFlag: Whether the episode has ended.

    Returns:
      The next action if the episode is not terminated, None otherwise.
    """
    currentValue = self.Q_matrix[curState + (curAct,)]

    if terminationFlag:
      futureValue = 0  # No future Q-value if terminal state
    else:
      nextAct = self._pick_action(nextState)
      futureValue = self.Q_matrix[nextState + (nextAct,)]

    delta = rewardVal + self.discountFactor * futureValue - currentValue
    self.Q_matrix[curState + (curAct,)] = currentValue + self.alpha * delta

    return None if terminationFlag else nextAct

  def train_model(self, episodes=1000, logInterval=1000, logRewards=None, logTimes=None):
    """
    Trains the Sarsa algorithm over a specified number of episodes.

    Args:
      episodes: The number of episodes to train for.
      logInterval: The interval at which to log average rewards.
      logRewards: A list to store episode rewards for logging.
      logTimes: A list to store training times for logging.

    Returns:
      A list of episode rewards.
    """
    rewardHistory = []

    if logRewards is not None:
      logRewards[0] = 0
    if logTimes is not None:
      logTimes[0] = t.time()

    for ep in range(episodes):
      initialState, _ = self.env.reset()
      discretizedState = Misc.util.discretize_state(initialState, self.binDividers)
      currentAction = self._pick_action(discretizedState)
      epReward = 0
      self.eps *= self.epsDecay

      for _ in range(300):  # Limit steps per episode
        newState, reward, done, terminated, _ = self.env.step(currentAction)
        discretizedNextState = Misc.util.discretize_state(newState, self.binDividers)
        epReward += reward

        nextAction = self._update_Q_value(discretizedState, currentAction, reward, discretizedNextState, done or terminated)

        if done or terminated:
          break

        discretizedState = discretizedNextState
        currentAction = nextAction

      rewardHistory.append(epReward)

      if logRewards is not None:
        logRewards[ep + 1] = epReward
      if logTimes is not None:
        logTimes[ep + 1] = t.time()

      if ep % logInterval == 0:
        print(f"Episode {ep}: Avg Reward (last {logInterval}): {np.mean(rewardHistory[-logInterval:]):.2f}")

    return rewardHistory