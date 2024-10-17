import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch

device = None

def get_env():
    return gym.make('CartPole-v0')

def get_state_distribution(env, n_episodes=100, max_steps=200):
    state_samples = []

    for episode in range(n_episodes):
        state, _ = env.reset()

        for _ in range(max_steps):
            state_samples.append(state)  # Record the state
            action = env.action_space.sample()  # Sample random action
            next_state, _, done, truncated, _ = env.step(action)
            state = next_state
            
            if done or truncated:
                break
                
    return np.array(state_samples)

# Visualize the distribution of observed data
def plot_distribution(observation_data, nbins, file):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    variables = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity']

    for i in range(4):
        obs_var = observation_data[:, i]
        axes[i // 2, i % 2].hist(obs_var, bins=nbins, color='black')
        axes[i // 2, i % 2].set_title(variables[i])
        
    plt.savefig(file)

# Dynamically set bins based on observed data
def get_bins_dynamic(observation_data, nbins):
    bins = np.zeros((nbins, 4))

    for i in range(4):
        _, bins[:, i] = np.histogram(observation_data[:, i], bins=nbins-1)
        
    return bins

# TODO: Implement uniform binning based on observation space
def get_bins_uniform(ranges, nbins):
    pass

# Convert continuous state data to discrete bins
def to_discrete(observation_data, bins):
    binned_data = np.zeros((observation_data.shape[0], 4), dtype=int)
    
    for i in range(4):
        binned_data[:, i] = np.digitize(
            observation_data[:, i] if len(observation_data.shape) > 1 else observation_data[i][None],
            bins[:, i]
        )

    return binned_data

# Wrapper for the to_discrete function
def discretize_state(state, bins):
    return tuple(to_discrete(state[None], bins)[0])

# Preprocess state data for DQN, supports batched and unbatched input
def preprocess(state, normalize=True):
    state_tensor = torch.FloatTensor(np.array(state))
    
    # Normalize state if required
    if normalize:
        state_tensor /= torch.FloatTensor([2.4, 4, np.radians(12), 5]).unsqueeze(0).unsqueeze(0)
        
    return state_tensor.to(device)

# Load a trained model from disk
def load_model(file, model):
    model.load_state_dict(torch.load(file))

# Save a trained model to disk
def save_model(file, model):
    torch.save(model.state_dict(), file)

# Save the weights of the linear approximation
def save_linear_weights(file, weights):
    np.save(file, weights)

# Load the weights of the linear approximation
def load_linear_weights(file):
    return np.load(file)
