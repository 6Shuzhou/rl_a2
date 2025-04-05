import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

# -------------------
# 1. Create the Policy Network
# -------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # Forward pass: outputs logits for each action
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_action(self, obs):
        """
        Given the state obs (np.ndarray), return (action, log_prob)
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(obs_tensor)
        action_probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

# -------------------
# Calculate Discounted Returns
# -------------------
def compute_returns(rewards, gamma=0.99):
    """
    For a sequence of rewards in an episode, compute the discounted return at each timestep:
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    Returns a PyTorch tensor of the same length as rewards.
    """
    discounted_returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_returns.insert(0, R)
    return torch.tensor(discounted_returns, dtype=torch.float32)

# -------------------
# Moving Average Function to Smooth Curves
# -------------------
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# -------------------
# Main function: Training, Plotting Learning Curve, and Demo Animation
# -------------------
def main():
    # Create environment and set hyperparameters
    env_name = "CartPole-v1"
    env = gym.make(env_name)  # No animation during training

    # CartPole has 4 state dimensions and 2 action choices
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    hidden_dim = 128
    learning_rate = 1e-3
    num_episodes = 2000
    gamma = 0.99

    # Initialize the policy network and optimizer
    policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    reward_history = []      # Record total reward per episode
    steps_history = []       # Record cumulative environment steps
    total_steps = 0          # Environment steps counter

    # -------------------
    # Training Loop
    # -------------------
    for episode in range(num_episodes):
        obs, info = env.reset()  # New gym interface returns (obs, info)
        terminated = False
        truncated = False

        log_probs = []
        rewards = []

        while not (terminated or truncated):
            action, log_prob = policy_net.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            obs = next_obs
            total_steps += 1

        # Compute discounted returns
        returns = compute_returns(rewards, gamma)

        # Loss function for REINFORCE algorithm
        loss = 0
        for log_prob, Gt in zip(log_probs, returns):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_return = sum(rewards)
        reward_history.append(episode_return)
        steps_history.append(total_steps)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward (last 50): {avg_reward:.2f}")

    env.close()

    # -------------------
    # Plot the Learning Curve: Reward vs. Cumulative Environment Steps
    # -------------------
    window_size = 50  # Smoothing window size
    smoothed_rewards = moving_average(np.array(reward_history), window_size)
    # Adjust x-axis steps since the smoothed data has window_size - 1 fewer points
    smoothed_steps = steps_history[window_size-1:]

    plt.figure(figsize=(10, 5))
    plt.plot(steps_history, reward_history, alpha=0.3, label='Raw Rewards')
    plt.plot(smoothed_steps, smoothed_rewards, label=f'{window_size}-step Moving Average', linewidth=2)
    plt.xlabel("Cumulative Environment Steps")
    plt.ylabel("Reward")
    plt.title("Learning Curve: Reward vs. Environment Steps")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("learning_curve_reinforce.png")
    plt.show()

    # -------------------
    # Evaluation Phase: Display Environment Animation
    # -------------------
    demo_env = gym.make(env_name, render_mode="human")
    obs, info = demo_env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        demo_env.render()
        action, _ = policy_net.get_action(obs)
        obs, reward, terminated, truncated, info = demo_env.step(action)
        time.sleep(0.02)  # Control animation speed

    demo_env.close()
    print("Final 10 episode rewards:", reward_history[-10:])

if __name__ == "__main__":
    main()
