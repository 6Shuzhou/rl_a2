import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# -------------------
# 1. Create Policy Network
# -------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # Forward pass: output logits for each action
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_action(self, obs):
        """
        Given a state obs (np.ndarray), 
        return (action, log_prob)
        """
        # Convert to float32 numpy array, then to tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape: [1, state_dim]

        logits = self.forward(obs_tensor)           # shape: [1, action_dim]
        action_probs = F.softmax(logits, dim=1)     # apply softmax to obtain probabilities for each action

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


def compute_returns(rewards, gamma=0.99):
    """
    Given a list of immediate rewards for a trajectory, compute the Monte Carlo return G_t for each timestep:
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    Returns a PyTorch tensor of the same length as rewards, representing the discounted cumulative return from that step onward.
    """
    discounted_returns = []
    R = 0
    # Compute backwards
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_returns.insert(0, R)
    # Convert to a PyTorch tensor
    discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)
    return discounted_returns


def main():
    # -------------------
    # 2. Create Environment & Hyperparameters
    # -------------------
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    # For CartPole, the observation space is 4-dimensional, and the action space consists of 2 discrete actions
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n             # 2

    # Hyperparameters
    hidden_dim = 128
    learning_rate = 1e-3
    num_episodes = 2000
    gamma = 0.99

    # Create the policy network and optimizer
    policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    # For tracking rewards for each episode
    reward_history = []

    # -------------------
    # 3. Training Loop
    # -------------------
    for episode in range(num_episodes):
        # New interface reset: returns (obs, info)
        obs, info = env.reset()
        terminated = False
        truncated = False

        log_probs = []
        rewards   = []

        while not (terminated or truncated):
            action, log_prob = policy_net.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store log_prob and reward
            log_probs.append(log_prob)
            rewards.append(reward)

            obs = next_obs

        # After the episode ends, compute the discounted returns
        returns = compute_returns(rewards, gamma)

        # Compute the REINFORCE loss
        loss = 0
        for log_prob, Gt in zip(log_probs, returns):
            loss += -log_prob * Gt

        # -------------------
        # 4. Perform a single gradient update
        # -------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record the total reward of the current episode
        total_reward = sum(rewards)
        reward_history.append(total_reward)

        # Print training progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward (last 50): {avg_reward:.2f}")

    env.close()

    # -------------------
    # 5. Evaluate Training Performance
    # -------------------
    print("Final 10 episode rewards:", reward_history[-10:])


if __name__ == "__main__":
    main()
