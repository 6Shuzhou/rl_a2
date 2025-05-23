# reinforce.py
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

    def get_action(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(obs_tensor)
        action_probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def train_reinforce(env_name="CartPole-v1", num_episodes=2000, hidden_dim=128,
                    learning_rate=1e-3, gamma=0.99):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    reward_history = []
    steps_history = []
    total_steps = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
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

        steps_history.append(total_steps)
        returns = compute_returns(rewards, gamma)

        loss = 0
        for log_prob, Gt in zip(log_probs, returns):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_return = sum(rewards)
        reward_history.append(episode_return)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"[REINFORCE] Episode {episode+1}/{num_episodes}, Avg Reward (last 50): {avg_reward:.2f}")

    env.close()
    return steps_history, reward_history
