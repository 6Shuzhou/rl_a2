# Actor_Critic.py
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action_and_value(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self.forward(obs_tensor)
        action_probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(0)

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def train_actor_critic(env_name="CartPole-v1", num_episodes=2000, hidden_dim=128,
                       learning_rate=1e-3, gamma=0.99):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ac_net = ActorCritic(state_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(ac_net.parameters(), lr=learning_rate)

    reward_history = []
    steps_history = []
    total_steps = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False

        log_probs = []
        values = []
        rewards = []

        while not (terminated or truncated):
            action, log_prob, value = ac_net.get_action_and_value(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            obs = next_obs
            total_steps += 1

        steps_history.append(total_steps)
        returns = compute_returns(rewards, gamma).detach()
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        advantages = returns - values.squeeze()

        actor_loss = - (log_probs * advantages.detach()).sum()
        critic_loss = F.mse_loss(values.squeeze(), returns)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_return = sum(rewards)
        reward_history.append(episode_return)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"[Actor-Critic] Episode {episode+1}/{num_episodes}, Avg Reward (last 50): {avg_reward:.2f}")

    env.close()
    return steps_history, reward_history
