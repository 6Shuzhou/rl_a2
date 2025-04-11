import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

# -------------------
# 1. Create the Actor-Critic Network
# -------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Actor head: outputs logits for each action
        self.actor = nn.Linear(hidden_dim, action_dim)
        # Critic head: outputs the state value
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Actor part returns logits
        logits = self.actor(x)
        # Critic part returns state value
        value = self.critic(x)
        return logits, value

    def get_action_and_value(self, obs):
        """
        Takes a state 'obs' (np.ndarray) as input and returns (action, log_prob, state_value).
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape: [1, state_dim]
        logits, value = self.forward(obs_tensor)
        action_probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(0)

# -------------------
# 2. Calculate Discounted Returns
# -------------------
def compute_returns(rewards, gamma=0.99):
    """
    Given a list of rewards from an episode, compute the Monte Carlo return G_t:
    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    Returns a PyTorch tensor of the same length as rewards.
    """
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

# -------------------
# 3. Smoothing Function: Moving Average
# -------------------
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# -------------------
# 4. Main Training Loop, Plotting, and Environment Rendering Demo
# -------------------
def main():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n                # 2
    hidden_dim = 128
    learning_rate = 1e-3
    num_episodes = 2000
    gamma = 0.99

    # Create the Actor-Critic network and optimizer
    ac_net = ActorCritic(state_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(ac_net.parameters(), lr=learning_rate)

    reward_history = []
    steps_history = []
    total_steps = 0

    for episode in range(num_episodes):
        obs, info = env.reset()  # New gym API returns (obs, info)
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

        # Record cumulative environment steps (recording cumulative steps for each episode's final step)
        steps_history.append(total_steps)
        # Compute the discounted returns for the entire episode
        returns = compute_returns(rewards, gamma)
        returns = returns.detach()  # No gradient needed

        # Convert list to tensor
        values = torch.stack(values)

        # Compute the advantage: return - estimated state value
        advantages = returns - values.squeeze()

        # Actor loss: -log_prob * advantage
        actor_loss = - (torch.stack(log_probs) * advantages.detach()).sum()
        # Critic loss: Mean Squared Error
        critic_loss = F.mse_loss(values.squeeze(), returns)

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        reward_history.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward (last 50): {avg_reward:.2f}")

    env.close()

    # -------------------
    # Plot the Learning Curve: Reward vs. Cumulative Environment Steps
    # -------------------
    window_size = 50  # Smoothing window size
    smoothed_rewards = moving_average(np.array(reward_history), window_size)
    # Adjust x-axis to account for the reduced number of smoothed data points (starting from window_size-1)
    smoothed_steps = steps_history[window_size-1:]

    plt.figure(figsize=(10, 5))
    plt.plot(steps_history, reward_history, alpha=0.3, label='Raw Rewards')
    plt.plot(smoothed_steps, smoothed_rewards, label=f'{window_size}-step Moving Average', linewidth=2)
    plt.xlabel("Cumulative Environment Steps")
    plt.ylabel("Reward")
    plt.title("Learning Curve: Reward vs. Environment Steps")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("learning_curve_actor_critic.png")
    plt.show()

    # -------------------
    # Demo Phase: Environment Rendering
    # -------------------
    demo_env = gym.make(env_name, render_mode="human")
    obs, info = demo_env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _, _ = ac_net.get_action_and_value(obs)
        obs, reward, terminated, truncated, info = demo_env.step(action)
        demo_env.render()
        time.sleep(0.02)  # Control rendering speed

    demo_env.close()
    print("Final 10 episode rewards:", reward_history[-10:])

if __name__ == "__main__":
    main()
