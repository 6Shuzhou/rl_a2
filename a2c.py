import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

# -------------------
# 1. Create the Actor-Critic (A2C) Network
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
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action_and_value(self, obs):
        """
        Given an observation, returns (action, log_prob, state_value, entropy).
        The entropy value is used as an exploration bonus.
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape: [1, state_dim]
        logits, value = self.forward(obs_tensor)
        action_probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()  # an average entropy value
        return action.item(), log_prob, value.squeeze(0), entropy

# -------------------
# 2. Compute Discounted Returns
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
# 3. Smoothing Function: Moving Average (for plotting)
# -------------------
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# -------------------
# 4. Main Training Loop: Advantage Actor-Critic
# -------------------
def main():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]   
    action_dim = env.action_space.n               
    hidden_dim = 128
    learning_rate = 1e-3
    num_episodes = 2000
    gamma = 0.99
    entropy_coef = 0.001  # Coefficient for the entropy bonus

    # Create the Actor-Critic network and the optimizer
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
        entropies = []

        while not (terminated or truncated):
            action, log_prob, value, entropy = ac_net.get_action_and_value(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

            obs = next_obs
            total_steps += 1

        steps_history.append(total_steps)
        total_reward = sum(rewards)
        reward_history.append(total_reward)

        # Compute discounted returns for the episode
        returns = compute_returns(rewards, gamma).detach()

        # Stack the list of values so that it forms a tensor.
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        # Compute advantages: advantage = discounted return - value estimate
        advantages = returns - values.squeeze()

        # Actor Loss: policy gradient loss weighted by the advantage
        actor_loss = - (log_probs * advantages.detach()).sum()
        # Critic Loss: Mean Squared Error loss to train the value function
        critic_loss = F.mse_loss(values.squeeze(), returns)
        # Entropy Bonus: to encourage exploration
        entropy_loss = - entropies.sum()

        # Total Loss: combine actor, critic, and entropy losses
        loss = actor_loss + critic_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward (last 50): {avg_reward:.2f}")

    env.close()

    # -------------------
    # Plot the Learning Curve: Reward vs. Cumulative Environment Steps
    # -------------------
    window_size = 50  # Smoothing window size for plotting
    smoothed_rewards = moving_average(np.array(reward_history), window_size)
    smoothed_steps = steps_history[window_size-1:]

    plt.figure(figsize=(10, 5))
    plt.plot(steps_history, reward_history, alpha=0.3, label='Raw Rewards')
    plt.plot(smoothed_steps, smoothed_rewards, label=f'{window_size}-step Moving Average', linewidth=2)
    plt.xlabel("Cumulative Environment Steps")
    plt.ylabel("Reward")
    plt.title("Learning Curve: Reward vs. Environment Steps (A2C)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("learning_curve_a2c.png")
    plt.show()

    # -------------------
    # Demo Phase: Environment Rendering
    # -------------------
    demo_env = gym.make(env_name, render_mode="human")
    obs, info = demo_env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _, _, _ = ac_net.get_action_and_value(obs)
        obs, reward, terminated, truncated, info = demo_env.step(action)
        demo_env.render()
        time.sleep(0.02)  # Slow down rendering for viewing

    demo_env.close()
    print("Final 10 episode rewards:", reward_history[-10:])

if __name__ == "__main__":
    main()
