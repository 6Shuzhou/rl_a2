import matplotlib.pyplot as plt
import numpy as np
from reinforce import train_reinforce
from Actor_Critic import train_actor_critic
from Advantage_Actor_Critic import train_adv_actor_critic

# Define the smoothing function
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    # Define global parameters here
    num_episodes = 1500         # Total training episodes
    learning_rate = 1e-3        # Learning rate
    gamma = 0.99                # Discount factor
    hidden_dim = 128            # Number of hidden neurons
    entropy_coef = 0.01         # Entropy coefficient for Advantage Actor-Critic (only applies to this algorithm)
    env_name = "CartPole-v1"    # Environment name
    window_size = 50            # Smoothing window size for plotting

    # Train three algorithms sequentially, using the same parameters
    print("Training REINFORCE ...")
    steps_reinforce, rewards_reinforce = train_reinforce(env_name=env_name, num_episodes=num_episodes,
                                                         hidden_dim=hidden_dim, learning_rate=learning_rate,
                                                         gamma=gamma)
    
    print("Training Actor-Critic ...")
    steps_actor_critic, rewards_actor_critic = train_actor_critic(env_name=env_name, num_episodes=num_episodes,
                                                                  hidden_dim=hidden_dim, learning_rate=learning_rate,
                                                                  gamma=gamma)
    
    print("Training Advantage Actor-Critic ...")
    steps_adv_actor_critic, rewards_adv_actor_critic = train_adv_actor_critic(env_name=env_name, num_episodes=num_episodes,
                                                                              hidden_dim=hidden_dim, learning_rate=learning_rate,
                                                                              gamma=gamma, entropy_coef=entropy_coef)

    # Smooth the reward data for all three algorithms
    smoothed_rewards_reinforce = moving_average(rewards_reinforce, window_size)
    smoothed_rewards_actor_critic = moving_average(rewards_actor_critic, window_size)
    smoothed_rewards_adv_actor_critic = moving_average(rewards_adv_actor_critic, window_size)
    
    steps_reinforce_smoothed = steps_reinforce[window_size-1:]
    steps_actor_critic_smoothed = steps_actor_critic[window_size-1:]
    steps_adv_actor_critic_smoothed = steps_adv_actor_critic[window_size-1:]
    
    # Plot the comparison of raw and smoothed learning curves
    plt.figure(figsize=(10, 5))
    # Raw data (with higher transparency)
    plt.plot(steps_reinforce, rewards_reinforce, label='REINFORCE Raw', alpha=0.3)
    plt.plot(steps_actor_critic, rewards_actor_critic, label='Actor-Critic Raw', alpha=0.3)
    plt.plot(steps_adv_actor_critic, rewards_adv_actor_critic, label='Adv Actor-Critic Raw', alpha=0.3)
    # Smoothed data (thicker lines)
    plt.plot(steps_reinforce_smoothed, smoothed_rewards_reinforce, label='REINFORCE Smoothed', linewidth=2)
    plt.plot(steps_actor_critic_smoothed, smoothed_rewards_actor_critic, label='Actor-Critic Smoothed', linewidth=2)
    plt.plot(steps_adv_actor_critic_smoothed, smoothed_rewards_adv_actor_critic, label='Adv Actor-Critic Smoothed', linewidth=2)
    
    plt.xlabel("Cumulative Environment Steps")
    plt.ylabel("Reward")
    plt.title("Learning Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
