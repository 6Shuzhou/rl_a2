import matplotlib.pyplot as plt
import numpy as np
from reinforce import train_reinforce
from Actor_Critic import train_actor_critic
from Advantage_Actor_Critic import train_adv_actor_critic

# Define the smoothing function
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def average_experiments(train_func, num_experiments, **kwargs):
    """
    Repeat experiments num_experiments times for the given training function,
    and return the element-wise average of cumulative steps and reward_history for each experiment.
    """
    all_steps = []
    all_rewards = []
    for i in range(num_experiments):
        print(f"Experiment {i+1}/{num_experiments} for {train_func.__name__} ...")
        steps, rewards = train_func(**kwargs)
        all_steps.append(steps)
        all_rewards.append(rewards)
    # Convert the results into numpy arrays with shape (num_experiments, num_episodes)
    all_steps_arr = np.array(all_steps)
    all_rewards_arr = np.array(all_rewards)
    mean_steps = all_steps_arr.mean(axis=0)
    mean_rewards = all_rewards_arr.mean(axis=0)
    return mean_steps, mean_rewards

def main():
    # Global parameters
    num_episodes = 2000         # Total number of training episodes
    learning_rate = 1e-3        # Learning rate
    gamma = 0.99                # Discount factor
    hidden_dim = 128            # Number of hidden neurons
    entropy_coef = 0.01         # Entropy coefficient for Advantage Actor-Critic
    env_name = "CartPole-v1"    # Environment name
    window_size = 50            # Smoothing window size for plotting
    num_experiments = 5         # Number of repeated experiments

    # Run experiments for each algorithm
    print("Training REINFORCE ...")
    steps_reinforce, rewards_reinforce = average_experiments(
        train_func=train_reinforce,
        num_experiments=num_experiments,
        env_name=env_name,
        num_episodes=num_episodes,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma
    )
    
    print("Training Actor-Critic ...")
    steps_actor_critic, rewards_actor_critic = average_experiments(
        train_func=train_actor_critic,
        num_experiments=num_experiments,
        env_name=env_name,
        num_episodes=num_episodes,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma
    )
    
    print("Training Advantage Actor-Critic ...")
    steps_adv_actor_critic, rewards_adv_actor_critic = average_experiments(
        train_func=train_adv_actor_critic,
        num_experiments=num_experiments,
        env_name=env_name,
        num_episodes=num_episodes,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        entropy_coef=entropy_coef
    )

    # Smooth the reward data for the three algorithms
    smoothed_rewards_reinforce = moving_average(rewards_reinforce, window_size)
    smoothed_rewards_actor_critic = moving_average(rewards_actor_critic, window_size)
    smoothed_rewards_adv_actor_critic = moving_average(rewards_adv_actor_critic, window_size)
    
    # Adjust the x-axis (cumulative steps) accordingly to match the smoothed points
    steps_reinforce_smoothed = steps_reinforce[window_size-1:]
    steps_actor_critic_smoothed = steps_actor_critic[window_size-1:]
    steps_adv_actor_critic_smoothed = steps_adv_actor_critic[window_size-1:]
    
    # Plot the raw and smoothed data
    plt.figure(figsize=(10, 5))
    # Raw data (with lower opacity)
    plt.plot(steps_reinforce, rewards_reinforce, label='REINFORCE Raw', alpha=0.3)
    plt.plot(steps_actor_critic, rewards_actor_critic, label='Actor-Critic Raw', alpha=0.3)
    plt.plot(steps_adv_actor_critic, rewards_adv_actor_critic, label='Adv Actor-Critic Raw', alpha=0.3)
    # Smoothed data (with thicker lines)
    plt.plot(steps_reinforce_smoothed, smoothed_rewards_reinforce, label='REINFORCE Smoothed', linewidth=2)
    plt.plot(steps_actor_critic_smoothed, smoothed_rewards_actor_critic, label='Actor-Critic Smoothed', linewidth=2)
    plt.plot(steps_adv_actor_critic_smoothed, smoothed_rewards_adv_actor_critic, label='Adv Actor-Critic Smoothed', linewidth=2)
    
    plt.xlabel("Cumulative Environment Steps")
    plt.ylabel("Reward")
    plt.title("Learning Curve Comparison (Averaged over {} runs)".format(num_experiments))
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
