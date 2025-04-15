import matplotlib.pyplot as plt
import numpy as np
from Advantage_Actor_Critic import train_adv_actor_critic

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def average_experiments(train_func, num_experiments, **kwargs):
    """
    Repeat the experiment num_experiments times and compute the average of the returned steps_history and reward_history
    (averaged per episode).
    """
    all_steps = []
    all_rewards = []
    for i in range(num_experiments):
        print(f"Experiment {i+1}/{num_experiments} for entropy_coef={kwargs.get('entropy_coef')}")
        steps, rewards = train_func(**kwargs)
        all_steps.append(steps)
        all_rewards.append(rewards)
    all_steps_arr = np.array(all_steps)      # shape: (num_experiments, num_episodes)
    all_rewards_arr = np.array(all_rewards)  # shape: (num_experiments, num_episodes)
    mean_steps = all_steps_arr.mean(axis=0)
    mean_rewards = all_rewards_arr.mean(axis=0)
    return mean_steps, mean_rewards

def main():
    # Global parameters
    num_episodes = 2000         # Number of episodes per training
    learning_rate = 1e-3        # Learning rate
    gamma = 0.99                # Discount factor
    hidden_dim = 128            # Number of neurons in the hidden layer
    env_name = "CartPole-v1"    # Training environment
    num_experiments = 5         # Number of repeated experiments for each entropy_coef
    window_size = 10            # Smoothing window size (used for smoothing the plot)

    # Settings for different entropy coefficients
    entropy_coefs = [0.005, 0.01, 0.02, 0.05]

    results = {}  

    for coef in entropy_coefs:
        print(f"\nStarting experiments with entropy_coef = {coef}")
        mean_steps, mean_rewards = average_experiments(
            train_func=train_adv_actor_critic,
            num_experiments=num_experiments,
            env_name=env_name,
            num_episodes=num_episodes,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            entropy_coef=coef
        )
        results[coef] = (mean_steps, mean_rewards)

    plt.figure(figsize=(10, 6))
    for coef, (steps, rewards) in results.items():
        smoothed_rewards = moving_average(rewards, window_size)
        steps_smoothed = steps[window_size - 1:]  
        plt.plot(steps_smoothed, smoothed_rewards, linewidth=2, label=f"entropy_coef={coef}")

    plt.xlabel("Cumulative Environment Steps")
    plt.ylabel("Reward")
    plt.title(f"Advantage Actor-Critic: Effect of Entropy Coefficient\n(Averaged over {num_experiments} runs)")
    plt.legend()
    plt.grid(True)
    plt.savefig("entropy_coef_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
