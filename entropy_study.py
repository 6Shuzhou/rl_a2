# entropy_study.py
import matplotlib.pyplot as plt
import numpy as np
from Advantage_Actor_Critic import train_adv_actor_critic

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def average_experiments(train_func, num_experiments, **kwargs):
    """
    重复实验 num_experiments 次，计算返回的 steps_history 和 reward_history 的平均值（按回合求均值）。
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
    # 全局参数
    num_episodes = 2000         # 每次训练的回合数
    learning_rate = 1e-3        # 学习率
    gamma = 0.99                # 折扣因子
    hidden_dim = 128            # 隐层神经元数
    env_name = "CartPole-v1"    # 训练环境
    num_experiments = 5        # 每个 entropy_coef 重复实验次数
    window_size = 50            # 平滑窗口大小（用于绘图时的平滑处理）

    # 不同的熵系数设置
    entropy_coefs = [ 0.005, 0.01, 0.02, 0.05]

    results = {}  # 用于保存不同 entropy_coef 下的平均实验结果

    # 针对每个 entropy_coef 进行实验
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

    # 绘制学习曲线对比图（平滑后的数据）
    plt.figure(figsize=(10, 6))
    for coef, (steps, rewards) in results.items():
        smoothed_rewards = moving_average(rewards, window_size)
        steps_smoothed = steps[window_size - 1:]  # 对应的 x 轴数据
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
