# Reinforcement Learning Experiments: CartPole-v1

This repository contains multiple Reinforcement Learning (RL) algorithms—REINFORCE, Actor-Critic (AC), and Advantage Actor-Critic (A2C)—all tested on the [CartPole-v1 environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) from OpenAI Gym. Additionally, there is an entropy coefficient study that explores the impact of different entropy regularization factors in A2C.

## Project Structure


1. **`Reinforce.py`**  
   - Implements REINFORCE: a Monte Carlo policy gradient method without a baseline.

2. **`Actor_Critic.py`**  
   - Implements a basic Actor-Critic algorithm where the policy (Actor) and value function (Critic) share most network layers but have separate heads for policy and state-value prediction.

3. **`Advantage_Actor_Critic.py`**  
   - Implements A2C, adding an entropy term.  

4. **`train.py`**  
   - A single script to train REINFORCE, AC, and A2C in one go, comparing their performances.  
   - Each algorithm is trained for a specified number of episodes, and the learning curves are plotted against the total environment steps.  
   - Results are saved to disk in PNG plots.

5. **`entropy_coef_study.py`**  
   - Investigates different entropy regularization coefficients in the A2C algorithm.  
   - Runs the A2C agent multiple times, each time with a different coefficient (\( \beta \)), and aggregates results to visualize how entropy scaling affects performance.

## Getting Started

1. **install dependencies** :
   ```bash
   pip install -r requirements.txt
2. **All three algorithms comparison** :
   ```bash
   python train.py
3. **Entropy coefficient study** :
   ```bash
   python entropy_coef_study.py
