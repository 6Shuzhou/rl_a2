import torch
import torch.optim as optim
from networks.policy import PolicyNetwork
from utils.common import compute_returns

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def update(self, log_probs, rewards):
        returns = compute_returns(rewards, self.gamma)
        loss = (-torch.stack(log_probs) * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
