"""Double DQN; a time-limit truncation retains the bootstrap term."""
from dataclasses import dataclass
import random
import numpy as np
import torch
from torch import nn


class QNet(nn.Module):
    def __init__(self, obs_dim=8, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim,256),nn.ReLU(),nn.Linear(256,256),
                                 nn.ReLU(),nn.Linear(256,n_actions))

    def forward(self, x):
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool


class ReplayBuffer:
    def __init__(self, capacity=200_000, seed=1):
        self.items = []
        self.capacity = capacity
        self.position = 0
        self.rng = random.Random(seed)

    def push(self, transition):
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.position] = transition
        self.position = (self.position+1) % self.capacity

    def sample(self, size):
        batch = self.rng.sample(self.items,size)
        return (np.stack([b.state for b in batch]),np.array([b.action for b in batch]),
                np.array([b.reward for b in batch],dtype=np.float32),
                np.stack([b.next_state for b in batch]),
                np.array([b.terminated for b in batch],dtype=np.float32))

    def __len__(self):
        return len(self.items)


def double_dqn_target(rewards, terminated, online, target, gamma=.99):
    with torch.no_grad():
        actions = online.argmax(dim=1, keepdim=True)
        next_q = target.gather(1,actions).squeeze(1)
        return rewards + (1-terminated)*gamma*next_q


def load_policy(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    weights = payload.get("state_dict",payload)
    policy = QNet().to(device)
    policy.load_state_dict(weights)
    policy.eval()
    return policy
