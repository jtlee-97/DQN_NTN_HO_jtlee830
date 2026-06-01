"""DQN and Double-DQN agents for D2 threshold selection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from leohosim_py.config import DQNConfig


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool] | Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, action_dim: int | None = None):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.action_dim = action_dim

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        next_valid_masks = []
        for item in batch:
            if len(item) == 5:
                state, action, reward, next_state, done = item
                if self.action_dim is None:
                    raise ValueError("ReplayBuffer needs action_dim to backfill next_valid_mask for 5-tuple transitions")
                mask = np.ones(self.action_dim, dtype=np.bool_)
            else:
                state, action, reward, next_state, done, mask = item
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            next_valid_masks.append(mask)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            np.asarray(next_valid_masks, dtype=np.bool_),
        )

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class DQNTrainStats:
    loss: float | None = None
    epsilon: float = 0.0


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: DQNConfig,
        seed: int = 7,
        double_dqn: bool = False,
        device: str | None = None,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.cfg = cfg
        self.action_dim = action_dim
        self.double_dqn = double_dqn
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_net = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.replay = ReplayBuffer(cfg.buffer_size, action_dim=action_dim)
        self.steps = 0

    def epsilon(self, episode: int) -> float:
        frac = min(max(episode, 0) / max(self.cfg.epsilon_decay_episodes, 1), 1.0)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def select_action(self, state: np.ndarray, epsilon: float, valid_actions: np.ndarray | None = None) -> int:
        if valid_actions is None:
            valid_actions = np.arange(self.action_dim)
        else:
            valid_actions = np.asarray(valid_actions, dtype=np.int64)
            if len(valid_actions) == 0:
                valid_actions = np.arange(self.action_dim)
        if random.random() < epsilon:
            return int(random.choice(valid_actions.tolist()))
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t).squeeze(0)
            masked_q = q_values[torch.as_tensor(valid_actions, dtype=torch.long, device=self.device)]
            best_local = int(torch.argmax(masked_q).item())
            return int(valid_actions[best_local])

    def remember(self, transition: Transition) -> None:
        self.replay.push(transition)

    def train_step(self) -> float | None:
        self.steps += 1
        if len(self.replay) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        if self.steps % self.cfg.train_freq != 0:
            return None

        states, actions, rewards, next_states, dones, next_valid_masks = self.replay.sample(self.cfg.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(1)
        next_valid_masks_t = torch.as_tensor(next_valid_masks, dtype=torch.bool, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            if self.double_dqn:
                next_q_policy = self.policy_net(next_states_t).masked_fill(~next_valid_masks_t, -1e9)
                next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, next_actions)
            else:
                next_q_all = self.target_net(next_states_t).masked_fill(~next_valid_masks_t, -1e9)
                next_q = next_q_all.max(dim=1, keepdim=True).values
            target_q = rewards_t + self.cfg.gamma * next_q * (~dones_t)

        if self.cfg.loss_type == "mse":
            loss = F.mse_loss(q_values, target_q)
        else:
            loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.gradient_clip)
        self.optimizer.step()

        if self.steps % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def supervised_action_step(self, state: np.ndarray, action: int) -> float:
        """One behavior-cloning update on the Q-network logits.

        This is used only for optional teacher warmup; the same network is then
        fine-tuned by temporal-difference learning.
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor([int(action)], dtype=torch.long, device=self.device)
        q_values = self.policy_net(state_t)
        loss = F.cross_entropy(q_values, action_t)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.gradient_clip)
        self.optimizer.step()
        if self.steps % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "steps": self.steps,
                "double_dqn": self.double_dqn,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["policy_net"]))
        self.steps = int(checkpoint.get("steps", self.steps))
