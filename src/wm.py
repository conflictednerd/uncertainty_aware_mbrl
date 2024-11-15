import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .nets import WorldModelNetwork

logger = logging.getLogger(__name__)


class BaseWorldModel(ABC):
    @abstractmethod
    def step(
        self, obs: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Args:
            obs (np.ndarray): batch of observations
            action (np.ndarray): batch of actions

        Given a batch of observations and actions, return next observations, rewards, dones, info(=None)
        """
        pass

    @abstractmethod
    def train(
        self, dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]
    ) -> None:
        """
        Trains the WM neural net on a given dataset.
        Assume that the dataset is a list of tuples (s, a, s', r, d).
        """
        pass


class SimpleWorldModel(BaseWorldModel):
    def __init__(
        self,
        layers,
        obs_dim,
        action_dim,
        lr,
        state_coef,
        reward_coef,
        term_coef,
        batch_size,
        epochs,
        num_envs,
        horizon,
        cuda,
        run_name,
        writer,
    ) -> None:
        super().__init__()
        self.nn = WorldModelNetwork(layers, obs_dim, action_dim)
        self.state_coef = state_coef
        self.reward_coef = reward_coef
        self.term_coef = term_coef
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_envs = num_envs
        self.horizon = horizon
        self.run_name = run_name
        self.writer = writer
        self.step_counter = np.zeros(num_envs, dtype=np.int32)
        self._r = np.zeros(num_envs)

        self.optimizer = optim.Adam(self.nn.parameters(), lr)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cuda else "cpu"
        )

        self.nn.to(self.device)

        self._train_ticks = 0

    @torch.no_grad()
    def step(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray | Dict]:
        # At the beginning of an episode a random obs is used instead of the passed obs.
        if np.any(self.step_counter == 0):
            obs[self.step_counter == 0] = self._get_start_obs()[self.step_counter == 0]

        # np to torch
        obs = torch.from_numpy(obs).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)

        # step
        new_obs, reward, terms = self.nn.step(obs, action)

        # torch to np
        new_obs = new_obs.cpu().numpy()
        reward = reward.cpu().numpy()
        terms = terms.cpu().numpy()

        # termination and truncation
        terms = terms >= 0
        truncs = self.step_counter >= self.horizon - 1

        self.step_counter += 1
        assert self._r.shape == reward.shape, "{self._r.shape}, {reward.shape}"
        self._r += reward
        info = {
            "episode": {"r": np.copy(self._r), "l": np.copy(self.step_counter)},
            "success": terms,
        }

        # reset terminated/truncated episodes
        self.step_counter[np.logical_or(terms, truncs)] = 0
        self._r[np.logical_or(terms, truncs)] = 0

        return new_obs, reward, terms, truncs, info

    def _get_start_obs(self):
        # Get num_envs starting obs from self.env and return them
        start_obs = []
        for i in range(self.num_envs):
            start_obs.append(self.env.reset()[0])
        return np.stack(start_obs)

    def reset(self):
        # resets step_counter
        self.step_counter = np.zeros(self.num_envs)
        self._r = np.zeros(self.num_envs)

    def set_env(self, env):
        # sets self.env (only used to generate new observations on reset)
        self.env = env

    def train(self, dataset):
        sl_mean, rl_mean, tl_mean = 0.0, 0.0, 0.0

        self.nn.train()
        dataset = [(s, a, sp, r, d) for s, a, sp, r, d, *_ in dataset]
        loader = DataLoader(dataset, self.batch_size, shuffle=True, num_workers=4)
        for epoch in range(self.epochs):
            for batch in loader:
                s, a, sp, r, term = batch
                s = s.float().to(self.device)
                a = a.float().to(self.device)
                sp = sp.float().to(self.device)
                r = r.float().to(self.device)
                term = term.float().to(self.device)

                pred_sp, pred_r, pred_term = self.nn.step(s, a)

                state_loss = F.mse_loss(pred_sp, sp)
                reward_loss = F.mse_loss(pred_r, r)
                termination_loss = F.binary_cross_entropy_with_logits(pred_term, term)

                sl_mean = 0.01 * state_loss.item() + 0.99 * sl_mean
                rl_mean = 0.01 * reward_loss.item() + 0.99 * rl_mean
                tl_mean = 0.01 * termination_loss.item() + 0.99 * tl_mean
                self.writer.add_scalar("WM/sl_mean", sl_mean, self._train_ticks)
                self.writer.add_scalar("WM/rl_mean", rl_mean, self._train_ticks)
                self.writer.add_scalar("WM/tl_mean", tl_mean, self._train_ticks)

                loss = (
                    self.state_coef * state_loss
                    + self.reward_coef * reward_loss
                    + self.term_coef * termination_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                logger.info(
                    f"state_loss: {state_loss.item():.6f}, reward_loss: {reward_loss.item():.4f}, termination_loss: {termination_loss.item():.4f}"
                )
                self.writer.add_scalar(
                    "WM/state_loss", state_loss.item(), self._train_ticks
                )
                self.writer.add_scalar(
                    "WM/reward_loss", reward_loss.item(), self._train_ticks
                )
                self.writer.add_scalar(
                    "WM/termination_loss", termination_loss.item(), self._train_ticks
                )
                self.writer.add_scalar("WM/loss", loss.item(), self._train_ticks)
                self._train_ticks += 1

    def save_wm(self):
        torch.save(self.nn.state_dict(), f"runs/{self.run_name}/wm.pt")

    def load_wm(self):
        self.nn.load_state_dict(
            torch.load(
                f"runs/{self.cfg.run_name}/wm.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
