import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets_v1 import WorldModelNetwork
torch.set_printoptions(sci_mode=False)

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
        self._test_ticks = 0

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
        #### Regression variables
        all_sp, all_pred_sp = [], []
        all_r, all_pred_r = [], []
        #### Classification variable
        all_term, all_pred_term = [], []
        self.nn=self.nn.to(self.device)
        self.nn.train()
        dataset = [(s, a, sp, r, d) for s, a, sp, r, d, *_ in dataset]
        r_values = [r for _, _, _, r, _ in dataset]
        # Convert to a PyTorch tensor for efficient computation
        r_tensor = torch.tensor(r_values, dtype=torch.float32)

        # Compute the min and max of `r`
        r_min = torch.min(r_tensor)
        r_max = torch.max(r_tensor)

        loader = DataLoader(dataset, self.batch_size, shuffle=True, num_workers=0)
        total_iterations = self.epochs * len(loader)  # Total number of steps
        progress_bar = tqdm(total=total_iterations, desc="Training Progress")

        for epoch in range(self.epochs):
            all_r, all_pred_r = [], []
            for batch_idx,batch in enumerate(loader):
                s, a, sp, r, term = batch
                s = s.float().to(self.device)
                a = a.float().to(self.device)
                sp = sp.float().to(self.device)
                r = r.float().to(self.device)
                # #####
                r = torch.log(1 + r - r_min)
                # r = (r-r_min)/(r_max-r_min)
                term = term.float().to(self.device)

                pred_sp, pred_r, pred_term = self.nn.step(s, a)

                # Accumulate predictions and ground truths
                all_sp.append(sp.detach().cpu().numpy())
                all_pred_sp.append(pred_sp.detach().cpu().numpy())
                all_r.append(r.detach().cpu().numpy())
                all_pred_r.append(pred_r.detach().cpu().numpy())
                all_term.append(term.detach().cpu().numpy())
                all_pred_term.append(torch.sigmoid(pred_term).detach().cpu().numpy())

                # Compute cumulative R-squared
                cumulative_sp = np.concatenate(all_sp, axis=0)
                cumulative_pred_sp = np.concatenate(all_pred_sp, axis=0)
                cumulative_r = np.concatenate(all_r, axis=0)
                cumulative_pred_r = np.concatenate(all_pred_r, axis=0)
                r2_state = r2_score(cumulative_sp.flatten(), cumulative_pred_sp.flatten())
                r2_reward = r2_score(cumulative_r, cumulative_pred_r)

                state_loss = F.mse_loss(pred_sp, sp)
                reward_loss = F.mse_loss(pred_r, r)
                termination_loss = F.binary_cross_entropy_with_logits(pred_term, term)

                sl_mean = 0.01 * state_loss.item() + 0.99 * sl_mean
                rl_mean = 0.01 * reward_loss.item() + 0.99 * rl_mean
                tl_mean = 0.01 * termination_loss.item() + 0.99 * tl_mean

                # Termination accuracy
                pred_term_binary = (torch.sigmoid(pred_term) > 0.5).float()
                batch_accuracy = accuracy_score(term.cpu().numpy(), pred_term_binary.cpu().numpy())

                # Update tqdm progress bar
                progress_bar.update(1)  # Increment progress bar
                progress_bar.set_postfix({
                    "Epoch": f"{epoch + 1}/{self.epochs}"
                })

                self.writer.add_scalar("WM/sl_mean", sl_mean, self._train_ticks)
                self.writer.add_scalar("WM/rl_mean", rl_mean, self._train_ticks)
                self.writer.add_scalar("WM/tl_mean", tl_mean, self._train_ticks)
                self.writer.add_scalar("WM/r2_state_train", r2_state, self._train_ticks)
                self.writer.add_scalar("WM/r2_reward_train", r2_reward, self._train_ticks)
                self.writer.add_scalar("WM/accuracy_termination_train", batch_accuracy, self._train_ticks)

                loss = (
                    self.state_coef * state_loss
                    + self.reward_coef * reward_loss*1
                    + self.term_coef * termination_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logger.info(
                    f"state_loss: {state_loss.item():.6f}, reward_loss: {reward_loss.item():.4f}, termination_loss: {termination_loss.item():.4f}, r2_state: {r2_state:.4f},r2_reward: {r2_reward:.4f}, accuracy_termination: {batch_accuracy:.4f}"
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

        progress_bar.close()
        #### For confusion matrix
        all_term = np.concatenate(all_term, axis=0)
        all_pred_term = np.concatenate(all_pred_term, axis=0)
        # Binary predictions for termination
        all_pred_term_binary = (all_pred_term > 0.5).astype(float)

        cumulative_accuracy = accuracy_score(all_term, all_pred_term_binary)
        conf_matrix = confusion_matrix(all_term, all_pred_term_binary)
        self.writer.add_scalar("WM/culmulative_accuracy_train", cumulative_accuracy, self._train_ticks)
        # Plot confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        self.writer.add_figure("Confusion Matrix", plt.gcf(), self._train_ticks)
        plt.close()

    def test(self, dataset):

        sl_mean, rl_mean, tl_mean = 0.0, 0.0, 0.0
        #### Regression variables
        all_sp, all_pred_sp = [], []
        all_r, all_pred_r = [], []
        #### Classification variable
        all_term, all_pred_term = [], []

        self.nn.eval()
        dataset = [(s, a, sp, r, d) for s, a, sp, r, d, *_ in dataset]
        loader = DataLoader(dataset, self.batch_size, shuffle=True, num_workers=0)
        with torch.no_grad():
            ### batch size=64
            for batch in loader:
                s, a, sp, r, term = batch
                s = s.float().to(self.device)
                a = a.float().to(self.device)
                sp = sp.float().to(self.device)
                r = r.float().to(self.device)
                term = term.float().to(self.device)

                pred_sp, pred_r, pred_term = self.nn.step(s, a)

                # Accumulate predictions and ground truths
                all_sp.append(sp.cpu().numpy())
                all_pred_sp.append(pred_sp.cpu().numpy())
                all_r.append(r.cpu().numpy())
                all_pred_r.append(pred_r.cpu().numpy())
                all_term.append(term.cpu().numpy())
                all_pred_term.append(torch.sigmoid(pred_term).cpu().numpy())
                # Compute cumulative R-squared
                cumulative_sp = np.concatenate(all_sp, axis=0)
                cumulative_pred_sp = np.concatenate(all_pred_sp, axis=0)
                cumulative_r = np.concatenate(all_r, axis=0)
                cumulative_pred_r = np.concatenate(all_pred_r, axis=0)
                r2_state = r2_score(cumulative_sp.flatten(), cumulative_pred_sp.flatten())
                r2_reward = r2_score(cumulative_r, cumulative_pred_r)


                ### Loss for all of them
                state_loss = F.mse_loss(pred_sp, sp)
                reward_loss = F.mse_loss(pred_r, r)
                termination_loss = F.binary_cross_entropy_with_logits(pred_term, term)

                sl_mean = 0.01 * state_loss.item() + 0.99 * sl_mean
                rl_mean = 0.01 * reward_loss.item() + 0.99 * rl_mean
                tl_mean = 0.01 * termination_loss.item() + 0.99 * tl_mean

                # Termination accuracy
                pred_term_binary = (torch.sigmoid(pred_term) > 0.5).float()
                batch_accuracy = accuracy_score(term.cpu().numpy(), pred_term_binary.cpu().numpy())

                self.writer.add_scalar("WM/sl_mean_test", sl_mean, self._test_ticks)
                self.writer.add_scalar("WM/rl_mean_test", rl_mean, self._test_ticks)
                self.writer.add_scalar("WM/tl_mean_test", tl_mean, self._test_ticks)
                self.writer.add_scalar("WM/r2_state_test", r2_state, self._test_ticks)
                self.writer.add_scalar("WM/r2_reward_test", r2_reward, self._test_ticks)
                self.writer.add_scalar("WM/accuracy_termination_test", batch_accuracy, self._test_ticks)

                loss = (
                        self.state_coef * state_loss
                        + self.reward_coef * reward_loss
                        + self.term_coef * termination_loss
                )


                logger.info(
                    f"state_loss: {state_loss.item():.6f}, reward_loss: {reward_loss.item():.4f},\
                     termination_loss: {termination_loss.item():.4f}, r2_state: {r2_state:.4f},\
                      r2_reward: {r2_reward:.4f}, accuracy_termination: {batch_accuracy:.4f}"
                )
                self.writer.add_scalar(
                    "WM/state_loss_test", state_loss.item(), self._test_ticks
                )
                self.writer.add_scalar(
                    "WM/reward_loss_test", reward_loss.item(), self._test_ticks
                )
                self.writer.add_scalar(
                    "WM/termination_loss_test", termination_loss.item(), self._test_ticks
                )
                self.writer.add_scalar("WM/loss_test", loss.item(), self._test_ticks)
                self._test_ticks += 1

        #### For confusion matrix
        all_term = np.concatenate(all_term, axis=0)
        all_pred_term = np.concatenate(all_pred_term, axis=0)
        # Binary predictions for termination
        all_pred_term_binary = (all_pred_term > 0.5).astype(float)

        cumulative_accuracy = accuracy_score(all_term, all_pred_term_binary)
        conf_matrix = confusion_matrix(all_term, all_pred_term_binary)
        self.writer.add_scalar("WM/culmulative_accuracy_test", cumulative_accuracy, self._test_ticks)
        # Plot confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        self.writer.add_figure("Confusion Matrix", plt.gcf(), self._test_ticks)
        plt.close()

    def save_wm(self):
        torch.save(self.nn.state_dict(), f"runs/{self.run_name}/wm.pt")

    def load_wm(self,run_name):
        self.nn.load_state_dict(
            torch.load(
                f"runs/{run_name}/wm.pt",
                map_location=self.device,
                weights_only=True,
            )
        )