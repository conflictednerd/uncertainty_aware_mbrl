import logging
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from .nets import Actor, SoftQNetwork
from .replay_buffer import ReplayBuffer
from .utils import make_env
from .wm import BaseWorldModel

logger = logging.getLogger(__name__)


class Policy(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, timesteps: int, from_scratch: bool) -> None:
        pass


class SAC(Policy):
    def __init__(self, cfg, seed, writer, use_wandb=False) -> None:
        self.cfg = cfg
        self.writer = writer
        self.wandb = use_wandb
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"
        )
        self.global_step = 0

        # Env setup
        self.envs = gym.vector.AsyncVectorEnv(
            [
                make_env(
                    cfg.env_id,
                    cfg.env_kwargs,
                    cfg.success_wrapper,
                    seed + i,
                    i,
                    cfg.capture_video and (i == 0),
                    cfg.run_name,
                )
                for i in range(cfg.num_envs)
            ]
        )
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        # Networks
        self.actor = Actor(self.envs).to(self.device)
        self.qf1 = SoftQNetwork(self.envs).to(self.device)
        self.qf2 = SoftQNetwork(self.envs).to(self.device)
        self.qf1_target = SoftQNetwork(self.envs).to(self.device)
        self.qf2_target = SoftQNetwork(self.envs).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Automatic entropy tuning
        if cfg.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.envs.single_action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = cfg.alpha

        self._init_optimizers()

        # Replay buffer
        self.envs.single_observation_space.dtype = np.float32
        self.rb = ReplayBuffer(
            cfg.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            n_envs=1,
            handle_timeout_termination=False,
        )

        self.obs, _ = self.envs.reset(seed=seed)
        self.autoreset = np.zeros(self.envs.num_envs)

    def _init_optimizers(self):
        """Initializes actor and critic's optimizers"""
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.cfg.q_lr
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.cfg.policy_lr
        )
        if self.cfg.autotune:
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.cfg.q_lr)

    def update(
        self,
        timesteps: int,
        from_scratch: bool = False,
        wm: BaseWorldModel = None,
    ):
        """
        Args:
            timesteps: Total number of timesteps (env interactions) to train for.
            from_scratch: If True, will initialize the buffer with random actions.
            wm: If not None, will use it to generate training data.
        """
        start_time = time.time()
        if wm is not None:
            wm.reset()
            wm.nn.eval()
        for step in range(0, timesteps, self.envs.num_envs):
            if step < self.cfg.learning_starts and from_scratch:
                actions = np.array(
                    [
                        self.envs.single_action_space.sample()
                        for _ in range(self.envs.num_envs)
                    ]
                )
            else:
                actions, _, _ = self.actor.get_action(
                    torch.Tensor(self.obs).to(self.device)
                )
                actions = actions.detach().cpu().numpy()

            if wm:
                next_obs, rewards, terminations, truncations, infos = wm.step(
                    self.obs, actions
                )
            else:
                next_obs, rewards, terminations, truncations, infos = self.envs.step(
                    actions
                )
            for j in range(self.envs.num_envs):
                if not self.autoreset[j]:
                    self.rb.add(
                        self.obs[j],
                        next_obs[j],
                        actions[j],
                        rewards[j],
                        terminations[j],
                        None,
                    )

            self.obs = next_obs
            self.autoreset = np.logical_or(terminations, truncations)

            if np.any(self.autoreset):
                for j in range(self.envs.num_envs):
                    if self.autoreset[j]:
                        logger.info(
                            f"step: {self.global_step + step}, episodic_return={infos['episode']['r'][j]}"
                        )
                        self._log(
                            "SAC/episodic_return",
                            float(infos["episode"]["r"][j]),
                            self.global_step + step,
                        )
                        self._log(
                            "SAC/episodic_length",
                            float(infos["episode"]["l"][j]),
                            self.global_step + step,
                        )

            if step > self.cfg.learning_starts:
                # Accounting for multiple env steps in parallel
                for _ in range(1):
                    data = self.rb.sample(self.cfg.batch_size)
                    # Update critics
                    with torch.no_grad():
                        rewards = data.rewards
                        next_state_actions, next_state_log_pi, _ = (
                            self.actor.get_action(data.next_observations)
                        )
                        qf1_next_target = self.qf1_target(
                            data.next_observations, next_state_actions
                        )
                        qf2_next_target = self.qf2_target(
                            data.next_observations, next_state_actions
                        )
                        min_qf_next_target = (
                            torch.min(qf1_next_target, qf2_next_target)
                            - self.alpha * next_state_log_pi
                        )
                        next_q_value = rewards.flatten() + (
                            1 - data.dones.flatten()
                        ) * self.cfg.gamma * (min_qf_next_target).view(-1)

                    qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                    qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    self.q_optimizer.zero_grad()
                    qf_loss.backward()
                    self.q_optimizer.step()

                    # Update actor
                    if step % self.cfg.policy_frequency == 0:
                        for _ in range(self.cfg.policy_frequency):
                            pi, log_pi, _ = self.actor.get_action(data.observations)
                            qf1_pi = self.qf1(data.observations, pi)
                            qf2_pi = self.qf2(data.observations, pi)
                            min_qf_pi = torch.min(qf1_pi, qf2_pi)
                            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                            self.actor_optimizer.zero_grad()
                            actor_loss.backward()
                            self.actor_optimizer.step()

                            if self.cfg.autotune:
                                with torch.no_grad():
                                    _, log_pi, _ = self.actor.get_action(
                                        data.observations
                                    )
                                alpha_loss = (
                                    -self.log_alpha.exp()
                                    * (log_pi + self.target_entropy)
                                ).mean()

                                self.a_optimizer.zero_grad()
                                alpha_loss.backward()
                                self.a_optimizer.step()
                                self.alpha = self.log_alpha.exp().item()

                    # update target networks
                    if step % self.cfg.target_network_frequency == 0:
                        for param, target_param in zip(
                            self.qf1.parameters(), self.qf1_target.parameters()
                        ):
                            target_param.data.copy_(
                                self.cfg.tau * param.data
                                + (1 - self.cfg.tau) * target_param.data
                            )
                        for param, target_param in zip(
                            self.qf2.parameters(), self.qf2_target.parameters()
                        ):
                            target_param.data.copy_(
                                self.cfg.tau * param.data
                                + (1 - self.cfg.tau) * target_param.data
                            )

                if step % 200 == 0:
                    logging_step = self.global_step + step
                    self._log(
                        "SAC/qf1_values",
                        qf1_a_values.mean().item(),
                        logging_step,
                    )
                    self._log(
                        "SAC/qf1_values",
                        qf1_a_values.mean().item(),
                        logging_step,
                    )
                    self._log(
                        "SAC/qf2_values",
                        qf2_a_values.mean().item(),
                        logging_step,
                    )
                    self._log(
                        "SAC/qf1_loss",
                        qf1_loss.item(),
                        logging_step,
                    )
                    self._log(
                        "SAC/qf2_loss",
                        qf2_loss.item(),
                        logging_step,
                    )
                    self._log(
                        "SAC/qf_loss",
                        qf_loss.item() / 2.0,
                        logging_step,
                    )
                    self._log(
                        "SAC/actor_loss",
                        actor_loss.item(),
                        logging_step,
                    )
                    self._log(
                        "SAC/alpha",
                        self.alpha,
                        logging_step,
                    )
                    self._log(
                        "SAC/SPS",
                        int(step / (time.time() - start_time)),
                        logging_step,
                    )
                    if self.cfg.autotune:
                        self._log(
                            "SAC/alpha_loss",
                            alpha_loss.item(),
                            logging_step,
                        )
                if wm is None and step % 100000 == 0:
                    self._evaluate(logging_step=self.global_step + step)
        self.global_step += timesteps

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        is_batched = obs.ndim > 1
        if not is_batched:
            obs = obs.unsqueeze(0)
        action, log_prob, mean = self.actor.get_action(obs)
        if not is_batched:
            action = action.squeeze(0)
        return action.cpu().numpy()

    def _log(self, name, value, step):
        self.writer.add_scalar(name, value, step)
        if self.wandb:
            wandb.log({name: value}, step=step)

    @torch.no_grad()
    def _evaluate(self, logging_step=None):
        seed = 232323
        n_evals = 50  # Number of evaluation episodes
        envs = gym.vector.AsyncVectorEnv(
            [
                make_env(
                    self.cfg.env_id,
                    self.cfg.env_kwargs,
                    self.cfg.success_wrapper,
                    seed + i,
                    i,
                    False,
                    None,
                )
                for i in range(8)
            ]
        )

        episode_count = 0
        epi_lens, epi_rews, success = [], [], []
        obs, infos = envs.reset()
        while episode_count < n_evals:
            actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
            actions = actions.detach().cpu().numpy()

            obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            autoreset = np.logical_or(terminations, truncations)

            if np.any(autoreset):
                for j in range(8):
                    if autoreset[j]:
                        epi_rews.append(float(infos["episode"]["r"][j]))
                        epi_lens.append(float(infos["episode"]["l"][j]))
                        success.append(
                            bool(infos["success"][j])
                            and bool(infos["grasp_success"][j])
                        )
                        episode_count += 1

            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             epi_rews.append(float(info["episode"]["r"]))
            #             epi_lens.append(float(info["episode"]["l"]))
            #             assert "success" in info, f"info.keys={list(info.keys())}"
            #             success.append(
            #                 bool(info["success"]) and bool(info["grasp_success"])
            #             )
            #             episode_count += 1

        logger.info("===== Evaluation =====")
        logger.info(f"n episodes = {len(epi_lens)}")
        logger.info(
            f"episode length = {float(np.mean(epi_lens)):.2f} +- {float(np.std(epi_lens)):.2f}"
        )
        logger.info(
            f"episode reward = {float(np.mean(epi_rews)):.2f} +- {float(np.std(epi_rews)):.2f}"
        )
        logger.info(
            f"success rate = {float(np.mean(success)):.3f} +- {float(np.std(success)):.3f}"
        )

        if logging_step is not None:
            self._log(
                "SAC/Evaluation/episode length",
                float(np.mean(epi_lens)),
                logging_step,
            )
            self._log(
                "SAC/Evaluation/episode reward",
                float(np.mean(epi_rews)),
                logging_step,
            )
            self._log(
                "SAC/Evaluation/success rate",
                float(np.mean(success)),
                logging_step,
            )

    @torch.no_grad()
    def collect_dataset(
        self,
        size,
        do_random=False,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]:
        """
        Returns a dataset of transitions using the policy
        """
        seed = 232323
        envs = gym.vector.AsyncVectorEnv(
            [
                make_env(
                    self.cfg.env_id,
                    self.cfg.env_kwargs,
                    self.cfg.success_wrapper,
                    seed + i,
                    i,
                    False,
                    None,
                )
                for i in range(8)
            ]
        )

        ds = []
        obs, infos = envs.reset()
        autoreset = np.zeros(8)
        while len(ds) < size:
            if do_random:
                actions = envs.action_space.sample()
            else:
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                actions = actions.detach().cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = self.envs.step(
                actions
            )
            for j in range(8):
                if not autoreset[j]:
                    ds.append(
                        (obs[j], actions[j], next_obs[j], rewards[j], terminations[j])
                    )

            autoreset = np.logical_or(terminations, truncations)
            obs = next_obs

        return ds[:size]
