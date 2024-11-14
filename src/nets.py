import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector import VectorEnv


class SoftQNetwork(nn.Module):
    def __init__(self, envs: VectorEnv):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(envs.single_observation_space.shape).prod()
            + np.prod(envs.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device


LOG_STD_MIN, LOG_STD_MAX = -5, 2


class Actor(nn.Module):
    def __init__(self, envs: VectorEnv):
        super().__init__()
        self.fc1 = nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(envs.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(envs.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    @property
    def device(self):
        return next(self.parameters()).device


class OnlineNormalizer(nn.Module):
    """
    Performs online normalization by maintaining running statistics.
    """

    def __init__(self, num_features, momentum=0.01, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Register buffers for running statistics
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(0, keepdim=True)
            batch_var = x.var(0, keepdim=True, unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * batch_mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * batch_var
                self.num_batches_tracked += 1

            # Use batch statistics for normalization during training
            return (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Use running statistics for normalization during inference
            return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)


class WorldModelNetwork(nn.Module):
    def __init__(self, layers, obs_dim, action_dim) -> None:
        super().__init__()
        self.normalizer = OnlineNormalizer(obs_dim + action_dim)
        self.layers = nn.ModuleList(
            [
                nn.Linear(_in, _out)
                for _in, _out in zip([obs_dim + action_dim] + layers, layers)
            ]
        )
        self.obs_head = nn.Linear(layers[-1], obs_dim)
        self.reward_head = nn.Linear(layers[-1], 1)
        self.termination_head = nn.Linear(layers[-1], 1)

    def forward(self, s, a):
        latent = torch.cat((s, a), dim=-1)
        latent = self.normalizer(latent)
        for layer in self.layers:
            latent = F.relu(layer(latent))
        obs_diff = (self.obs_head(latent).sigmoid() - 0.5) * 4  # obs_diff in [-2, 2]
        reward = (self.reward_head(latent).sigmoid() - 0.5) * 20  # reward in [-10, 10]
        termination = self.termination_head(latent)
        return obs_diff, reward, termination

    def step(self, s, a):
        """
        Returns:
            obs: next observation (Bx|S|)
            reward: predicted reward (B)
            termination: logit values for termination prediction (B)
        """
        obs_diff, reward, termination = self(s, a)
        obs = s + obs_diff
        return obs, reward.flatten(), termination.flatten()
