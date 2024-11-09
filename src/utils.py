import gymnasium as gym
import numpy as np
from gymnasium.core import Env
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as ALL_ENVS

# TODO:
DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
    "azimuth": 160,
    "elevation": -15.0,
    "lookat": np.array([0.0, 0.0, 0.5]),  # Use this for button push
}

DEFAULT_SIZE = 284


class CameraWrapper(gym.Wrapper):
    def __init__(self, env: Env, seed: int):
        super().__init__(env)

        self.unwrapped.model.vis.global_.offwidth = DEFAULT_SIZE
        self.unwrapped.model.vis.global_.offheight = DEFAULT_SIZE
        self.unwrapped.mujoco_renderer = MujocoRenderer(
            env.model,
            env.data,
            DEFAULT_CAMERA_CONFIG,
        )

        # Hack: enable random reset
        self.unwrapped._freeze_rand_vec = False
        self.unwrapped.seed(seed)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)


class SuccessWrapper(gym.Wrapper):
    """
    Wrapper to terminate the episode on task completion.
    """

    def __init__(self, env: Env, seed: int):
        super().__init__(env)

        # Hack: enable random reset
        # TODO: I don't know if this is needed
        self.unwrapped._freeze_rand_vec = False
        self.unwrapped.seed(seed)

    def reset(self, seed=None, options=None):
        self._h = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._h += 1
        obs, reward, term, trunc, info = self.env.step(action)
        term = term or (bool(info["success"]) and bool(info["grasp_success"]))
        trunc = trunc or (self._h >= 500)
        # TODO Add big reward on done
        # if term:
        #     reward += 100
        return obs, reward, term, trunc, info


def make_env(env_id, env_kwargs, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = ALL_ENVS[env_id](seed=seed, render_mode="rgb_array", **env_kwargs)
            env = SuccessWrapper(env, seed)
            env = CameraWrapper(env, seed)
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                disable_logger=True,
                episode_trigger=lambda n: (n % 8) == 0,
            )
        else:
            env = ALL_ENVS[env_id](seed=seed, **env_kwargs)
            env = SuccessWrapper(env, seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk
