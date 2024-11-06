from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


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
    def update(
        self, dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]
    ) -> None:
        """
        Trains the WM neural net on a given dataset.
        Assume that the dataset is a list of tuples (s, a, s', r, d).
        """
        pass
