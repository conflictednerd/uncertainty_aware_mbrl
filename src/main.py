import logging
import random
import time
import uuid

import hydra
import numpy as np
import torch

# from .sac import SAC
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # Initializations
    run_name = (
        f"{cfg.policy.cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    )
    cfg.policy.cfg.run_name = run_name

    # Logging
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg.policy).items()])),
    )
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_id = str(uuid.uuid4())
        tags = [cfg.policy.env_id] + cfg.wandb.tags
        wandb_logger = wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=run_name,
            id=wandb_id,
            tags=tags,
            resume="allow",  # allow resuming from same run ID
        )

    # Testing SAC
    # instantiate uses _target_ class defined in `sac_default.yaml` and passes arguments defined there to its __init__.
    sac_policy = instantiate(
        cfg.policy, seed=cfg.seed, writer=writer, use_wandb=cfg.wandb.enable
    )
    sac_policy.update(cfg.policy.cfg.total_timesteps)
    sac_policy.envs.close()


if __name__ == "__main__":
    main()
