import logging
import random
import time
import uuid

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as ALL_ENVS
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


@hydra.main(version_base=None, config_path="../config", config_name="config")
def dream(cfg: DictConfig):
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
        % (
            "\n".join(
                [f"|policy_{key}|{value}|" for key, value in vars(cfg.policy).items()]
            )
        ),
    )
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [f"|wm_{key}|{value}|" for key, value in vars(cfg.world_model).items()]
            )
        ),
    )

    wm = instantiate(cfg.world_model, writer=writer)
    wm.set_env(ALL_ENVS[cfg.policy.cfg.env_id](seed=0, **cfg.policy.cfg.env_kwargs))
    policy = instantiate(
        cfg.policy, seed=cfg.seed, writer=writer, use_wandb=cfg.wandb.enable
    )
    all_ds = policy.collect_dataset(30000, do_random=True)

    wm.train(all_ds)

    for itr in range(cfg.iterations):
        # Collect experience with policy
        ds = policy.collect_dataset(10000)  # (cfg.wm_ds_size)
        all_ds.extend(ds)

        # Train world model on the collected data
        wm.train(all_ds)

        # Train policy using world model
        policy.update(
            25000,
            # cfg.policy.cfg.total_timesteps // cfg.iterations,
            wm=wm,
            from_scratch=itr == 0,
        )

        # Evaluate policy in real env.
        policy._evaluate(logging_step=policy.global_step)

    policy.envs.close()


if __name__ == "__main__":
    # main()
    dream()
