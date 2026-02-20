import argparse
import logging
import os
import os.path as osp

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

from .trainer import Trainer
from the_well.benchmark.trainer.utils import set_master_config
from flowers.utils import configure_experiment
from flowers.data import WellDataModule


def get_distrib_config():
    distrib_env_variables = ["SLURM_PROCID", "SLURM_LOCALID", "SLURM_STEP_NUM_TASKS"]
    if not (set(distrib_env_variables) <= set(os.environ)):
        is_distributed = False
        rank = 0
        local_rank = 0
        world_size = 1
        logger.debug("Slurm configuration not detected in the environment")
    else:
        is_distributed = True
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_STEP_NUM_TASKS"])
        logger.debug(
            f"Slurm configuration detected, rank {rank}({local_rank})/{world_size}"
        )
    return is_distributed, world_size, rank, local_rank


class MSELossWell(nn.Module):
    """MSE Loss wrapper that accepts metadata argument (required by The Well trainer)."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # No reduction - return per-element loss

    def forward(self, pred, target, metadata=None):
        """
        Forward pass - metadata is ignored but required by The Well trainer.

        Returns per-sample, per-timestep, per-field losses.
        Expected shape: (batch, time, ..., channels) -> (batch, time, channels)
        """
        # Compute MSE without reduction
        loss = self.mse(pred, target)

        # Average over spatial dimensions but keep batch, time, and channel dimensions
        # Input shapes: (batch, time, height, width, channels) from formatter
        # We want output: (batch, time, channels)
        spatial_dims = tuple(range(2, loss.ndim - 1))  # All dims except batch, time, and channels
        if len(spatial_dims) > 0:
            loss = loss.mean(dim=spatial_dims)

        return loss

logger = logging.getLogger("the_well")
logger.setLevel(level=logging.DEBUG)

# Add console handler if not already present (for modular config mode)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def load_modular_config(data_config: str, model_config: str, train_config: str) -> DictConfig:
    """Load and merge separate data, model, and train configs."""
    logger.info(f"Loading modular configs:")
    logger.info(f"  Data: {data_config}")
    logger.info(f"  Model: {model_config}")
    logger.info(f"  Train: {train_config}")

    # Load each config
    data_cfg = OmegaConf.load(data_config)
    model_cfg = OmegaConf.load(model_config)
    train_cfg = OmegaConf.load(train_config)

    # Merge them together: data (base) < train (defaults) < model (highest priority)
    cfg = OmegaConf.merge(data_cfg, train_cfg, model_cfg)

    return cfg


def train(
    cfg: DictConfig,
    experiment_folder: str,
    checkpoint_folder: str,
    artifact_folder: str,
    viz_folder: str,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 1,
    local_rank: int = 1,
):
    """Instantiate the different objects required for training and run the training loop."""
    validation_mode = cfg.validation_mode

    # Determine batch size from model config's batch_size_map (using dataset name)
    dataset_name = cfg.data.well_dataset_name
    if hasattr(cfg, 'batch_size_map') and cfg.batch_size_map is not None:
        if dataset_name in cfg.batch_size_map:
            batch_size = cfg.batch_size_map[dataset_name]
        else:
            msg = (
                f"Dataset '{dataset_name}' not found in batch_size_map. "
                f"Available datasets: {list(cfg.batch_size_map.keys())}"
            )
            raise ValueError(msg)
        logger.info(f"Auto-selected batch size {batch_size} for dataset '{dataset_name}'")
    elif 'batch_size' in cfg.data:
        # If batch_size is explicitly set in data config, use it
        batch_size = cfg.data.batch_size
        logger.info(f"Using batch size {batch_size} from data config")
    else:
        raise ValueError(
            "No batch_size_map in model config and no batch_size in data config. "
            "Please add batch_size_map to your model config or batch_size to your data config."
        )

    # Instantiate the datamodule with the correct batch size
    logger.info(f"Instantiate datamodule {cfg.data._target_}")
    datamodule: WellDataModule = instantiate(
        cfg.data, world_size=world_size, rank=rank, data_workers=cfg.data_workers, batch_size=batch_size
    )

    # Get metadata from the instantiated datamodule
    dset_metadata = datamodule.train_dataset.metadata

    # Calculate input/output dimensions based on dataset metadata and timestep history
    # Following The Well benchmark: 4 timestep input, stacked as channels
    # Note: constant_scalars are passed as separate meta tensor (B, N), not as field channels
    n_input_fields = (
        cfg.data.n_steps_input * dset_metadata.n_fields
        + dset_metadata.n_constant_fields
    )
    n_output_fields = dset_metadata.n_fields

    # Number of meta features = number of scalars specified in meta_scalars config
    # If meta_scalars is empty or not specified, no meta conditioning
    meta_scalars_list = cfg.data.get("meta_scalars", []) or []
    n_meta = len(meta_scalars_list)

    logger.info(
        f"Dataset: {cfg.data.well_dataset_name}\n"
        f"  Spatial dims: {dset_metadata.n_spatial_dims}\n"
        f"  Resolution: {dset_metadata.spatial_resolution}\n"
        f"  Fields: {dset_metadata.n_fields}\n"
        f"  Constant fields: {dset_metadata.n_constant_fields}\n"
        f"  Meta features (constant scalars): {n_meta}\n"
        f"  Input timesteps: {cfg.data.n_steps_input}\n"
        f"  Total input channels: {n_input_fields}\n"
        f"  Output channels: {n_output_fields}"
    )

    logger.info(f"Instantiate model {cfg.model._target_}")
    model: torch.nn.Module = instantiate(
        cfg.model,
        n_spatial_dims=dset_metadata.n_spatial_dims,
        spatial_resolution=dset_metadata.spatial_resolution,
        dim_in=n_input_fields,
        dim_out=n_output_fields,
        dim_meta=n_meta,
        boundary_condition_types=dset_metadata.boundary_condition_types
    )
    summary(model, depth=5)

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda")
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model.to(device)

    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    if not validation_mode:
        optimizer: torch.optim.Optimizer = instantiate(
            cfg.optimizer, params=model.parameters()
        )
    else:
        optimizer = None

    if hasattr(cfg, "lr_scheduler") and not validation_mode:
        # Instantiate LR scheduler
        logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            max_epochs=cfg.trainer.epochs,
            warmup_start_lr=cfg.optimizer.lr * 0.1,
            eta_min=cfg.optimizer.lr * 0.1,
        )
    else:
        logger.info("No learning rate scheduler")
        lr_scheduler = None

    # Print final config, but also log it to experiment directory.
    logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Instantiate trainer {cfg.trainer._target_}")
    trainer: Trainer = instantiate(
        cfg.trainer,
        checkpoint_folder=checkpoint_folder,
        artifact_folder=artifact_folder,
        viz_folder=viz_folder,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        is_distributed=is_distributed,
        rank=rank,
    )

    if validation_mode:
        logger.info("Running in validation mode only")
        trainer.validate_best()
    else:
        # Save config to directory folder
        with open(osp.join(experiment_folder, "extended_config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

        # Train the model
        logger.info("Starting training...")
        trainer.train()

        logger.info("=" * 80)
        logger.info("Training complete!")
        logger.info("=" * 80)


def run_main(cfg: DictConfig):

    # Torch optimization settings
    torch.backends.cudnn.benchmark = (
        True  # If input size is fixed, this will usually make the computation faster
    )
    torch.set_float32_matmul_precision("high")  # Use TF32 when supported

    # Configure experiment directories and logging
    (
        cfg,
        experiment_name,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
    ) = configure_experiment(cfg, logger)

    logger.info(f"Run experiment {experiment_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Retrieve multiple processes context to setup DDP FIRST (before wandb)
    is_distributed, world_size, rank, local_rank = get_distrib_config()

    logger.info(f"Distributed training: {is_distributed} (rank {rank}/{world_size})")
    if is_distributed:
        set_master_config()
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )

    # Initiate wandb logging ONLY on rank 0
    if rank == 0:
        wandb_logged_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_logged_cfg["experiment_folder"] = experiment_folder

        # Load training wandb run ID if it exists
        training_run_id = None
        wandb_id_file = osp.join(experiment_folder, "wandb_run_id.txt")
        if osp.exists(wandb_id_file):
            with open(wandb_id_file, "r") as f:
                training_run_id = f.read().strip()

        if cfg.validation_mode and training_run_id:
            # Create a separate linked validation run
            logger.info(f"Creating validation run linked to training run {training_run_id}")
            wandb_logged_cfg["training_run_id"] = training_run_id
            wandb.init(
                dir=experiment_folder,
                project=cfg.wandb_project_name,
                group=f"{cfg.data.well_dataset_name}",
                config=wandb_logged_cfg,
                name=f"{experiment_name}_validation",
                tags=["validation", training_run_id],
            )
        elif cfg.auto_resume and training_run_id:
            # Resume existing training run
            logger.info(f"Resuming training run with ID: {training_run_id}")
            wandb.init(
                dir=experiment_folder,
                project=cfg.wandb_project_name,
                group=f"{cfg.data.well_dataset_name}",
                config=wandb_logged_cfg,
                name=experiment_name,
                id=training_run_id,
                resume="allow",
            )
        else:
            # Create new training run
            logger.info("Creating new training run")
            wandb.init(
                dir=experiment_folder,
                project=cfg.wandb_project_name,
                group=f"{cfg.data.well_dataset_name}",
                config=wandb_logged_cfg,
                name=experiment_name,
            )
            # Save run ID for future resuming/linking
            with open(wandb_id_file, "w") as f:
                f.write(wandb.run.id)
            logger.info(f"Saved wandb run ID: {wandb.run.id}")

    train(
        cfg,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
        is_distributed,
        world_size,
        rank,
        local_rank,
    )

    if rank == 0:
        wandb.finish()

if __name__ == "__main__":
    # Parse command line args BEFORE Hydra to check for modular config mode
    parser = argparse.ArgumentParser(description="Train models on The Well benchmark", add_help=False)
    parser.add_argument("--data", type=str, help="Path to data config (e.g., configs/data/viscoelastic_instability.yaml)")
    parser.add_argument("--model", type=str, help="Path to model config (e.g., configs/models/fionet_patched.yaml)")
    parser.add_argument("--train", type=str, help="Path to train config (e.g., configs/train.yaml)")

    args, unknown = parser.parse_known_args()

    cfg = load_modular_config(args.data, args.model, args.train)

    # Merge CLI overrides (e.g., validation_mode=true auto_resume=true)
    if unknown:
        cli_overrides = OmegaConf.from_cli(unknown)
        cfg = OmegaConf.merge(cfg, cli_overrides)

    run_main(cfg)
