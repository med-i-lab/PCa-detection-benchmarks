#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import sys
import time

import torch

import submitit
from torch import distributed as dist
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import yaml
import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader, DistributedSampler
import hydra
import logging
from torch import nn

NUM_NODES = 1
NUM_TASKS_PER_NODE = 2


NUM_CPUS_PER_TASK = 1
PARTITION = "a40,t4v2,rtx6000"
LOGS_DIR = ".slurm_logs"


from torch import nn


class VICRegTrainer:
    def __init__(self, args: DictConfig):
        self.args = args
        self.workdir = Path(args.workdir)
        self._state = {}
        self._loading_checkpoint = False

    def setup(self):
        print("Setting up experiment with args: ")
        OmegaConf.resolve(self.args)
        print(OmegaConf.to_yaml(self.args))
        # setup the distributed environment
        print("Setting up the VICReg trainer")
        print("exporting PyTorch distributed environment variables")
        dist_env = submitit.helpers.TorchDistributedEnvironment().export()
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")
        # print_env()

        # Using the (default) env:// initialization method
        dist.init_process_group(backend="nccl")

        print("Distributed environment initialized")
        print(f"{dist.get_rank()=}")
        print(f"{dist.get_world_size()=}")

        print("Testing communication with other processes: ")
        print("Creating Barrier")
        t1 = time.time()
        dist.barrier()
        t2 = time.time()
        print("barrier passed after", t2 - t1, "seconds")

        print("Setting up the working directory")
        if dist.get_rank() == 0:
            if not self.workdir.exists():
                self.workdir.mkdir(parents=True, exist_ok=True)
            if not (self.workdir / "checkpoints").exists():
                os.symlink(
                    self.args.checkpoint_dir,
                    os.path.join(self.args.workdir, "checkpoints"),
                    target_is_directory=True,
                )
            if not (self.workdir / "config.yaml").exists():
                with open(os.path.join(self.args.workdir, "config.yaml"), "w") as f:
                    print("Writing config to file")
                    yaml.dump(OmegaConf.to_container(self.args), f)

        dist.barrier()

        if (self.workdir / "checkpoints" / "last_experiment_state.pth").exists():
            print(
                f"Loading checkpoint from {self.workdir / 'checkpoints' / 'last_experiment_state.pth'}"
            )
            self._state = torch.load(
                self.workdir / "checkpoints" / "last_experiment_state.pth"
            )
            self._loading_checkpoint = True
        else:
            print("No checkpoint found. Starting from scratch")

        self.create_datasets()
        self.create_dataloaders()
        self.create_model()
        self.create_optimizer()

    def create_datasets(self):
        """Create the datasets in place"""
        from trusnet.data.exact.nct02079025.dataset.rf_datasets import (
            PatchesDataset,
            PatchViewConfig,
        )
        from trusnet.data.exact.nct02079025.dataset import (
            RF_PATCHES_MEAN,
            RF_PATCHES_STD,
        )
        from trusnet.data.exact import cohort_selection as cs

        train_patients, val_patients, test_patients = cs.get_patient_splits(
            self.args.fold, self.args.num_folds
        )
        train_cores = cs.get_cores_for_patients(train_patients)
        val_cores = cs.get_cores_for_patients(val_patients)
        test_cores = cs.get_cores_for_patients(test_patients)

        if self.args.ssl.undersample_benign_cores:
            train_cores_for_self_supervision = cs.undersample_benign(train_cores)
        else:
            train_cores_for_self_supervision = train_cores
        train_cores_for_supervision = cs.undersample_benign(train_cores)

        if self.args.get("remove_low_involvement_cores_in_evaluation"):
            val_cores = cs.remove_cores_below_threshold_involvement(val_cores, 40)
            test_cores = cs.remove_cores_below_threshold_involvement(test_cores, 40)

        self.basic_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(RF_PATCHES_MEAN, RF_PATCHES_STD),
                T.Resize((224, 224)),
            ]
        )
        self._basic_augmentation = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(RF_PATCHES_MEAN, RF_PATCHES_STD),
                T.Resize((224, 224)),
                T.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomInvert(),
            ]
        )
        patch_view_config_ssl = PatchViewConfig(
            patch_size=(5, 5),
            patch_strides=(2, 2),
            needle_region_only=self.args.ssl.needle_only,
            prostate_region_only=False,
        )
        patch_view_config_sup = PatchViewConfig(
            patch_size=(5, 5),
            patch_strides=(1, 1),
            needle_region_only=True,
            prostate_region_only=False,
        )
        print("Creating SSL training set")
        self.train_dataset_ssl = PatchesDataset(
            train_cores_for_self_supervision,
            patch_view_config_ssl,
            transform=self.ssl_augmentations,
            target_transform=lambda x: torch.tensor(x).long(),
        )
        print("Creating supervised training set")
        self.train_dataset_sup = PatchesDataset(
            train_cores_for_supervision,
            patch_view_config_sup,
            transform=self.basic_transform,
            target_transform=lambda x: torch.tensor(x).long(),
        )
        print("Creating validation set")
        self.val_dataset = PatchesDataset(
            val_cores,
            patch_view_config_sup,
            transform=self.basic_transform,
            target_transform=lambda x: torch.tensor(x).long(),
        )
        print("Creating test set")
        self.test_dataset = PatchesDataset(
            test_cores,
            patch_view_config_sup,
            transform=self.basic_transform,
            target_transform=lambda x: torch.tensor(x).long(),
        )

    def ssl_augmentations(self, x: np.ndarray):
        """Perform the SSL augmentations"""
        return self._basic_augmentation(x), self._basic_augmentation(x)

    def create_dataloaders(self):
        """Create the dataloaders in place"""
        self.train_dataloader_ssl = DataLoader(
            self.train_dataset_ssl,
            batch_size=self.args.ssl.batch_size,
            sampler=DistributedSampler(self.train_dataset_ssl),
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.train_dataloader_sup = DataLoader(
            self.train_dataset_sup,
            batch_size=self.args.sl.batch_size // self.args.num_gpus,
            sampler=DistributedSampler(self.train_dataset_sup),
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.args.sl.batch_size
            // self.args.num_gpus
            * self.args.sl.batch_size_increase_for_eval,
            sampler=DistributedSampler(self.val_dataset, shuffle=False),
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.args.sl.batch_size
            // self.args.num_gpus
            * self.args.sl.batch_size_increase_for_eval,
            sampler=DistributedSampler(self.test_dataset, shuffle=False),
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def create_model(self):
        """Create the model in place"""
        from trusnet.modeling.registry import resnet18

        model = resnet18()
        model.fc = torch.nn.Identity()
        model.to(dist.get_rank())
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
        if self._loading_checkpoint:
            model.load_state_dict(self._state["model"])

        self.model = model
        dist.barrier()
        print(self.model)

    def create_optimizer(self):
        """Create the optimizer and scheduler in place"""
        self.optimizer = torch.nn.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs
        )
        if self._loading_checkpoint:
            self.optimizer.load_state_dict(self._state["optimizer"])
            self.scheduler.load_state_dict(self._state["scheduler"])

    def train(self):
        self.setup()


def test():
    print(f"Available gpus: {torch.cuda.device_count()}")
    print(f"{torch.cuda.current_device()}=")

    print("exporting PyTorch distributed environment variables")
    dist_env = submitit.helpers.TorchDistributedEnvironment().export()
    print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
    print(f"rank: {dist_env.rank}")
    print(f"world size: {dist_env.world_size}")
    print(f"local rank: {dist_env.local_rank}")
    print(f"local world size: {dist_env.local_world_size}")
    # print_env()

    # Using the (default) env:// initialization method
    dist.init_process_group(backend="nccl")

    print("Distributed environment initialized")
    print(f"{dist.get_rank()=}")
    print(f"{dist.get_world_size()=}")

    print("Testing communication with other processes: ")
    print("Creating Barrier")
    t1 = time.time()
    dist.barrier()
    t2 = time.time()
    print("barrier passed after", t2 - t1, "seconds")

    A = torch.eye(5).to(dist.get_rank())
    print("A=", A)

    from trusnet.modeling.registry import resnet18

    model = resnet18()
    model.fc = torch.nn.Identity()
    model.to(dist.get_rank())
    model = nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
    print(model)


@hydra.main(config_path="config", config_name="vicreg_2")
def main(args):
    executor = submitit.AutoExecutor(folder=LOGS_DIR)
    executor.update_parameters(
        nodes=NUM_NODES,
        gpus_per_node=NUM_TASKS_PER_NODE,
        tasks_per_node=NUM_TASKS_PER_NODE,
        cpus_per_task=NUM_CPUS_PER_TASK,
        slurm_partition=PARTITION,
        timeout_min=15,
    )
    trainer = VICRegTrainer(args)
    job = executor.submit(test)
    submitit.helpers.monitor_jobs([job])
    print(job.results()[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
