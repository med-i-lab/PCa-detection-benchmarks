# append path one level up
import sys
import os


from tqdm import tqdm

# from functools import partialmethod
#
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from rich.logging import RichHandler
import rich
from trusnet.modeling.vicreg import VICReg
from trusnet.modeling.registry.registry import create_model
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trusnet.utils.driver.stateful import StatefulCollection
from tqdm import tqdm
from contextlib import nullcontext
import wandb
import pandas as pd
from functools import partial
import logging
import hydra
from omegaconf import OmegaConf
import json


LOGGING_LEVEL = logging.DEBUG
VALIDATE_EVERY_N_EPOCHS = 1
BATCH_SIZE_FACTOR_FOR_VALIDATION = 4  # amount to increase the batch size for validation


@hydra.main(config_path="config", config_name="vicreg")
def main(args):
    rich.print(args)
    torch.multiprocessing.spawn(run, nprocs=args.num_gpus, args=(args,))


def run(rank, args):
    # setup distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        "nccl", rank=rank, world_size=args.num_gpus, init_method="env://"
    )
    dist.barrier()

    # set up workdir
    if rank == 0:
        os.makedirs(args.workdir, exist_ok=True)
    dist.barrier()
    # os.chdir(args.workdir)

    if rank != 0:
        # disable tqdm and logging for non-rank 0 processes
        logging.basicConfig(
            level=LOGGING_LEVEL,
            format=f"RANK {rank} %(message)s",
            handlers=[logging.NullHandler()],
        )
        from tqdm import tqdm
        from functools import partialmethod

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    else:
        logging.basicConfig(
            level=LOGGING_LEVEL,
            format="%(message)s",
            handlers=[
                RichHandler(rich_tracebacks=True),
                logging.FileHandler("log.txt"),
            ],
        )

    # set up logging
    logging.info("Setting up logging")
    if not args.debug and rank == 0:
        wandb.init(
            dir=args.workdir,
            config=OmegaConf.to_container(args, resolve=True),
            **args.wandb,
        )

    else:
        wandb.log = lambda x: None

    # set up datasets
    logging.info("Setting up dataset")
    train_dataset = create_ssl_dataset(args)

    (
        supervised_dataset_train,
        supervised_dataset_val,
        supervised_dataset_test,
    ) = create_supervised_datasets(args)
    # set up dataloaders
    local_batch_size = args.batch_size // args.num_gpus
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.num_gpus, rank=rank
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )

    sl_train_dataset, sl_val_dataset, sl_test_dataset = create_supervised_datasets(args)

    # set up model
    logging.info("Setting up model")
    backbone = create_model(args.backbone_name)

    from trusnet.modeling.vicreg import VICReg

    model = VICReg(backbone, [512, 512], 512)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # set up optimizer

    logging.info("Setting up optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=5 * len(train_dataloader),
        max_epochs=args.num_epochs * len(train_dataloader),
    )

    # load checkpoint
    if os.path.exists("checkpoint.pth"):
        logging.info("Loading checkpoint from checkpoint.pth")
        epoch = load_checkpoint(model, optimizer, scheduler, "checkpoint.pth")
    else:
        logging.info("No checkpoint found, starting from scratch")
        epoch = 1
    dist.barrier()

    # train
    while epoch <= args.num_epochs:
        train_epoch(model, optimizer, scheduler, train_dataloader, rank, epoch)
        if epoch % VALIDATE_EVERY_N_EPOCHS == 0 and rank == 0:
            validate_epoch(
                args,
                epoch,
                backbone,
                supervised_dataset_train,
                supervised_dataset_val,
                supervised_dataset_test,
            )
        dist.barrier()
        epoch += 1
        if rank == 0:
            logging.info(
                f"Saving snapshot at end of epoch {epoch - 1} to {args.workdir}"
            )
            checkpoint(model, optimizer, scheduler, epoch, "checkpoint.pth")
        dist.barrier()

    if rank == 0:
        wandb.finish()

    dist.destroy_process_group()


def setup_dist(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_epoch(model, optimizer, scheduler, dataloader, rank, epoch):
    model.train()
    logging.debug(f"Training epoch {epoch} - model.train()")

    losses_accumulator = {}

    for i, batch in enumerate(
        tqdm(dataloader, desc=f"Training epoch {epoch}", leave=False)
    ):
        logging.debug(f"Training epoch {epoch} - batch {i}")
        optimizer.zero_grad()
        X1, X2 = batch[0]
        X1 = X1.to(rank)
        logging.debug(f"Training epoch {epoch} - batch {i} - X1.to(rank)")
        X2 = X2.to(rank)
        logging.debug(f"Training epoch {epoch} - batch {i} - X2.to(rank)")
        losses = model(X1, X2)
        losses["loss"].backward()
        logging.debug(f"Training epoch {epoch} - batch {i} - backward")

        for k, v in losses.items():
            if k not in losses_accumulator:
                losses_accumulator[k] = v.item()
            else:
                losses_accumulator[k] += v.item()

        optimizer.step()
        scheduler.step()

        if rank == 0 and i % 10 == 0:
            metrics = {f"train/{k}": v / (i + 1) for k, v in losses_accumulator.items()}
            metrics["epoch"] = epoch
            metrics["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(metrics)


class LinearModelClosure:
    def __init__(self, X, y, model, optimizer):
        self.X = X
        self.y = y
        self.model = model
        self.optimizer = optimizer

    def __call__(self):
        self.optimizer.zero_grad()
        y_pred = self.model(self.X)
        loss = F.cross_entropy(y_pred, self.y)
        loss.backward()
        return loss


def validate_epoch(args, epoch, model, train_set, val_set, test_set):
    logging.info(f"Validating epoch {epoch}")

    model.eval()

    from trusnet.utils.accumulators import DictConcatenation

    accumulator = DictConcatenation()
    create_loader = partial(
        DataLoader,
        batch_size=args.batch_size * BATCH_SIZE_FACTOR_FOR_VALIDATION,
        num_workers=0,
        shuffle=False,
    )
    train_loader = create_loader(train_set)
    val_loader = create_loader(val_set)
    test_loader = create_loader(test_set)

    with torch.no_grad():
        for i, (X, y, info) in enumerate(
            tqdm(train_loader, desc="Collecting X_train for linear eval", leave=False)
        ):
            logging.debug(f"Collecting X_train for linear eval - batch {i}")
            accumulator(
                {
                    "label": y,
                    "core_specifier": info["core_specifier"],
                    "position": info["position"],
                    "feats": model(X.to(0)),
                }
            )

        out_train = accumulator.compute()
        logging.debug(out_train.keys())
        accumulator.reset()

        for X, y, info in tqdm(
            val_loader, desc="Collecting X_val for linear eval", leave=False
        ):
            accumulator(
                {
                    "label": y,
                    "core_specifier": info["core_specifier"],
                    "position": info["position"],
                    "feats": model(X.to(0)),
                }
            )
        out_val = accumulator.compute()
        accumulator.reset()

        for X, y, info in tqdm(
            test_loader, desc="Collecting X_test for linear eval", leave=False
        ):
            accumulator(
                {
                    "label": y,
                    "core_specifier": info["core_specifier"],
                    "position": info["position"],
                    "feats": model(X.to(0)),
                }
            )
        out_test = accumulator.compute()
        accumulator.reset()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    logging.info("Training linear classifier")
    print(out_train.keys())
    X_train = out_train["feats"].to(0)
    y_train = out_train["label"].to(0)
    X_val = out_val["feats"].to(0)
    y_val = out_val["label"].to(0)
    X_test = out_test["feats"].to(0)
    y_test = out_test["label"].to(0)

    linear = torch.nn.Linear(X_train.shape[1], 2).to(0)
    optimizer = torch.optim.LBFGS(linear.parameters(), lr=1e-3, max_iter=10000)
    closure = LinearModelClosure(X_train, y_train, linear, optimizer)
    loss_init = closure()
    logging.info(f"Initial loss: {loss_init}")
    optimizer.step(closure)
    loss_final = closure()
    logging.info(f"Final loss: {loss_final}")

    logging.info("Evaluating linear classifier")
    y_pred_train = linear(X_train)
    y_pred_val = linear(X_val)
    y_pred_test = linear(X_test)

    y_pred_train = y_pred_train.detach().cpu().numpy()
    y_pred_val = y_pred_val.detach().cpu().numpy()
    y_pred_test = y_pred_test.detach().cpu().numpy()
    y_true_train = y_train.detach().cpu().numpy()
    y_true_val = y_val.detach().cpu().numpy()
    y_true_test = y_test.detach().cpu().numpy()

    auc_train = roc_auc_score(y_true_train, y_pred_train[:, 1])
    auc_val = roc_auc_score(y_true_val, y_pred_val[:, 1])
    auc_test = roc_auc_score(y_true_test, y_pred_test[:, 1])

    metrics = {
        "train/linear_auc": auc_train,
        "val/linear_auc": auc_val,
        "test/linear_auc": auc_test,
    }
    wandb.log(metrics)

    # create csv with predictions, labels, core_specifier, and patch position
    # def make_csv(out):
    #     out_new = {}
    #     out_new["pred"] = out["pred"]
    #     out_new["label"] = out["label"]
    #     out_new["core_specifier"] = out["core_specifier"]
    #     out_new["pos_axial"] = out["position"][:, 0]
    #     out_new["pos_lateral"] = out["position"][:, 2]
    #     return pd.DataFrame(out_new)


#
# out_train = make_csv(out_train)
# out_val = make_csv(out_val)
# out_test = make_csv(out_test)
#
# # save csvs
# logging.info("Saving predictions to csv")
# out_train.to_csv(f"train_preds_{epoch}.csv")
# out_val.to_csv(f"val_preds_{epoch}.csv")
# out_test.to_csv(f"test_preds_{epoch}.csv")
#
# # log metrics
# metrics = {
#     "train/roc_auc": roc_auc_score(out_train["label"], out_train["pred"]),
#     "val/roc_auc": roc_auc_score(out_val["label"], out_val["pred"]),
#     "test/roc_auc": roc_auc_score(out_test["label"], out_test["pred"]),
#     "epoch": epoch,
# }
# wandb.log(metrics)


def checkpoint(model, optimizer, scheduler, epoch, checkpoint_path):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        },
        checkpoint_path + ".tmp",
    )
    os.rename(
        checkpoint_path + ".tmp",
        checkpoint_path,
    )


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    return epoch


def create_supervised_datasets(args):
    from trusnet.data.exact.nct02079025.cohort_selection import (
        get_patient_splits,
        get_cores_for_patients,
        remove_cores_below_threshold_involvement,
        remove_benign_cores_from_positive_patients,
        undersample_benign,
    )

    train_patients, val_patients, test_patients = get_patient_splits(
        args.data.fold, args.data.n_folds
    )

    train_cores = get_cores_for_patients(train_patients)
    val_cores = get_cores_for_patients(val_patients)
    test_cores = get_cores_for_patients(test_patients)
    if args.data.limit_cores is not None:
        train_cores = train_cores[: args.data.limit_cores]
        val_cores = val_cores[: args.data.limit_cores]
        test_cores = test_cores[: args.data.limit_cores]

    train_cores = remove_cores_below_threshold_involvement(train_cores, 0.4)
    train_cores = remove_benign_cores_from_positive_patients(train_cores)
    train_cores = undersample_benign(train_cores)

    val_cores = remove_cores_below_threshold_involvement(val_cores, 0.4)
    val_cores = remove_benign_cores_from_positive_patients(val_cores)

    test_cores = remove_cores_below_threshold_involvement(test_cores, 0.4)
    test_cores = remove_benign_cores_from_positive_patients(test_cores)

    from trusnet.data.exact.nct02079025.dataset.rf_datasets import (
        PatchesDataset,
        PatchViewConfig,
    )

    patch_view_config = PatchViewConfig(
        patch_size=args.data.patch_size,
        patch_strides=(1, 1),
        needle_region_only=True,
        prostate_region_only=False,
    )

    from trusnet.data.exact.nct02079025.dataset import RF_PATCHES_MEAN, RF_PATCHES_STD
    from torchvision import transforms as T

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(RF_PATCHES_MEAN, RF_PATCHES_STD),
            T.Resize((224, 224)),
        ]
    )

    train_dataset = PatchesDataset(
        train_cores,
        patch_view_config,
        transform=transform,
    )

    val_dataset = PatchesDataset(
        val_cores,
        patch_view_config,
        transform=transform,
    )

    test_dataset = PatchesDataset(
        test_cores,
        patch_view_config,
        transform=transform,
    )

    return train_dataset, val_dataset, test_dataset


class ApplyTwice:
    def __init__(self, func) -> None:
        self.func = func

    def __call__(self, X):
        return self.func(X), self.func(X)


def create_ssl_dataset(args):
    from trusnet.data.exact.nct02079025.cohort_selection import (
        get_patient_splits,
        get_cores_for_patients,
        remove_cores_below_threshold_involvement,
        remove_benign_cores_from_positive_patients,
        undersample_benign,
    )

    train_patients, val_patients, test_patients = get_patient_splits(
        args.data.fold, args.data.n_folds
    )

    train_cores = get_cores_for_patients(train_patients)
    if args.data.limit_cores is not None:
        train_cores = train_cores[: args.data.limit_cores]

    from trusnet.data.exact.nct02079025.dataset.rf_datasets import (
        PatchesDataset,
        PatchViewConfig,
    )

    patch_view_config = PatchViewConfig(
        patch_size=args.data.patch_size,
        patch_strides=args.data.ssl_patch_strides,
        needle_region_only=args.data.ssl_needle_only,
        prostate_region_only=False,
    )

    from trusnet.data.exact.nct02079025.dataset import RF_PATCHES_MEAN, RF_PATCHES_STD
    from torchvision import transforms as T

    _transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(RF_PATCHES_MEAN, RF_PATCHES_STD),
            T.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomInvert(),
        ]
    )

    transform = ApplyTwice(_transform)

    train_dataset = PatchesDataset(
        train_cores,
        patch_view_config,
        transform=transform,
    )

    return train_dataset


if __name__ == "__main__":
    main()
