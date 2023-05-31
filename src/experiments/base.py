import torch
from trusnet.utils.accumulators import DictConcatenation
from tqdm import tqdm
import os
import argparse
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from trusnet.utils.checkpoints import save_checkpoint
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
import json
from torch.nn import functional as F
from torch.utils.data import DataLoader
from projects.IJCARS_2023.utils import (
    show_prob_histogram,
    show_reliability_diagram,
    convert_patchwise_to_corewise_dataframe,
    apply_temperature_calibration,
)
import rich
from submitit.helpers import Checkpointable, DelayedSubmission
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import Literal


@dataclass
class DatasetArgs:
    fold: int = 0
    n_folds: int = 5
    augmentations_mode: Literal[
        "none", "ultrasound_augs", "tensor_augs", "both"
    ] = "none"
    normalization: str = "instance"
    benign_undersampling_kfold: str = None
    remove_benign_cores_from_positive_patients: bool = False
    remove_low_involvement_cores_for_evaluation: bool = False


class BaseExperiment:
    def __init__(self, args):
        self.args = args
        self.epoch = None
        self.epoch_out = None

    def __call__(self, checkpoint=None):
        if checkpoint is None:
            state = None
            logging.info("Starting main.py")
            print(OmegaConf.to_yaml(self.args))

            os.makedirs(self.args.exp_dir, exist_ok=True)
            if not os.path.exists(os.path.join(self.args.exp_dir, "checkpoints")):
                os.symlink(
                    self.args.ckpt_dir,
                    os.path.join(self.args.exp_dir, "checkpoints"),
                    target_is_directory=True,
                )

            with open(os.path.join(self.args.exp_dir, "config.yaml"), "w") as f:
                yaml.dump(OmegaConf.to_container(self.args, resolve=True), f)

            logging.info("Saved config.yaml")
        else:
            state = torch.load(checkpoint)

        # set up logging
        if self.args.debug:
            wandb.init = lambda **kwargs: None
            wandb.log = lambda x: print(x)

        wandb.init(
            dir=self.args.exp_dir,
            config=OmegaConf.to_container(self.args, resolve=True),
            **self.args.wandb,
        )

        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.args.exp_dir, "log.txt")),
            ],
        )

        torch.random.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        (
            self.train_ds,
            self.val_ds,
            self.test_ds,
            self.ood_test_ds,
        ) = self.create_datasets(self.args)

        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.ood_loader,
        ) = self.create_dataloaders(
            self.train_ds, self.val_ds, self.test_ds, self.ood_test_ds
        )

        model = self.create_model()
        if state is not None:
            model.load_state_dict(state["model"])
        model = model.to(self.args.device)

        optimizer, sched = self.create_optimizer(model)
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])

        if self.epoch_out is None:
            self.epoch_out = DictConcatenation()

        if self.epoch is None:
            self.epoch = 1

        if state is not None:
            torch.random.set_rng_state(state["rng"])

        while self.epoch <= self.args.num_epochs:
            logging.info(f"Starting epoch {self.epoch}")

            metrics = {}
            train_metrics, train_df = self.train_epoch(
                model, self.train_loader, optimizer, self.epoch
            )
            metrics.update({f"train/{k}": v for k, v in train_metrics.items()})

            eval_metrics, eval_dfs, figures = self.eval_epoch(
                model, self.val_loader, self.test_loader, self.ood_loader, self.epoch
            )
            metrics.update(eval_metrics)

            output_dir_for_epoch = os.path.join(
                self.args.exp_dir, "checkpoints", f"epoch_{self.epoch}"
            )
            os.makedirs(output_dir_for_epoch, exist_ok=True)
            for name, df in eval_dfs.items():
                df.to_csv(os.path.join(output_dir_for_epoch, f"{name}.csv"))
            for name, fig in figures.items():
                fig.savefig(os.path.join(output_dir_for_epoch, f"{name}.png"))

            # log figures to wandb
            for name, fig in figures.items():
                fig = wandb.Image(fig)
                wandb.log({name: fig, "epoch": self.epoch})

            metrics["lr"] = optimizer.param_groups[0]["lr"]
            metrics["epoch"] = self.epoch
            self.epoch_out(metrics)
            wandb.log(metrics)

            self.epoch_out.compute("dataframe").to_csv(
                os.path.join(self.args.exp_dir, "metrics.csv")
            )

            if sched is not None:
                sched.step()

            if self.epoch % self.args.checkpoint_freq == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir_for_epoch, "model.pth"),
                )
            self.save_experiment_state(model, optimizer, sched)
            self.epoch += 1

        # post best values
        df = self.epoch_out.compute("dataframe")
        best_ind = df["val/core_auc"].values.argmax()
        best_score = df["val/core_auc"].values[best_ind]
        best_metrics = {
            f"{column}_best": df[column].values[best_ind] for column in df.columns
        }
        wandb.log(best_metrics)
        wandb.finish()

        return best_score

    def train_epoch(self, model, train_loader, optimizer, epoch):
        model.train()
        acc = DictConcatenation()
        for i, batch in enumerate(
            tqdm(
                train_loader, desc=f"Epoch {epoch}/{self.args.num_epochs}", leave=False
            )
        ):
            outputs = self.train_step(model, batch, optimizer, epoch)
            acc(outputs)

        df = acc.compute("dataframe")
        metrics = self.compute_metrics(df)

        return metrics, df

    def train_step(self, model, batch, optimizer, epoch):
        optimizer.zero_grad()
        x = batch.pop("patch")
        y = batch.pop("label")
        x = x.to(self.args.device)
        y = y.to(self.args.device)
        y_hat = model(x)
        prob = F.softmax(y_hat, dim=1)
        pred = prob.argmax(dim=1)
        confidence = prob.max(dim=1).values
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        return {
            "loss": F.cross_entropy(y_hat, y, reduction="none"),
            "y": y,
            "prob": prob,
            "confidence": confidence,
            **batch,
        }

    def eval_epoch(self, model, val_loader, test_loader, ood_loader, epoch):
        logging.info("Evaluating on validation set")

        output_tables = {}
        metrics = {}
        figures = {}

        model.eval()
        acc = DictConcatenation()
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(val_loader, desc=f"Epoch {epoch} validation", leave=False)
            ):
                out = self.eval_step(model, batch)
                acc(out)

        val_df = acc.compute("dataframe")

        logging.info("Evaluating on test set")
        acc = DictConcatenation()
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(test_loader, desc=f"Epoch {epoch} Testing", leave=False)
            ):
                out = self.eval_step(model, batch)
                acc(out)

        test_df = acc.compute("dataframe")

        if self.args.get("test_ood"):
            logging.info("Evaluating on OOD set")
            acc = DictConcatenation()
            with torch.no_grad():
                for i, batch in enumerate(
                    tqdm(ood_loader, desc=f"Epoch {epoch} OOD testing", leave=False)
                ):
                    out = self.eval_step(model, batch)
                    acc(out)

            ood_df = acc.compute("dataframe")
            output_tables["ood"] = ood_df

            conf = ood_df.confidence.values
            inside_prostate = ood_df.prostate_intersection.values > 0.2
            inside_prostate = inside_prostate.astype(int)
            metrics.update({"ood/auroc": roc_auc_score(inside_prostate, conf)})

        logging.info("Computing metrics")
        plt.figure()
        show_prob_histogram(val_df)
        plt.title("Validation set - Probability histogram - uncalibrated")
        figures["val_uncalibrated_prob_hist"] = plt.gcf()

        plt.figure()
        show_prob_histogram(test_df)
        plt.title("Test set - Probability histogram - uncalibrated")
        figures["test_uncalibrated_prob_hist"] = plt.gcf()

        plt.figure()
        show_reliability_diagram(val_df)
        plt.title("Validation set - Reliability diagram - uncalibrated")
        figures["val_uncalibrated_reliability_diagram"] = plt.gcf()

        plt.figure()
        show_reliability_diagram(test_df)
        plt.title("Test set - Reliability diagram - uncalibrated")
        figures["test_uncalibrated_reliability_diagram"] = plt.gcf()

        if self.args.use_calibration:
            val_df, test_df = apply_temperature_calibration(val_df, test_df)

            plt.figure()
            show_prob_histogram(val_df)
            plt.title("Validation set - Probability histogram - calibrated")
            figures["val_calibrated_prob_hist"] = plt.gcf()

            plt.figure()
            show_prob_histogram(test_df)
            plt.title("Test set - Probability histogram - calibrated")
            figures["test_calibrated_prob_hist"] = plt.gcf()

            plt.figure()
            show_reliability_diagram(val_df)
            plt.title("Validation set - Reliability diagram - calibrated")
            figures["val_calibrated_reliability_diagram"] = plt.gcf()

            plt.figure()
            show_reliability_diagram(test_df)
            plt.title("Test set - Reliability diagram - calibrated")
            figures["test_calibrated_reliability_diagram"] = plt.gcf()

        output_tables["val_patchwise"] = val_df
        output_tables["test_patchwise"] = test_df

        # Get patchwise metrics
        metrics.update(
            {f"val/patch_{k}": v for k, v in self.compute_metrics(val_df).items()}
        )
        metrics.update(
            {f"test/patch_{k}": v for k, v in self.compute_metrics(test_df).items()}
        )

        # Get corewise metrics
        val_df_corewise = convert_patchwise_to_corewise_dataframe(val_df)
        test_df_corewise = convert_patchwise_to_corewise_dataframe(test_df)

        plt.figure()
        show_prob_histogram(val_df_corewise)
        plt.title("Validation set - Probability histogram - corewise - uncalibrated")
        figures["val_uncalibrated_prob_hist_corewise"] = plt.gcf()

        plt.figure()
        show_prob_histogram(test_df_corewise)
        plt.title("Test set - Probability histogram - corewise - uncalibrated")
        figures["test_uncalibrated_prob_hist_corewise"] = plt.gcf()

        plt.figure()
        show_reliability_diagram(val_df_corewise)
        plt.title("Validation set - Reliability diagram - corewise - uncalibrated")
        figures["val_uncalibrated_reliability_diagram_corewise"] = plt.gcf()

        plt.figure()
        show_reliability_diagram(test_df_corewise)
        plt.title("Test set - Reliability diagram - corewise - uncalibrated")
        figures["test_uncalibrated_reliability_diagram_corewise"] = plt.gcf()

        if self.args.use_calibration:
            val_df_corewise, test_df_corewise = apply_temperature_calibration(
                val_df_corewise, test_df_corewise, lr=1e-3
            )

            plt.figure()
            show_prob_histogram(val_df_corewise)
            plt.title("Validation set - Probability histogram - corewise - calibrated")
            figures["val_calibrated_prob_hist_corewise"] = plt.gcf()

            plt.figure()
            show_prob_histogram(test_df_corewise)
            plt.title("Test set - Probability histogram - corewise - calibrated")
            figures["test_calibrated_prob_hist_corewise"] = plt.gcf()

            plt.figure()
            show_reliability_diagram(val_df_corewise)
            plt.title("Validation set - Reliability diagram - corewise - calibrated")
            figures["val_calibrated_reliability_diagram_corewise"] = plt.gcf()

            plt.figure()
            show_reliability_diagram(test_df_corewise)
            plt.title("Test set - Reliability diagram - corewise - calibrated")
            figures["test_calibrated_reliability_diagram_corewise"] = plt.gcf()

        output_tables["val_corewise"] = val_df_corewise
        output_tables["test_corewise"] = test_df_corewise

        metrics.update(
            {
                f"val/core_{k}": v
                for k, v in self.compute_metrics(val_df_corewise).items()
            }
        )
        metrics.update(
            {
                f"test/core_{k}": v
                for k, v in self.compute_metrics(test_df_corewise).items()
            }
        )

        return (
            metrics,
            output_tables,
            figures,
        )

    def eval_step(self, model, batch):
        x = batch.pop("patch")
        y = batch.pop("label")
        x = x.to(self.args.device)
        y = y.to(self.args.device)
        y_hat = model(x)
        prob = F.softmax(y_hat, dim=1)
        confidence = prob.max(dim=1).values
        return {
            "loss": F.cross_entropy(y_hat, y, reduction="none"),
            "y": y,
            "prob": prob,
            "confidence": confidence,
            **batch,
        }

    def create_model(self):
        from trusnet.modeling.registry import create_model

        return create_model(self.args.model_name).cuda()

    def create_optimizer(self, model):
        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=0.9,
            )
        elif self.args.optimizer == "novograd":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {self.args.optimizer} not supported.")

        if self.args.scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            sched = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs)

        else:
            sched = None

        return optimizer, sched

    def compute_metrics(self, dataframe):
        from sklearn.metrics import roc_auc_score, recall_score
        from trusnet.utils.metrics import brier_score, expected_calibration_error

        auc = roc_auc_score(dataframe.y, dataframe.prob_1)
        sensitivity = recall_score(dataframe.y, dataframe.prob_1 > 0.5)
        specificity = recall_score(1 - dataframe.y, dataframe.prob_1 < 0.5)

        probs = dataframe.prob_1.values
        targets = dataframe.y.values
        preds = (probs > 0.5).astype(int)
        conf = np.max(np.stack([probs, 1 - probs], axis=1), axis=1).squeeze()

        ece, _ = expected_calibration_error(preds, conf, targets, n_bins=20)

        brier = brier_score(probs, targets)

        return {
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "brier_score": brier,
            "ece": ece,
        }

    @staticmethod
    def create_datasets(args: DatasetArgs = DatasetArgs()):
        from trusnet.data.exact.nct02079025.cohort_selection import (
            get_cores_for_patients,
            get_patient_splits,
            remove_benign_cores_from_positive_patients,
            remove_cores_below_threshold_involvement,
            undersample_benign,
            undersample_benign_as_kfold,
        )

        train_patients, val_patients, test_patients = get_patient_splits(
            fold=args.fold, n_folds=args.n_folds
        )
        train_cores = get_cores_for_patients(train_patients)
        val_cores = get_cores_for_patients(val_patients)
        test_cores = get_cores_for_patients(test_patients)

        train_cores = remove_cores_below_threshold_involvement(train_cores, 40)
        if args.remove_low_involvement_cores_for_evaluation:
            val_cores = remove_cores_below_threshold_involvement(val_cores, 40)
            test_cores = remove_cores_below_threshold_involvement(test_cores, 40)

        train_cores = remove_benign_cores_from_positive_patients(train_cores)
        if args.benign_undersampling_kfold is None:
            train_cores = undersample_benign(train_cores)
        else:
            train_cores = undersample_benign_as_kfold(
                train_cores,
            )[args.benign_undersampling_kfold]

        from trusnet.data.exact.dataset.rf_datasets import (
            PatchesDataset,
            PatchViewConfig,
        )
        from trusnet.data.exact.transforms import (
            TransformV3,
            TensorImageAugmentation,
            UltrasoundArrayAugmentation,
            Normalize,
        )

        patch_view_cfg = PatchViewConfig(
            needle_region_only=True,
            prostate_region_only=False,
        )

        if args.normalization == "instance":
            norm = Normalize()
        else:
            norm = Normalize(
                mode="global",
                type="z-score",
            )

        eval_transform = TransformV3(norm=norm)
        if args.augmentations_mode == "none":
            train_transform = eval_transform
        elif args.augmentations_mode == "tensor_augs":
            train_transform = TransformV3(
                norm=norm,
                tensor_transform=TensorImageAugmentation(
                    random_resized_crop=True,
                    random_affine_rotation=10,
                    random_affine_translation=[0.1, 0.1],
                ),
            )
        elif args.augmentations_mode == "ultrasound_augs":
            train_transform = TransformV3(
                norm=norm,
                us_augmentation=UltrasoundArrayAugmentation(
                    random_phase_shift=True,
                    random_phase_distort=True,
                    random_envelope_distort=True,
                ),
            )
        elif args.augmentations_mode == "both":
            train_transform = TransformV3(
                norm=norm,
                tensor_transform=TensorImageAugmentation(
                    random_resized_crop=True,
                    random_affine_rotation=10,
                    random_affine_translation=[0.1, 0.1],
                ),
                us_augmentation=UltrasoundArrayAugmentation(
                    random_phase_shift=True,
                    random_phase_distort=True,
                    random_envelope_distort=True,
                ),
            )
        else:
            raise ValueError("Unknown augmentations_mode")

        train_dataset = PatchesDataset(
            core_specifier_list=train_cores,
            patch_view_config=patch_view_cfg,
            transform=train_transform,
            target_transform=BaseExperiment._label_transform,
        )
        val_dataset = PatchesDataset(
            core_specifier_list=val_cores,
            patch_view_config=patch_view_cfg,
            transform=eval_transform,
            target_transform=BaseExperiment._label_transform,
        )
        test_dataset = PatchesDataset(
            core_specifier_list=test_cores,
            patch_view_config=patch_view_cfg,
            transform=eval_transform,
            target_transform=BaseExperiment._label_transform,
        )

        # we also need to create a dataset to test OOD
        from trusnet.data.exact.server.segmentation import (
            list_available_prostate_segmentations,
        )

        ood_test_cores = list(
            set(test_cores) & set(list_available_prostate_segmentations())
        )
        ood_patch_view_cfg = PatchViewConfig(
            needle_region_only=False,
            prostate_region_only=False,
            patch_strides=(2, 2),
        )
        ood_test_dataset = PatchesDataset(
            core_specifier_list=ood_test_cores,
            patch_view_config=ood_patch_view_cfg,
            transform=eval_transform,
            target_transform=BaseExperiment._label_transform,
        )

        return train_dataset, val_dataset, test_dataset, ood_test_dataset

    def create_dataloaders(self, train_ds, val_ds, test_ds, ood_ds):
        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        ood_loader = DataLoader(
            ood_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader, ood_loader

    def save_experiment_state(self, model, optimizer, scheduler):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": torch.get_rng_state(),
        }
        tmp = os.path.join(self.args.exp_dir, "checkpoints", "tmp.pth")
        permanent = os.path.join(self.args.exp_dir, "checkpoints", "last.pth")
        torch.save(state, tmp)
        os.rename(tmp, permanent)

    def checkpoint(self, *args, **kwargs):
        logging.info(f"Caught checkpoint signal at epoch {self.epoch}.")
        if os.path.exists(os.path.join(self.args.exp_dir, "checkpoints", "last.pth")):
            logging.info("Detected checkpoint file. Requeuing with checkpoint.")
            kwargs["checkpoint"] = os.path.join(
                self.args.exp_dir, "checkpoints", "last.pth"
            )
        else:
            logging.info(
                "No checkpoint file detected. Requeuing to start from scratch."
            )
        return DelayedSubmission(self, *args, **kwargs)

    @staticmethod
    def _label_transform(label):
        return torch.tensor(label, dtype=torch.long)
