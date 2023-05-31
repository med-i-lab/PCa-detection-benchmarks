from src.dataset_factory import (
    create_datasets,
    add_dataset_args,
)
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from trusnet.modeling.registry import resnet10
from src.experiments.base import BaseExperiment
from src.training import eval_epoch, compute_metrics, EvalStep
import numpy as np


def add_args(parser):
    parser = add_dataset_args(parser)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--late_tc", action="store_true", default=False)
    parser.add_argument("--early_tc", action="store_true", default=False)


def test_ensemble(args):
    # look for experiment dir
    assert os.path.exists(
        args.exp_dir
    ), f"Experiment dir {args.exp_dir} does not exist."

    subdirs = sorted(
        [
            os.path.join(args.exp_dir, subdir)
            for subdir in os.listdir(args.exp_dir)
            if subdir.startswith("seed_")
        ]
    )

    _, val_ds, test_ds = create_datasets(args)
    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    dataframes = []

    for dir in subdirs:
        seed = int(dir.split("_")[-1])
        metrics_table = pd.read_csv(os.path.join(dir, "metrics.csv"))
        best_epoch = metrics_table["val/core_auc"].idxmax() + 1
        print(f"Best epoch for seed {seed}: {best_epoch}")
        checkpoint = torch.load(
            os.path.join(dir, "checkpoints", f"epoch_{best_epoch}", "model.pth")
        )

        model = resnet10()
        model.load_state_dict(checkpoint)
        model.cuda()

        metrics, dataframes_, figures = eval_epoch(
            EvalStep(),
            model,
            val_loader,
            test_loader,
            ood_loader=None,
            epoch=None,
            use_calibration=args.early_tc,
        )
        dataframes.append(dataframes_)

    # convert to ensemble
    ensemble_dataframes = {}
    ensemble_metrics = {}

    for key in dataframes[0].keys():
        preds = np.stack([df[key].prob_1.values for df in dataframes])
        preds = np.mean(preds, axis=0)
        ensemble_dataframes[key] = dataframes[0][key].copy()
        ensemble_dataframes[key]["prob_1"] = preds

    if args.late_tc:
        val_df = ensemble_dataframes["val_patchwise"]
        test_df = ensemble_dataframes["test_patchwise"]

        from utils import (
            apply_temperature_calibration,
            convert_patchwise_to_corewise_dataframe,
        )

        val_df, test_df = apply_temperature_calibration(val_df, test_df)

        ensemble_dataframes["val_patchwise"] = val_df
        ensemble_dataframes["test_patchwise"] = test_df

        ensemble_dataframes["val_corewise"] = convert_patchwise_to_corewise_dataframe(
            val_df
        )
        ensemble_dataframes["test_corewise"] = convert_patchwise_to_corewise_dataframe(
            test_df
        )

    for key in ensemble_dataframes.keys():
        ensemble_metrics[key] = compute_metrics(ensemble_dataframes[key])
        ensemble_dataframes[key].to_csv(
            args.exp_dir + f"/{key}_ensemble{args.output_suffix}.csv"
        )

    # save ensemble metrics as well
    pd.DataFrame(ensemble_metrics).to_csv(
        args.exp_dir + f"/metrics_ensemble{args.output_suffix}.csv"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    from submitit import AutoExecutor

    executor = AutoExecutor(folder=os.path.join(args.exp_dir, "submitit"))
    executor.update_parameters(
        mem_gb=36,
        cpus_per_task=16,
        timeout_min=120,
        slurm_gres="gpu:1",
        slurm_partition="a40,t4v2,rtx6000",
    )
    job = executor.submit(test_ensemble, args)
    print(f"Submitted job {job.job_id}")
    print(f"Log file: {job.paths.stdout}")
    print(f"Err file: {job.paths.stderr}")
    # job.result()


if __name__ == "__main__":
    main()
