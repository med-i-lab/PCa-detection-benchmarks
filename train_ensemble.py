import hydra
import logging
import rich
from rich.logging import RichHandler
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import importlib
import submitit
from projects.IJCARS_2023.src.experiments.base import BaseExperiment
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np


@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig):
    logger.info(f"Running with args: {OmegaConf.to_yaml(args)}")
    with open_dict(args):
        target = args.pop("_target_")
    _module = ".".join(target.split(".")[:-1])
    _class = target.split(".")[-1]
    _module = importlib.import_module(_module)
    _class = getattr(_module, _class)

    experiment: BaseExperiment = _class(args)

    ex = submitit.AutoExecutor(".slurm_logs")
    ex.update_parameters(
        mem_gb=24,
        cpus_per_task=16,
        timeout_min=120,
        slurm_gres="gpu:1",
        slurm_partition="a40,t4v2,rtx6000",
    )

    ensemble_trainer = TrainEnsemble(experiment)
    seeds = list(range(args.num_seeds))
    jobs = ex.map_array(ensemble_trainer, seeds)

    print(f"Submitted jobs: {[job.job_id for job in jobs]}")
    print(f"Waiting for jobs to finish...")

    results = [job.result() for job in jobs]
    print(f"Validation aurocs for members:")
    for seed in seeds:
        print(f"Seed {seed}: {results[seed]}")

    # now all jobs are done, we can aggregate the results and test the ensemble
    job = ex.submit(test_ensemble, experiment)
    job.result()


class TrainEnsemble:
    def __init__(self, experiment):
        self.experiment = experiment

    def __call__(self, seed, benign_cores_fold=None):
        self.experiment.args = self.experiment.args.copy()
        self.experiment.args.seed = seed
        # hacky way to set the benign_cores_fold
        if self.experiment.args.benign_undersampling_kfold is not None:
            self.experiment.args.benign_undersampling_kfold = seed
        self.experiment.args.exp_dir = self.experiment.args.exp_dir + f"/seed_{seed}"
        return self.experiment()


def test_ensemble(base_experiment: BaseExperiment):
    OmegaConf.resolve(base_experiment.args)

    train_ds, val_ds, test_ds, ood_ds = base_experiment.create_datasets(
        base_experiment.args
    )
    (
        train_loader,
        val_loader,
        test_loader,
        ood_loader,
    ) = base_experiment.create_dataloaders(train_ds, val_ds, test_ds, ood_ds)

    dataframes = []
    for seed in range(base_experiment.args.num_seeds):
        dir = base_experiment.args.exp_dir + f"/seed_{seed}"
        metrics_table = pd.read_csv(dir + "/metrics.csv")
        best_epoch = metrics_table["val/core_auc"].idxmax() + 1
        print(f"Best epoch for seed {seed}: {best_epoch}")
        checkpoint = torch.load(dir + f"/checkpoints/epoch_{best_epoch}/model.pth")
        model = base_experiment.create_model()
        model.load_state_dict(checkpoint)
        model.cuda()

        metrics, dataframes_, figures = base_experiment.eval_epoch(
            model, val_loader, test_loader, ood_loader, epoch=best_epoch
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

        ensemble_metrics[key] = base_experiment.compute_metrics(
            ensemble_dataframes[key]
        )
        ensemble_dataframes[key].to_csv(
            base_experiment.args.exp_dir + f"/{key}_ensemble.csv"
        )
    # save ensemble metrics as well
    pd.DataFrame(ensemble_metrics).to_csv(
        base_experiment.args.exp_dir + f"/metrics_ensemble.csv"
    )


if __name__ == "__main__":
    main()
