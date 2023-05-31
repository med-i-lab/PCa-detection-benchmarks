import hydra
from rich import print as rprint
import submitit
from omegaconf import DictConfig
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
import pandas as pd


@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig):

    with open_dict(args):
        target = args.pop("_target_")

    target_module = ".".join(target.split(".")[:-1])
    target_class = target.split(".")[-1]

    import importlib

    target_module = importlib.import_module(target_module)
    target_class = getattr(target_module, target_class)

    print(target_class)
    experiment = target_class(args)
    print(f"Experiment: {target_class}")
    print("Args:")
    print(OmegaConf.to_yaml(args))

    ex = submitit.AutoExecutor(".slurm_logs")
    ex.update_parameters(
        mem_gb=16,
        cpus_per_task=16,
        timeout_min=180,
        gres="gpu:1",
        slurm_partition="a40,t4v2,rtx6000",
    )

    k_fold_train = TrainFold(experiment)
    folds = range(args.n_folds)

    jobs = ex.map_array(k_fold_train, folds)
    print(f"Submitted jobs: {[job.job_id for job in jobs]}")
    print(f"Waiting for jobs to finish...")
    results = [job.result() for job in jobs]
    for i, result in enumerate(results):
        print(f"Fold {i}: {result}")
    print(f"Jobs finished!")

    print("Getting results...")
    best_metrics_by_fold = []
    for fold in folds:
        dir = experiment.args.exp_dir + f"/fold_{fold}"
        metrics_by_epoch = pd.read_csv(dir + "/metrics.csv")
        best_epoch = metrics_by_epoch["val/core_auc"].idxmax()
        best_metrics = metrics_by_epoch.iloc[best_epoch].to_dict()
        best_metrics_by_fold.append(best_metrics)

    best_metrics_by_fold = pd.DataFrame(best_metrics_by_fold)
    best_metrics_by_fold.index.name = "fold"
    best_metrics_by_fold.to_csv(experiment.args.exp_dir + "/best_metrics.csv")
    print("Results saved!")
    best_metrics_by_fold.describe().to_csv(
        experiment.args.exp_dir + "/best_metrics_summary.csv"
    )


class TrainFold:
    def __init__(self, experiment):
        self.experiment = experiment

    def __call__(self, fold):
        self.experiment.args = self.experiment.args.copy()
        self.experiment.args.fold = fold
        self.experiment.args.exp_name = self.experiment.args.exp_name + f"_fold_{fold}"
        self.experiment.args.exp_dir = self.experiment.args.exp_dir + f"/fold_{fold}"
        return self.experiment()

    def checkpoint(self, *args, **kwargs):
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


if __name__ == "__main__":
    main()
