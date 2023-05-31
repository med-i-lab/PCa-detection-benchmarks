import submitit
import hydra
import optuna
import os
from omegaconf import DictConfig


def objective(trial: optuna.Trial, args: DictConfig):
    args = args.copy()
    from projects.IJCARS_2023.sngp import SNGP

    args.exp_dir = os.path.join(args.exp_dir, f"trial_{trial.number}")
    args.lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    args.weight_decay = trial.suggest_loguniform("weight_decay", 1e-7, 1e-2)
    args.feature_scale = trial.suggest_loguniform("feature_scale", 1e-2, 10)
    args.ridge_penalty = trial.suggest_loguniform("ridge_penalty", 1e-2, 10)
    args.batch_size = trial.suggest_int("batch_size", 32, 128)

    executor = submitit.AutoExecutor(args.exp_dir)
    executor.update_parameters(
        mem_gb=24,
        cpus_per_task=16,
        timeout_min=90,
        slurm_gres="gpu:1",
        slurm_partition="a40,t4v2,rtx6000",
    )

    f = SNGP(args)
    job = executor.submit(f)
    print(f"Submitted job: {job.job_id}")
    print(f"Job stdout at: {job.paths.stdout}")

    return job.result()


@hydra.main(config_path="config", config_name="sngp_sweep")
def main(args):

    if not "optuna.db" in os.listdir():
        print("Creating new study")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            storage="sqlite:///optuna.db",
            study_name="sngp_sweep",
        )
    else:
        print("Loading study from optuna.db")
        study = optuna.load_study(
            storage="sqlite:///optuna.db",
            study_name="sngp_sweep",
        )

    from functools import partial

    _objective = partial(objective, args=args)
    study.optimize(
        _objective,
        n_trials=20,
        n_jobs=4,
    )


if __name__ == "__main__":
    main()
