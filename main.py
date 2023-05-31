import hydra
from rich import print as rprint
import submitit
from omegaconf import DictConfig
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate


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

    ex = submitit.AutoExecutor(".")
    ex.update_parameters(
        mem_gb=16,
        cpus_per_task=16,
        timeout_min=180,
        gres="gpu:1",
        slurm_partition="a40",  # "a40,t4v2,rtx6000",
    )

    job = ex.submit(experiment)
    print(f"Submitted job: {job.job_id}")
    print(f"Job stdout at: {job.paths.stdout}")
    print(f"Job stderr at: {job.paths.stderr}")

    # print("Waiting for job to finish...")
    # if args.async_mode:
    #     print("Running in async mode")
    #     return
    # else:
    #     try:
    #         job.result()
    #     except KeyboardInterrupt:
    #         print("Keyboard interrupt detected... Interrupting!")
    #         job.cancel()
    #         print("Job cancelled!")


if __name__ == "__main__":
    main()
