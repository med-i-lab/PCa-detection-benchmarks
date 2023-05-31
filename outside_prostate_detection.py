from pathlib import Path


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path)
    return parser.parse_args()


class Trainer:
    def __call__(self, args):
        import torch

        OLD_STATE = (
            torch.load((args.workdir / "checkpoints" / "last.ckpt"))
            if (args.workdir / "checkpoints" / "last.ckpt").exists()
            else None
        )

        if OLD_STATE is not None:
            print("Resuming training from checkpoint.")
        else:
            print("Starting training from scratch.")
            import os

            os.symlink(
                f'/checkpoint/{os.getenv("USER")}/{os.getenv("SLURM_JOB_ID")}',
                args.workdir / "checkpoints",
                target_is_directory=True,
            )

        from trusnet.data.exact.nct02079025.cohort_selection import (
            get_cores_for_patients,
            get_patient_splits,
            remove_benign_cores_from_positive_patients,
            remove_cores_below_threshold_involvement,
            undersample_benign,
        )

        from trusnet.data.exact.nct02079025.server.segmentation import (
            list_available_prostate_segmentations,
        )

        train_patients, val_patients, test_patients = get_patient_splits(
            fold=0, n_folds=5
        )

        from trusnet.data.exact.nct02079025.dataset.rf_datasets import (
            PatchesDataset,
            PatchViewConfig,
        )

        patch_view_config = PatchViewConfig(
            patch_size=(5, 5),
            patch_strides=(3, 3),
            needle_region_only=False,
            prostate_region_only=False,
        )

        train_cores = get_cores_for_patients(train_patients)
        train_cores = [
            core
            for core in train_cores
            if core in list_available_prostate_segmentations()
        ]
        test_cores = get_cores_for_patients(test_patients)
        test_cores = [
            core
            for core in test_cores
            if core in list_available_prostate_segmentations()
        ]

        from trusnet.data.exact import data_access
        from trusnet.data.exact.transforms import TransformV3, Normalize

        t = TransformV3(norm=Normalize(mode="global"))
        train_dataset = PatchesDataset(
            core_specifier_list=train_cores,
            patch_view_config=patch_view_config,
            transform=t,
        )
        test_dataset = PatchesDataset(
            core_specifier_list=test_cores,
            patch_view_config=patch_view_config,
            transform=t,
        )

        from trusnet.modeling.registry import resnet10

        model = resnet10()
        if OLD_STATE is not None:
            model.load_state_dict(OLD_STATE["state_dict"])
        model.cuda()

        from torch.utils.data import DataLoader
        import torch

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
        )

        from tqdm import tqdm

        from torchmetrics import Accuracy, MetricCollection, AUROC

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if OLD_STATE is not None:
            optimizer.load_state_dict(OLD_STATE["optimizer"])

        metrics = MetricCollection(
            {
                "acc": Accuracy(),
                "acc_macro": Accuracy(num_classes=2, average="macro"),
                "auroc": AUROC(num_classes=2),
            }
        ).cuda()

        start_epoch = OLD_STATE["epoch"] if OLD_STATE is not None else 1

        for epoch in range(start_epoch, 10):
            model.train()
            metrics.reset()
            for batch in tqdm(train_loader):
                x, y, metadata = batch
                x = x.cuda()
                y = (metadata["prostate_intersection"] > 0.5).long().cuda()
                y_hat = model(x)
                loss = torch.nn.functional.cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                metrics(y_hat, y)
            print(f"Epoch {epoch} train acc: {metrics.compute()}")

            metrics.reset()
            with torch.no_grad():
                model.eval()
                for batch in test_loader:
                    x, y, metadata = batch
                    x = x.cuda()
                    y = (metadata["prostate_intersection"] > 0.5).long().cuda()
                    y_hat = model(x)
                    metrics(y_hat, y)
            print(f"Epoch {epoch} test acc: {metrics.compute()}")

            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                (args.workdir / "checkpoints" / "last.ckpt"),
            )

        def checkpoint(self, *args, **kwargs):
            from submitit.core.utils import DelayedSubmission

            print("Preempted. Restarting job.")

            return DelayedSubmission(Trainer(), *args, **kwargs)


def main():
    args = parse_args()
    import submitit

    ex = submitit.AutoExecutor(args.workdir / "submitit")
    ex.update_parameters(
        mem_gb=16,
        cpus_per_task=16,
        timeout_min=180,
        gres="gpu:1",
        slurm_partition="a40,t4v2,rtx6000",
    )

    job = ex.submit(Trainer(), args)
    print(f"Submitted job: {job.job_id}")
    print(f"Job stdout at: {job.paths.stdout}")
    print(f"Job stderr at: {job.paths.stderr}")


if __name__ == "__main__":
    main()
