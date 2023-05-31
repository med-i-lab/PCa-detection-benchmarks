# Copyright (c) Arthur Mensch <arthur.mensch@m4x.org>
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD 3-clauses license.
# Original at https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html
#

import functools
import pickle
import time
from pathlib import Path

import submitit
from submitit.helpers import DelayedSubmission


# class MnistTrainer(submitit.helpers.Checkpointable):
#    """
#    This shows how to rewrite a monolith function so that it can handle preemption nicely,
#    and not restart from scratch everytime it's preempted.
#    """
#
#    def __init__(self, clf):
#        # This is the state that will be saved by `checkpoint`
#        self.train_test = None
#        self.scaler = None
#        self.clf = clf
#        self.trained_clf = False
#        self.stage = "0"
#
#    def __call__(self, train_samples: int, model_path: Path = None):
#        # `train_samples` and `model_path` will also be saved
#        log = functools.partial(print, flush=True)
#        log(f"*** Starting from stage '{self.stage}' ***")
#
#        if self.train_test is None:
#            self.stage = "Data Loading"
#            t0 = time.time()
#            log(f"*** Entering stage '{self.stage}' ***")
#            # Load data from https://www.openml.org/d/554
#            X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
#            X, y = X.numpy(), y.numpy()
#
#            random_state = check_random_state(0)
#            permutation = random_state.permutation(X.shape[0])
#            X = X[permutation]
#            y = y[permutation]
#            X = X.reshape((X.shape[0], -1))
#
#            # Checkpoint 1: save the train/test splits
#            X_train, X_test, y_train, y_test = train_test_split(
#                X, y, train_size=train_samples, test_size=10000
#            )
#            self.train_test = X_train, X_test, y_train, y_test
#            log(f"Loaded data, shuffle and split in {time.time() - t0:.1f}s")
#
#        X_train, X_test, y_train, y_test = self.train_test
#        if self.scaler is None:
#            self.stage = "Data Cleaning"
#            t0 = time.time()
#            log(f"*** Entering stage '{self.stage}' ***")
#            scaler = StandardScaler()
#            X_train = scaler.fit_transform(X_train)
#            X_test = scaler.transform(X_test)
#            # Scaling is actual pretty fast, make it a bit slower to allow preemption to happen here
#            time.sleep(10)
#            # Checkpoint 2: save the scaler and the preprocessed data
#            self.scaler = scaler
#            self.train_test = X_train, X_test, y_train, y_test
#            log(f"Scaled the data took {time.time() - t0:.0f}s")
#
#        if not self.trained_clf:
#            self.stage = "Model Training"
#            t0 = time.time()
#            log(f"*** Entering stage '{self.stage}' ***")
#            self.clf.C = 50 / train_samples
#            self.clf.fit(X_train, y_train)
#            # Checkpoint 3: mark the classifier as trained
#            self.trained_clf = True
#            log(f"Training took {time.time() - t0:.0f}s")
#
#        sparsity = np.mean(self.clf.coef_ == 0) * 100
#        score = self.clf.score(X_test, y_test)
#        log(f"Sparsity with L1 penalty: {sparsity / 100:.2%}")
#        log(f"Test score with L1 penalty: {score:.4f}")
#
#        if model_path:
#            self.save(model_path)
#        return score
#
#    def checkpoint(self, *args, **kwargs):
#        print(f"Checkpointing at stage '{self.stage}'")
#        return super().checkpoint(*args, **kwargs)
#
#    def save(self, model_path: Path):
#        with open(model_path, "wb") as o:
#            pickle.dump((self.scaler, self.clf), o, pickle.HIGHEST_PROTOCOL)

from dataclasses import dataclass


class Experiment:
    def __init__(self, max_steps=100):
        self.step = 1
        self.max_steps = max_steps

    def __call__(self):
        while self.step < self.max_steps:
            print(f"Running step {self.step}/{self.max_steps}")
            time.sleep(1)
            self.step += 1

    def checkpoint(self, *args, **kwargs):
        print(f"Checkpointing at step {self.step}")
        return DelayedSubmission(self, *args, **kwargs)


def main():
    ex = submitit.AutoExecutor("log_test")
    if ex.cluster == "slurm":
        print("Executor will schedule jobs on Slurm.")
    else:
        print(
            f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'"
        )

    # Specify the job requirements.
    # Reserving only as much resource as you need ensure the cluster resource are
    # efficiently allocated.
    ex.update_parameters(
        mem_gb=16,
        cpus_per_task=16,
        timeout_min=120,
        gres="gpu:1",
        slurm_partition="a40,t4v2,rtx6000",
    )
    job = ex.submit(Experiment())

    while not job.state == "RUNNING":
        print(f"Waiting for job to start. Current state: {job.state}")
        time.sleep(1)

    print(f"Job is running. Current state: {job.state}")
    print(f"Job stdout at: {job.paths.stdout}")

    time.sleep(10)
    print("Simulating preemption")
    job._interrupt()

    while not job.state == "RUNNING":
        print(f"Waiting for job to start. Current state: {job.state}")
        time.sleep(1)

    print(f"Job result: {job.result()}")


if __name__ == "__main__":
    main()
