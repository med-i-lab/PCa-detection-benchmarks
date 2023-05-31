import logging
from trusnet.utils.accumulators import DictConcatenation
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from projects.IJCARS_2023.utils import (
    show_prob_histogram,
    show_reliability_diagram,
    convert_patchwise_to_corewise_dataframe,
    apply_temperature_calibration,
)
import numpy as np
from torch.nn import functional as F


def compute_metrics(dataframe):
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


class EvalStep:
    def __init__(self, device="cuda"):
        self.device = device

    def __call__(self, model, batch):
        x = batch.pop("patch")
        y = batch.pop("label")
        x = x.to(self.device)
        y = y.to(self.device)
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


def eval_epoch(
    step,
    model,
    val_loader,
    test_loader,
    ood_loader=None,
    epoch=None,
    use_calibration=True,
):
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
            out = step(model, batch)
            acc(out)

    val_df = acc.compute("dataframe")

    logging.info("Evaluating on test set")
    acc = DictConcatenation()
    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(test_loader, desc=f"Epoch {epoch} validation", leave=False)
        ):
            out = step(model, batch)
            acc(out)

    test_df = acc.compute("dataframe")

    if ood_loader is not None:
        logging.info("Evaluating on OOD set")
        acc = DictConcatenation()
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(ood_loader, desc=f"Epoch {epoch} validation", leave=False)
            ):
                out = step(model, batch)
                acc(out)

        ood_df = acc.compute("dataframe")
        output_tables["ood"] = ood_df

        conf = ood_df.confidence.values
        inside_prostate = ood_df.prostate_intersection.values > 0.2
        inside_prostate = inside_prostate.astype(int)
        metrics.update({"ood/auroc": roc_auc_score(inside_prostate, conf)})

    logging.info("Computing metrics")
    metrics = {}

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

    if use_calibration:
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
    metrics.update({f"val/patch_{k}": v for k, v in compute_metrics(val_df).items()})
    metrics.update({f"test/patch_{k}": v for k, v in compute_metrics(test_df).items()})

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

    if use_calibration:
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
        {f"val/core_{k}": v for k, v in compute_metrics(val_df_corewise).items()}
    )

    metrics.update(
        {f"test/core_{k}": v for k, v in compute_metrics(test_df_corewise).items()}
    )

    return (
        metrics,
        output_tables,
        figures,
    )


class TrainStep:
    def __init__(
        self,
        device,
    ):
        self.device = device

    def __call__(self, model, batch, optimizer, epoch):
        optimizer.zero_grad()
        x, y, metadata = batch
        x = x.to(self.device)
        y = y.to(self.device)
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
            **metadata,
        }


def train_epoch(step, model, train_loader, optimizer, epoch, num_epochs):
    model.train()
    acc = DictConcatenation()
    for i, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
    ):
        outputs = step(model, batch, optimizer, epoch)
        acc(outputs)

    df = acc.compute("dataframe")
    metrics = compute_metrics(df)

    return metrics, df
