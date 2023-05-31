import matplotlib.pyplot as plt
from trusnet.utils.metrics import (
    expected_calibration_error,
    brier_score,
    reliability_diagram,
)
import torch
import logging
import numpy as np


def show_reliability_diagram(df):
    probs = df.prob_1.values.squeeze()
    targets = df.y.values.squeeze()
    preds = (probs > 0.5).astype(int)
    conf = np.max(np.stack([probs, 1 - probs], axis=1), axis=1).squeeze()

    ece, _ = expected_calibration_error(preds, conf, targets, n_bins=20)
    brier = brier_score(probs, targets)

    plt.figure()
    reliability_diagram(preds, conf, targets, n_bins=20)
    # put floating caption on plot saying ECE
    plt.text(
        0.1,
        0.9,
        f"ECE: {ece:.3f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.8,
        f"Brier: {brier:.3f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )


def show_prob_histogram(df):
    plt.figure()
    df.query("y == 1").prob_1.hist(bins=20, alpha=0.5, label="cancer", density=True)
    df.query("y == 0").prob_1.hist(bins=20, alpha=0.5, label="benign", density=True)
    plt.axvline(
        0.5,
        color="k",
        linestyle="--",
    )
    plt.legend()
    plt.xlabel("Probability of cancer")
    plt.ylabel("Density")


def convert_patchwise_to_corewise_dataframe(df):
    corewise_df = df.groupby(["core_specifier"]).prob_1.mean().reset_index()
    corewise_df["y"] = (
        df.groupby(["core_specifier"]).y.first().reset_index().y.astype(int)
    )
    return corewise_df


def apply_temperature_calibration(val_df, test_df, lr=1e-2):
    val_probs = torch.tensor(val_df.prob_1.values).view(-1, 1)
    val_y = torch.tensor(val_df.y.values).view(-1, 1)

    from trusnet.utils.calibration import (
        compute_temperature_and_bias_for_calibration,
        apply_temperature_and_bias,
    )

    temp, bias = compute_temperature_and_bias_for_calibration(
        val_probs, val_y, lr=lr, mode="brier"
    )
    val_df["prob_1"] = apply_temperature_and_bias(val_df.prob_1.values, temp, bias)
    test_df["prob_1"] = apply_temperature_and_bias(test_df.prob_1.values, temp, bias)

    return val_df, test_df
