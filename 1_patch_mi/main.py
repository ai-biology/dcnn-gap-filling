#!/usr/bin/env python3

import os

import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from unet.mutual_information import mutual_information_discrete, entropy_discrete
from unet.dataset import TessellationDataset, read_line_gap_infos

# plot styling
sns.set_style("ticks")
sns.set_context(
    "paper",
    font_scale=0.6,
    rc={
        "figure.dpi": 300,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
    },
)

# some constants
EMPTY_PATCH_THRESHOLD = 0.23
TRUE_EMPTY_PATCH = 0
FALSE_EMPTY_PATCH = 1
LINE_PATCH = 2
OFFSET_PATCH = 3


def receptive_field_decomposition(X, Y, width, n_patches, rng, weights=None):
    """
    Decompose imaging dataset X, Y into receptive field patches given their width.
    :param n_patches: Number of patches to return
    :param rng: numpy random number generator
    :param weights: Weight the patches e.g. with an effective receptive field
    """
    img_size = X.shape[-1]
    assert X.shape[-2:] == (img_size, img_size)

    # sample from X and offsets
    img_ids = rng.choice(len(X), size=n_patches, replace=True)
    offsets = rng.choice(img_size - width + 1, size=(n_patches, 2), replace=True)

    # decompose X, Y into receptive fields
    R = np.zeros((n_patches, width, width))
    P = np.zeros(n_patches, dtype="uint8")

    for i, (img_id, (x_offset, y_offset)) in enumerate(zip(img_ids, offsets)):
        R[i] = X[img_id, x_offset : x_offset + width, y_offset : y_offset + width]
        P[i] = Y[img_id, x_offset + width // 2, y_offset + width // 2]

    if weights:
        R = R * weights

    return R, P


def classify_patches(patches, targets, threshold):
    """ Heuristically classify patches into archetypes """
    empty_mask = (patches < threshold).reshape(len(patches), -1).all(axis=-1)
    targets = targets.astype(bool)

    true_empty_mask = empty_mask & (~targets)
    false_empty_mask = empty_mask & targets
    line_mask = (~empty_mask) & targets
    offset_mask = (~empty_mask) & (~targets)
    assert np.all(true_empty_mask | false_empty_mask | line_mask | offset_mask)

    patch_classes = np.zeros(len(patches), "uint8")
    patch_classes[true_empty_mask] = TRUE_EMPTY_PATCH
    patch_classes[false_empty_mask] = FALSE_EMPTY_PATCH
    patch_classes[line_mask] = LINE_PATCH
    patch_classes[offset_mask] = OFFSET_PATCH

    return patch_classes


def subsample_patch_decomposition(dataset, rf, n_images, n_patches, rng):
    """
    Subsample images, run patch decomposition and subsample patches
    """
    idxs = rng.choice(len(dataset.observed), replace=False, size=n_images)
    X = torch.cat(dataset.observed)[idxs]
    Y = torch.stack(dataset.segmented)[idxs]
    patches, targets = receptive_field_decomposition(X, Y, rf, n_patches, rng)

    return patches, targets


def compute_patch_mi(dataset, rf, n_images, n_patches, rng):
    """
    Calculate mutual information etc. given receptive field size
    :param n_images: number of images to subsample from dataset
    :param n_patches: fix the number of patches used for MI estimation
    :param rng: numpy random number generator
    """
    # get archetypal patch classes
    patches, targets = subsample_patch_decomposition(
        dataset, rf, n_images, n_patches, rng
    )
    patch_classes = classify_patches(patches, targets, EMPTY_PATCH_THRESHOLD)

    # identify all empty patches
    patch_classes[np.nonzero(patch_classes == 1)] = TRUE_EMPTY_PATCH

    mi = mutual_information_discrete(patch_classes, targets)
    entropy = entropy_discrete(targets)
    prof = mi / entropy

    return {
        "method": "patch",
        "rf": rf,
        "n_patches": len(patches),
        "mutual_information": mi,
        "entropy": entropy,
        "proficiency": prof,
    }


def plot_results(outpath, results, gaps):
    """ Plot the profciencies and gap length distribution """
    plt.figure(dpi=300, figsize=(3.5, 2.0))

    hist_color = sns.color_palette("flare", n_colors=1)[0]
    line_color = sns.color_palette("crest", n_colors=1)[0]

    sns.histplot(
        data=gaps,
        x="length_manh",
        bins=np.logspace(0, 2, 50),
        stat="count",
        color=hist_color,
        alpha=0.3,
        zorder=-1,
    )
    plt.ylabel("# gaps")
    plt.xlabel("gap width / receptive field size")

    plt.twinx()

    patch_results = results.loc[results.method == "patch"]
    sns.lineplot(
        data=patch_results,
        x="rf",
        y="proficiency",
        marker="o",
        color=line_color,
    )

    plt.xscale("log")
    plt.ylabel("proficiency")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "mi_vs_rf.pdf"), bbox_inches="tight")


def plot_threshold(outpath, dataset, threshold):
    """ Plot the image pixel intensities and the threshold """
    pixels = torch.cat(dataset.observed).reshape(-1)

    hist_color = sns.color_palette("crest", n_colors=1)[0]
    line_color = sns.color_palette("flare", n_colors=1)[0]

    plt.figure(dpi=300, figsize=(3.5, 2.0))
    sns.histplot(
        pixels,
        bins=50,
        stat="density",
        color=hist_color,
        alpha=0.3,
        label="max pixel intensity histogram",
    )
    plt.axvline(
        threshold, color=line_color, linestyle="--", label="empty patch threshold"
    )

    plt.yscale("log")
    plt.xlabel("pixel intensity")
    # plt.title("Determining the threshold for empty patches")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "threshold.pdf"), bbox_inches="tight")


@click.command()
@click.option("--dataset", "-d", "dataset_path", required=True)
@click.option("--outpath", "-o", required=True)
def main(dataset_path, outpath):
    # experiment parameters
    rfs = [1, 3, 5, 7, 9, 13, 17, 23, 29, 37, 47, 61, 79, 99]
    n_patches = 250_000
    n_images = 1000

    print("RFs:")
    print(rfs)

    # load dataset
    dataset = TessellationDataset(dataset_path)
    gaps, _ = read_line_gap_infos(dataset_path, dataset.img_size)

    rng = np.random.default_rng(seed=42)
    results = []
    for rf in tqdm(rfs):
        result = compute_patch_mi(dataset, rf, n_images, n_patches, rng)
        results.append(result)
    results = pd.DataFrame(results)

    # plot result
    plot_results(outpath, results, gaps)

    # plot threshold
    plot_threshold(outpath, dataset, EMPTY_PATCH_THRESHOLD)


if __name__ == "__main__":
    main()
