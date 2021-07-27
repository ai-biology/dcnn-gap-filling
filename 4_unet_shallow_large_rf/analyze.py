#!/usr/bin/env python3

""" Plot gap covers vs gap length for different U-Nets """

from glob import glob
import os
from pprint import pprint

import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from unet.dataset import TessellationDataset, read_line_gap_infos
from unet.gap_cover import compute_single_gap_covers
from unet.receptive_field import get_rf_for_model
from unet.unet import make_unet
from unet.utils import count_parameters


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

BOXPLOT_STYLE = {
    "fliersize": 1,
    "linewidth": 0.7,
    "saturation": 0.6,
}
MARKERS = ["o", "^", "X", "s", "d"]
GAPPLOT_SIZE = (3.5, 2.5)


def load_results(paths):
    """ Load the trained models and results """
    return {
        cond: torch.load(path, map_location=torch.device("cpu"))
        for cond, path in paths.items()
    }


def load_model(result):
    """ Return the trained U-Net from a result dict """
    width = result["width"]
    depth = result["depth"]
    kernel_size = result["kernel_size"]
    unet = make_unet(
        depth,
        width,
        kernel_size=kernel_size,
        padding=True,
        batch_norm=True,
        train=False,
    )
    unet.load_state_dict(result["unet"])
    return unet


def compute_model_stats(results):
    """
    Compute receptive fields and number of parameters for models
    :returns: dict of condition -> rf
    """
    model_stats = {}
    for condition, result in results.items():
        # load the model
        unet = load_model(result)

        # compute receptive field
        model_stats[condition] = {}
        model_stats[condition]["rf"] = get_rf_for_model(unet)
        model_stats[condition]["#params"] = count_parameters(unet)

    print("Finished computing model stats")
    return model_stats


def test_gap_covers(results, dataset, gaps):
    """
    Test gap covers and return accumulated results
    :returns: dict of depth -> DataFrame of single gap covers
    """
    gap_covers = {}
    for condition, result in tqdm(results.items()):
        unet = load_model(result)
        gap_covers[condition] = compute_single_gap_covers(dataset, gaps, unet)

    print("Finished computing gap covers")
    return gap_covers


def plot_max_results(outpath, gap_covers, gaps):
    """
    Create plot showing
        - max gap covers for different gap widths and receptive fields
        - histogram of gap widths
    """
    sns.set_palette("crest", n_colors=2)

    # bins for plotting
    bins = np.logspace(0, 2.0, 50)

    plt.figure(figsize=GAPPLOT_SIZE)

    # plot the histogram
    hist_color = sns.color_palette("flare", n_colors=1)
    plt.hist(gaps.length_manh, bins=bins, color=hist_color, alpha=0.3, zorder=1)
    plt.ylabel("# Gaps")
    plt.xlabel("Gap Width")

    plt.twinx()

    for (label, gap_covers_at_depth), marker in zip(gap_covers.items(), MARKERS):
        # bin the gap widths
        # gap_covers_at_depth = gap_covers[depth]
        gap_covers_at_depth["length_binned"] = pd.cut(
            gap_covers_at_depth.length, bins=bins
        ).map(lambda b: 0.5 * (b.right + b.left))

        max_covers = gap_covers_at_depth.groupby("length_binned").cover.max()
        plt.plot(
            max_covers.index,
            max_covers,
            marker=marker,
            markersize=3,
            markevery=7,
            zorder=10,
            label=label,
        )

    plt.xscale("log")
    plt.ylabel("Max. Gap Cover")
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "gap_length_vs_max_cover.pdf"))
    plt.savefig(os.path.join(outpath, "gap_length_vs_max_cover.png"))


@click.command()
@click.option("--tessellations", "-t", "tessellations_path", required=True)
@click.option("--results", "-r", "results_path", required=True)
def main(tessellations_path, results_path):
    # load all results
    results = load_results(
        {
            "deep": os.path.join(results_path, "unet-deep.torch"),
            "shallow": os.path.join(results_path, "unet-shallow.torch"),
        }
    )
    dataset = TessellationDataset(tessellations_path)
    gaps, _ = read_line_gap_infos(tessellations_path, dataset.img_size)

    # compute gap covers
    gap_covers = test_gap_covers(results, dataset, gaps)

    # print receptive fields and # parameters
    model_stats = compute_model_stats(results)
    pprint(model_stats)

    # create the plot
    plot_max_results(results_path, gap_covers, gaps)


if __name__ == "__main__":
    main()
