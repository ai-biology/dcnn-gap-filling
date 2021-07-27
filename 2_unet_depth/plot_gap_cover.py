#!/usr/bin/env python3

""" Plot gap covers vs gap length for different U-Nets """

from glob import glob
import os
import pickle

import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
from skimage import draw
import torch

from unet.dataset import TessellationDataset, read_line_gap_infos
from unet.gap_cover import predict_single
from unet.unet import make_unet

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
BOXPLOT_SIZE = (1.1, 2.0)
GAPPLOT_SIZE = (3.5, 2.5)
EXAMPLE_SIZE = (3.5, 1.16)


def load_results(paths):
    """Load the trained models and results"""
    results = [torch.load(path, map_location=torch.device("cpu")) for path in paths]
    results.sort(key=lambda r: r["depth"])
    return results


def load_model(result):
    """Return the trained U-Net from a result dict"""
    width = result["width"]
    depth = result["depth"]
    unet = make_unet(depth, width, padding=True, batch_norm=True, train=False)
    unet.load_state_dict(result["unet"])
    return unet


def plot_max_results(outpath, receptive_fields, gap_covers, gaps):
    """
    Create plot showing
        - max gap covers for different gap widths and receptive fields
        - histogram of gap widths
    """
    # bins for plotting
    bins = np.logspace(0, 2.0, 50)
    gap_covers["length_binned"] = pd.cut(gap_covers.length, bins=bins).map(
        lambda b: 0.5 * (b.right + b.left)
    )

    plt.figure(figsize=GAPPLOT_SIZE)

    # plot the histogram
    hist_color = sns.color_palette("flare", n_colors=1)
    plt.hist(gaps.length_manh, bins=bins, color=hist_color, alpha=0.3, zorder=1)
    plt.ylabel("# Gaps")
    plt.xlabel("Gap Width")

    plt.twinx()
    for depth, marker in zip(sorted(receptive_fields.keys()), MARKERS):
        # bin the gap widths
        gap_covers_at_depth = gap_covers.loc[gap_covers.depth == depth]
        max_covers = gap_covers_at_depth.groupby(
            ["run_id", "length_binned"]
        ).cover.max()

        mean_max_covers = max_covers.mean(level=1)
        sem_max_covers = max_covers.sem(level=1)
        conf_interval = stats.t.interval(
            0.95,
            gap_covers.run_id.nunique() - 1,
            loc=mean_max_covers,
            scale=sem_max_covers,
        )
        bin_centers = mean_max_covers.index

        plt.plot(
            bin_centers,
            mean_max_covers,
            marker=marker,
            markersize=3,
            markevery=7,
            zorder=10,
            label=f"Depth: {depth}, RF: {receptive_fields[depth]:.0f}",
        )

        plt.fill_between(bin_centers, *conf_interval, zorder=5, alpha=0.3)

    plt.xscale("log")
    plt.ylabel("Max. Gap Cover")
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "gap_length_vs_max_cover.pdf"))
    plt.savefig(os.path.join(outpath, "gap_length_vs_max_cover.png"))


@click.command()
@click.option("--tessellations", "-t", "tessellations_path", required=True)
@click.option("--results", "-r", "results_path", required=True)
@click.option("--analysis", "-a", "analysis_path", required=True)
def main(tessellations_path, results_path, analysis_path):
    # load all results
    results = load_results(
        results_file for results_file in glob(os.path.join(results_path, "*.torch"))
    )
    dataset = TessellationDataset(tessellations_path)
    gaps, _ = read_line_gap_infos(tessellations_path, dataset.img_size)

    # configure seaborn colors
    sns.set_palette("crest", n_colors=len(results))

    # load the precomputed analysis
    with open(analysis_path, "rb") as f:
        analysis = pickle.load(f)

    receptive_fields = analysis["receptive_fields"]
    gap_covers = analysis["gap_covers"]
    performance = analysis["performance"]

    # configure seaborn colors
    sns.set_palette("crest", n_colors=len(receptive_fields))

    # create the plots
    plot_max_results(results_path, receptive_fields, gap_covers, gaps)


if __name__ == "__main__":
    main()
