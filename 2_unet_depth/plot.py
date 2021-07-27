#!/usr/bin/env python3

""" Plot gap covers vs gap length for different U-Nets """

from glob import glob
import os
import pickle

import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
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
    """ Load the trained models and results """
    results = [torch.load(path, map_location=torch.device("cpu")) for path in paths]
    results.sort(key=lambda r: r["depth"])
    return results


def load_model(result):
    """ Return the trained U-Net from a result dict """
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

    plt.figure(figsize=GAPPLOT_SIZE)

    # plot the histogram
    hist_color = sns.color_palette("flare", n_colors=1)
    plt.hist(gaps.length_manh, bins=bins, color=hist_color, alpha=0.3, zorder=1)
    plt.ylabel("# Gaps")
    plt.xlabel("Gap Width")

    plt.twinx()

    for depth, marker in zip(sorted(receptive_fields.keys()), MARKERS):
        # bin the gap widths
        gap_covers_at_depth = gap_covers[depth]
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
            label=f"Depth: {depth}, RF: {receptive_fields[depth]:.0f}",
        )

    plt.xscale("log")
    plt.ylabel("Max. Gap Cover")
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "gap_length_vs_max_cover.pdf"))
    plt.savefig(os.path.join(outpath, "gap_length_vs_max_cover.png"))


def plot_average_precision(outpath, eval_results):
    """ Plot boxplot of average precisions """
    depths = eval_results.depth.unique()

    plt.figure(figsize=BOXPLOT_SIZE)
    ax = sns.boxplot(
        data=eval_results,
        x="depth",
        y="average_precision",
        **BOXPLOT_STYLE,
    )
    sns.despine(offset=1, trim=True)

    # draw significance asterisks
    for d1, d2 in zip(depths[:-1], depths[1:]):
        lower = eval_results.query(f"depth == {d1}")["average_precision"]
        upper = eval_results.query(f"depth == {d2}")["average_precision"]
        pvalue = wilcoxon(lower, upper, alternative="less").pvalue

        if pvalue > 1e-3:
            # in plotting we assume significance, raise error if ot actually sign.
            raise ValueError(f"{d1}-{d2} not significant with p={pvalue}")

        ax.plot([d1 - 0.89, d2 - 1.11], [1.01, 1.01], color="black", lw=1.5)
        ax.text(d1 - 1 + 0.5, 1.015, "✱", va="baseline", ha="center", fontsize=5)

    plt.ylim(top=1.03)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel("Depth")
    plt.ylabel("Average Precision")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "average_precisions.pdf"))
    plt.savefig(os.path.join(outpath, "average_precisions.png"))


def plot_fpr_fnr(outpath, eval_results):
    """ Plot boxplots of false positive/negative rate """
    plt.figure(figsize=BOXPLOT_SIZE)
    ax = sns.boxplot(
        data=eval_results,
        x="depth",
        y="fp",
        **BOXPLOT_STYLE,
    )
    sns.despine(offset=1, trim=True)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel("Depth")
    plt.ylabel("False Positive Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "false_positive_rate.pdf"))
    plt.savefig(os.path.join(outpath, "false_positive_rate.png"))

    plt.figure(figsize=BOXPLOT_SIZE)
    ax = sns.boxplot(
        data=eval_results,
        x="depth",
        y="fn",
        **BOXPLOT_STYLE,
    )
    sns.despine(offset=1, trim=True)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel("Depth")
    plt.ylabel("False Negative Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "false_negative_rate.pdf"))
    plt.savefig(os.path.join(outpath, "false_negative_rate.png"))


def plot_example(
    outpath, dataset, receptive_fields, img_id, rf_center, margin, result_1, result_2
):
    """ Compare result_1 and result_2 by example """
    img, _, _ = dataset[img_id]
    unet_small = load_model(result_1).eval()
    rf_small = receptive_fields[result_1["depth"]]
    unet_large = load_model(result_2).eval()
    rf_large = receptive_fields[result_2["depth"]]

    out_small = predict_single(img, unet_small)
    out_large = predict_single(img, unet_large)

    # get highlighting colors
    c1, c2 = sns.color_palette("flare", n_colors=2)

    # convert to RGB
    img = np.moveaxis(np.repeat(1 - img.numpy(), 3, 0), 0, -1)
    out_small = np.moveaxis(np.repeat(1 - out_small, 3, 0), 0, -1)
    out_large = np.moveaxis(np.repeat(1 - out_large, 3, 0), 0, -1)

    # draw rf
    x1, x2 = rf_center[1] - rf_small // 2, rf_center[1] + rf_small // 2
    y1, y2 = rf_center[0] - rf_small // 2, rf_center[0] + rf_small // 2
    out_small[draw.rectangle_perimeter((x1, y1), (x2, y2))] = c1
    img[draw.rectangle_perimeter((x1, y1), (x2, y2))] = c1

    x1, x2 = rf_center[1] - rf_large // 2, rf_center[1] + rf_large // 2
    y1, y2 = rf_center[0] - rf_large // 2, rf_center[0] + rf_large // 2
    out_large[draw.rectangle_perimeter((x1, y1), (x2, y2))] = c2
    img[draw.rectangle_perimeter((x1, y1), (x2, y2))] = c2

    # crop the images
    x1, x2 = rf_center[1] - margin, rf_center[1] + margin
    y1, y2 = rf_center[0] - margin, rf_center[0] + margin
    img = img[x1:x2, y1:y2]
    out_small = out_small[x1:x2, y1:y2]
    out_large = out_large[x1:x2, y1:y2]

    _, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=EXAMPLE_SIZE, tight_layout=True)

    ax0.imshow(img, interpolation="none", vmin=0.0, vmax=1.0)
    ax1.imshow(out_small, interpolation="none", vmin=0.0, vmax=1.0)
    ax2.imshow(out_large, interpolation="none", vmin=0.0, vmax=1.0)

    for ax in (ax0, ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(os.path.join(outpath, "gap_filling_example.pdf"))
    plt.savefig(os.path.join(outpath, "gap_filling_example.png"))


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

    # create the plots
    plot_max_results(results_path, receptive_fields, gap_covers, gaps)
    plot_average_precision(results_path, performance)
    plot_fpr_fnr(results_path, performance)
    plot_example(
        results_path,
        dataset,
        receptive_fields,
        img_id=2,
        rf_center=(90, 198),
        margin=40,
        result_1=results[1],
        result_2=results[2],
    )


if __name__ == "__main__":
    main()
