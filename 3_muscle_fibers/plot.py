#!/usr/bin/env python3

""" Plot results from the muscle fibre dataset """

import os
import pickle

import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
NARROW_FIGSIZE = (1.5, 2.0)
WIDE_FIGSIZE = (2.8, 2.0)


def plot_grouped_avg_precision(results, groupby, outpath):
    """
    Plot average precision as swarmplot with trendline
    """
    plt.figure(figsize=WIDE_FIGSIZE)

    linecolor = sns.color_palette("flare", n_colors=1)[0]
    sns.lineplot(
        data=results,
        x=results[groupby].astype(str),
        y="average_precision",
        color=linecolor,
        zorder=100,
    )

    sns.swarmplot(data=results, x=groupby, y="average_precision", size=3, zorder=0)

    plt.xlabel(groupby.title())
    plt.ylabel("Average Precision")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "average_precisions.pdf"))
    plt.savefig(os.path.join(outpath, "average_precisions.png"))


def plot_grouped_false_negatives(results, groupby, outpath):
    """ Make grouped boxplot of false negative rates """
    plt.figure(figsize=NARROW_FIGSIZE)
    ax = sns.boxplot(data=results, x=groupby, y="fn", width=0.65, **BOXPLOT_STYLE)

    sns.despine(offset=1, trim=True)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel(groupby.title())
    plt.ylabel("False Negative Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "false_negative_rate.pdf"))
    plt.savefig(os.path.join(outpath, "false_negative_rate.png"))


def plot_grouped_false_positives(results, groupby, outpath):
    """ Make grouped boxplot of false positive rates """
    plt.figure(figsize=NARROW_FIGSIZE)
    ax = sns.boxplot(data=results, x=groupby, y="fp", width=0.65, **BOXPLOT_STYLE)

    sns.despine(offset=1, trim=True)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel(groupby.title())
    plt.ylabel("False Positive Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "false_positive_rate.pdf"))
    plt.savefig(os.path.join(outpath, "false_positive_rate.png"))


def plot_grouped_gap_hist(results, groupby, outpath, highlight):
    """
    Plot line histograms of the grouped results, aggregated across runs
    """
    # accumulate gap lengths
    gap_lengths = results.groupby(groupby)["gaps"].apply(
        lambda gaps: np.concatenate(gaps.values)
    )
    gap_lengths = gap_lengths.explode().astype(float).reset_index()
    bins = np.logspace(
        np.log10(gap_lengths.gaps.min()), np.log10(gap_lengths.gaps.max()), 50
    )

    if highlight:
        first_color = sns.color_palette("flare", n_colors=1)
        other_colors = sns.color_palette("crest", n_colors=results[groupby].nunique())
        palette = first_color + other_colors[1:]
    else:
        palette = "crest"

    plt.figure(figsize=WIDE_FIGSIZE)
    sns.histplot(
        data=gap_lengths,
        x="gaps",
        hue=groupby,
        bins=bins,
        element="step",
        palette=palette,
        fill=False,
    )

    plt.xscale("log")
    plt.xlabel("Gap Width")
    plt.ylabel("# Gaps")

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "gap_histogram.pdf"))
    plt.savefig(os.path.join(outpath, "gap_histogram.png"))


@click.command()
@click.option("--analysis-path", "-a", required=True)
@click.option("--experiment", "-e", required=True)
@click.option("--outpath", "-o", required=True)
@click.option("--highlight-gap-hist", default=False, is_flag=True)
def main(analysis_path, experiment, outpath, highlight_gap_hist):
    # load precomputed analysis results
    with open(analysis_path, "rb") as analysis:
        results = pickle.load(analysis)
    results = pd.DataFrame(results)

    # figure out how to group results
    if "width" in experiment:
        groupby = "width"
    else:
        groupby = "depth"

    # configure seaborn colors
    sns.set_palette("crest", n_colors=results[groupby].nunique())

    # create output folder
    os.makedirs(outpath, exist_ok=True)

    # generate the plots
    plot_grouped_avg_precision(results, groupby, outpath)
    plot_grouped_false_negatives(results, groupby, outpath)
    plot_grouped_false_positives(results, groupby, outpath)
    plot_grouped_gap_hist(results, groupby, outpath, highlight=highlight_gap_hist)


if __name__ == "__main__":
    main()
