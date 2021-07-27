#!/usr/bin/env python3

""" Analyse gap covers vs gap length for different U-Nets """

from glob import glob
import os
import pickle

import click
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, confusion_matrix
import torch
from tqdm import tqdm

from unet.dataset import TessellationDataset, read_line_gap_infos
from unet.gap_cover import predict_single, compute_single_gap_covers
from unet.receptive_field import get_rf_for_model
from unet.unet import make_unet


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


def compute_receptive_fields(results):
    """
    Compute receptive fields for models
    :returns: dict of depth -> rf
    """
    rfs = {}
    for result in results:
        # load the model
        unet = load_model(result)

        # compute receptive field
        depth = result["depth"]
        rfs[depth] = get_rf_for_model(unet)

    print("Finished computing receptive fields")
    return rfs


def test_gap_covers(results, dataset, gaps):
    """
    Test gap covers and return accumulated results
    :returns: dict of depth -> DataFrame of single gap covers
    """
    gap_covers = {}
    for result in tqdm(results, desc="Test gap covers"):
        unet = load_model(result)
        depth = result["depth"]

        gap_covers[depth] = compute_single_gap_covers(dataset, gaps, unet)

    print("Finished computing gap covers")
    return gap_covers


def evaluate_performance(results, dataset):
    """
    Evaluate performance of each unet in terms of
    confusion matrix and mean average precision across images
    """
    eval_results = []

    for result in tqdm(results, desc="Evaluate performance"):
        unet = load_model(result)
        unet.eval()

        for img_id, (x, y, _) in enumerate(dataset):
            y = y.detach().numpy()

            out = predict_single(x, unet)[0]

            avg_prec = average_precision_score(y.flatten(), out.flatten())
            tn, fp, fn, tp = confusion_matrix(
                y.flatten() > 0.5, out.flatten() > 0.5, normalize="all"
            ).flatten()

            eval_results.append(
                {
                    "depth": result["depth"],
                    "img_id": img_id,
                    "average_precision": avg_prec,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )

    return pd.DataFrame(eval_results)


@click.command()
@click.option("--tessellations", "-t", "tessellations_path", required=True)
@click.option("--results", "-r", "results_path", required=True)
def main(tessellations_path, results_path):
    # load all results
    results = load_results(
        results_file for results_file in glob(os.path.join(results_path, "*.torch"))
    )
    dataset = TessellationDataset(tessellations_path)
    gaps, _ = read_line_gap_infos(tessellations_path, dataset.img_size)

    # configure seaborn colors
    sns.set_palette("crest", n_colors=len(results))

    # analyse results
    receptive_fields = compute_receptive_fields(results)
    gap_covers = test_gap_covers(results, dataset, gaps)
    performance = evaluate_performance(results, dataset)

    # save analysis
    with open(os.path.join(results_path, "analysis.pickle"), "wb") as analysis:
        pickle.dump(
            {
                "receptive_fields": receptive_fields,
                "gap_covers": gap_covers,
                "performance": performance,
            },
            analysis,
        )


if __name__ == "__main__":
    main()
