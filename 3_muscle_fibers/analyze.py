#!/usr/bin/env python3

""" Analyse experiments with the muscle fibre dataset """

import os
from itertools import product
import json
import pickle

import click
import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter
from skimage.util import view_as_blocks
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from unet.evaluation import calc_jaccard
from unet.muscle_fiber_dataset import load_datasets_from_tsv
from unet.receptive_field import get_rf_for_model
from unet.unet import make_unet
from unet.utils import center_crop, select_torch_device


def load_params(params_file):
    """ Load experiment params from json file """
    with open(params_file) as f:
        params = json.load(f)
    return params


def get_unaugmented_patches(dataset, patch_size):
    """
    Get the unaugmented images from a muscle fiber dataset
    and cut the images into patches
    """
    obs_patches = []
    seg_patches = []
    for obs, seg in zip(dataset.observed, dataset.segmented):
        # cut into patches
        if patch_size is not None:
            obs = view_as_blocks(obs[..., 0], patch_size).reshape(-1, 1, *patch_size)
            seg = view_as_blocks(seg[..., 0], patch_size).reshape(-1, *patch_size)
        else:
            obs = obs[None, None, ..., 0]
            seg = seg[None, ..., 0]

        # convert mask
        seg[seg == 255] = 1

        seg_patches.append(torch.Tensor(seg).float())
        obs_patches.append(torch.Tensor(obs))

    return torch.cat(obs_patches), torch.cat(seg_patches)


def get_test_loader(samples_tsv, run_id, patch_size):
    """
    Return the test loader for a given run id.
    :param patch_size: The test images are cut into patches of this size.
                       Must be divisor of the original image size.
                       Can be None to skip cutting.
    """
    *_, test_set = load_datasets_from_tsv(samples_tsv, run_id)

    x, y = get_unaugmented_patches(test_set, patch_size)
    test_set = TensorDataset(x, y)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return test_loader


def load_unet(path, depth, width, device="cpu"):
    """ Load a U-Net from this path """
    result = torch.load(path, map_location=device)

    unet = make_unet(
        depth, width, train=False, padding=True, batch_norm=True, device=device
    )
    unet.load_state_dict(result.pop("unet"))
    unet.eval()

    return unet


def measure_gaps(y_true, y_pred, threshold, min_gap_width=5):
    """
    Detect connected false negatives (gaps) in a batch of images and return their widths.
    :param min_gap_width: The minimum gap size to be reported. A max-filter of this size is
                          convolved with the image to reduce gaps smaller than this size.
    """
    y_true = y_true.astype(bool)

    # sharpen predictions
    pred_sharp = maximum_filter(
        y_pred, size=(1, min_gap_width, min_gap_width), mode="nearest"
    )

    # draw gaps where boundary was not segmented
    gaps_sharp = np.zeros_like(y_true, dtype="uint8")
    gaps_sharp[y_true & (pred_sharp <= threshold)] = 255

    # detect and measure gaps
    radii = []
    gap_contours = []

    for gap_img in gaps_sharp:
        contours, _ = cv2.findContours(gap_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        gap_contours.append(contours)

        for contour in contours:
            _, radius = cv2.minEnclosingCircle(contour)
            radii.append(radius)

    return np.array(radii), gap_contours


def evaluate_unet(unet, test_loader, device):
    """ Run several evaluation procedures for a U-Net and report results as dict """
    y_pred = []
    y_true = []

    for x, y in test_loader:
        x = x.to(device)
        out = torch.sigmoid(unet(x)).detach().cpu()
        y_pred.append(out)
        y_true.append(y)

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    y_true = center_crop(y_true, y_pred)
    jaccard = calc_jaccard(y_pred, y_true, logits=False, pool_batch=True)

    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    avg_precision = average_precision_score(y_true.flatten(), y_pred.flatten())

    precision, recall, thresholds = precision_recall_curve(
        y_true.flatten(), y_pred.flatten()
    )
    f1_score = np.divide(
        2 * recall * precision, recall + precision, where=recall + precision > 0
    )
    threshold = thresholds[np.argmax(f1_score)]

    tn, fp, fn, tp = confusion_matrix(
        y_true.flatten(),
        y_pred.flatten() > threshold,
        normalize="all",
    ).ravel()

    gaps, gap_contours = measure_gaps(y_true, y_pred, threshold)

    rf = get_rf_for_model(unet)

    return {
        "receptive_field": rf,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "average_precision": avg_precision,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "gaps": gaps,
        "gap_contours": gap_contours,
        "preds": y_pred,
    }


@click.command()
@click.option("--params", "-p", "params_file", required=True)
@click.option("--experiment", "-e", required=True)
@click.option("--samples", "-s", "samples_tsv", required=True)
@click.option("--unets", "-u", "unets_path", required=True)
@click.option("--precrop/--no-precrop", default=True)
@click.option("--device")
def main(params_file, experiment, samples_tsv, unets_path, precrop, device):
    if device is None:
        device = select_torch_device()
    print(f"Torch device: {device}")

    params = load_params(params_file)

    n_runs = params[experiment].pop("n_runs")
    depths = params[experiment].pop("depths")
    widths = params[experiment].pop("widths")
    total = n_runs * len(depths) * len(widths)

    # run analyses
    results = []
    with tqdm(total=total) as pbar:
        for run_id in range(n_runs):
            patch_size = (900, 850) if precrop else None
            test_loader = get_test_loader(samples_tsv, run_id, patch_size)

            for depth, width in product(depths, widths):
                unet_path = os.path.join(
                    unets_path, f"unet-{run_id}-depth-{depth}-width-{width}.torch"
                )
                unet = load_unet(unet_path, depth, width, device)

                result = evaluate_unet(unet, test_loader, device)
                results.append(
                    {"run_id": run_id, "depth": depth, "width": width, **result}
                )
                pbar.update()

    # save results
    with open(os.path.join(unets_path, "analysis.pickle"), "wb") as analysis:
        pickle.dump(results, analysis)


if __name__ == "__main__":
    main()
