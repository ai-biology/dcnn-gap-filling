"""
Compute gap cover grouped by gap width
"""

import os

import numpy as np
import pandas as pd
from skimage.io import imread
import torch

from .utils import center_crop


def read_gap_img(basepath, img_id, line_id, gap_id):
    """ Read a gap image """
    img_path = os.path.join(basepath, f"gaps/gap_{img_id}_{line_id}_{gap_id}.png")
    gap_img = imread(img_path).astype(bool)
    if not gap_img.any():
        # No gap in img
        return None
    return gap_img


def calc_gap_cover_single_gap(pred, gap_img):
    """ Compute gap cover for single gap """
    gap_img = center_crop(gap_img, pred)
    pred_gap = pred[gap_img[None, ...]]
    return pred_gap.mean()


def predict_single(x, unet):
    """
    Segment x with U-Net and return probabilities
    """
    return torch.sigmoid(unet(x[None, ...])).detach().numpy()


def calc_gap_cover_image(img_id, dataset, unet, gaps):
    """
    Compute single gap covers for image
    """
    x, _, _ = dataset[img_id]
    out = predict_single(x, unet)

    gap_lengths = []
    gap_covers = []

    for (line_id, gap_id), gap in gaps.loc[img_id].iterrows():
        gap_img = read_gap_img(dataset.basepath, img_id, line_id, gap_id)

        if gap_img is None:
            continue

        gap_cover = calc_gap_cover_single_gap(out, gap_img)
        gap_lengths.append(gap.length_px)
        gap_covers.append(gap_cover)

    return gap_lengths, gap_covers


def compute_single_gap_covers(dataset, gaps, unet):
    """
    Evaluate gap covers for single gaps.
    :param dataset: The full tessellation dataset
    :param gaps: Dataframe describing the gaps
    :returns: Dataframe with gap lengths and gap covers
    """
    unet.eval()

    gap_lengths = []
    gap_covers = []

    for i in range(len(dataset)):
        gap_lengths_i, gap_covers_i = calc_gap_cover_image(i, dataset, unet, gaps)
        gap_lengths.extend(gap_lengths_i)
        gap_covers.extend(gap_covers_i)

    return pd.DataFrame({"length": gap_lengths, "cover": gap_covers})
