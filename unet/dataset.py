"""
Represents a dataset based on generated tessellations.
Filters can be applied
"""

import os
import re
import json

import numpy as np
import pandas as pd
from skimage.io import imread
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class TessellationDataset(Dataset):
    """ A dataset of synthetic tessellated images """

    def __init__(
        self,
        basepath,
        params_file="params.json",
        obs_pattern=r"observed_(\d+).png",
        seg_pattern=r"segmented_(\d+).png",
        gap_pattern=r"gaps_(\d+).png",
    ):
        """ Eagerly load a directory of tessellated images """
        self.basepath = basepath

        # idx -> filepath mappings
        observed = {}
        segmented = {}
        gaps = {}

        # discover files
        filenames = os.listdir(basepath)
        for filename in filenames:
            seg_match = re.match(seg_pattern, filename)
            obs_match = re.match(obs_pattern, filename)
            gap_match = re.match(gap_pattern, filename)

            filepath = os.path.join(basepath, filename)
            if seg_match:
                idx = int(seg_match[1])
                segmented[idx] = filepath
            elif obs_match:
                idx = int(obs_match[1])
                observed[idx] = filepath
            elif gap_match:
                idx = int(gap_match[1])
                gaps[idx] = filepath

        self.observed = [
            self.load_image(observed[i])[None, :, :] for i in range(len(observed))
        ]
        self.segmented = [self.load_image(segmented[i]) for i in range(len(segmented))]
        self.gaps = [self.load_image(gaps[i]).byte() for i in range(len(gaps))]

        assert len(self.observed) == len(self.segmented)
        assert len(self.observed) == len(self.gaps)

        with open(os.path.join(basepath, params_file)) as pf:
            self.params = json.load(pf)
            self.img_size = self.params["width"]

        self.n_channels = 1

    def apply_filters(self, filters: np.ndarray, decompose_fn, agg=None):
        """
        Apply the given filters to the dataset using decompose_fn
        and aggregate filtered images
        :param agg: Coefficient aggregation. Can be "max" or "mean".
        """
        for i, obs in enumerate(self.observed):
            obs = obs[:1].numpy()
            coeffs = decompose_fn(obs[0], filters)

            if agg == "max":
                coeffs = np.max(coeffs, axis=0, keepdims=True)
            elif agg == "mean":
                coeffs = np.mean(coeffs, axis=0, keepdims=True)

            obs = np.concatenate((obs, coeffs), axis=0)
            self.observed[i] = torch.Tensor(obs)

        self.n_channels = self.observed[0].shape[0]

    def clear_filters(self):
        """
        Remove any coefficients, resulting in single channel observations
        """
        for i, obs in enumerate(self.observed):
            self.observed[i] = obs[:1]
        self.n_channels = 1

    def load_image(self, filepath: str) -> torch.Tensor:
        """ Read an image and return Tensor """
        image = imread(filepath)

        # convert to float and invert
        image = 1 - (image / 255.0)
        return torch.Tensor(image)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.observed[i], self.segmented[i], self.gaps[i]
        elif isinstance(i, slice):
            return (
                torch.stack(self.observed[i]),
                torch.stack(self.segmented[i]),
                torch.stack(self.gaps[i]),
            )

        raise ValueError(f"{i} must be int or slice")

    def __len__(self):
        return len(self.observed)


def get_data_loaders(dataset, val_ratio, batch_size, test_ratio=None):
    """
    Initialize the dataset.
    :returns: tuple of DataLoaders (train_loader, val_loader)
    """
    val_len = int(len(dataset) * val_ratio)
    test_len = int(len(dataset) * test_ratio) if test_ratio is not None else 0
    train_len = len(dataset) - val_len - test_len

    if test_ratio is not None:
        train_set, val_set, test_set = random_split(
            dataset, [train_len, val_len, test_len]
        )
        test_loader = DataLoader(test_set, batch_size=batch_size, pin_memory=True)
    else:
        train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True)

    if test_ratio is not None:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def read_line_gap_infos(basepath, img_size):
    """
    Read tessellation info dataframes and compute lengths.
    """
    lines = pd.read_csv(os.path.join(basepath, "lines.csv"), index_col=(0, "line_id"))
    gaps = pd.read_csv(
        os.path.join(basepath, "gaps.csv"), index_col=(0, "line_id", "gap_id")
    )

    # exclude gaps that are completely out of bounds
    gaps["x_oob"] = (gaps[["x1", "x2"]].max(axis=1) < 0) | (
        gaps[["x1", "x2"]].min(axis=1) > img_size
    )
    gaps["y_oob"] = (gaps[["y1", "y2"]].max(axis=1) < 0) | (
        gaps[["y1", "y2"]].min(axis=1) > img_size
    )
    gaps = gaps.loc[~(gaps.x_oob | gaps.y_oob)]

    gaps["length_px"] = np.sqrt((gaps.x1 - gaps.x2) ** 2 + (gaps.y1 - gaps.y2) ** 2)
    gaps["length_manh"] = np.abs(gaps.x1 - gaps.x2) + np.abs(gaps.y1 - gaps.y2)
    gaps["length_max"] = np.maximum(
        np.abs(gaps.x1 - gaps.x2), np.abs(gaps.y1 - gaps.y2)
    )

    lines["length_px"] = np.sqrt(
        (lines.x1 - lines.x2) ** 2 + (lines.y1 - lines.y2) ** 2
    )
    lines["length_manh"] = np.abs(lines.x1 - lines.x2) + np.abs(lines.y1 - lines.y2)

    return gaps, lines
