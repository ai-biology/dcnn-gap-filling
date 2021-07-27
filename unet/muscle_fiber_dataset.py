""" Dataset and utilities for training on the muscle fiber dataset """

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class MuscleFiberDataset(Dataset):
    """
    A dataset of annotated muscle fiber images.

    The images are passed through an image augmentation pipeline:
      1) At init time, the images are center-cropped to remove artifacts and
         incomplete annotations at the boundaries
      2) The images converted to float and scaled to be in the range [0, 1]
         using the max from the whole dataset
      2) When getting an image, it is randomly flipped/transposed/rotated and distorted
      3) Finally a random square crop of the image is returned as pytorch tensor
    """

    def __init__(
        self,
        samples_df: pd.DataFrame,
        multiplicity: int = 5,
        pre_crop: Tuple[int, int] = (900, 1700),
        post_crop: int = 200,
        load_gaps: bool = False,
    ):
        """
        Eagerly load a directory of observations and segmentations.

        :param samples_df: DataFrame with columns observation and segmentation
                           containing absolute paths to images
        :param multiplicity: Artificially increase length of dataset by this factor to
                             allow larger batch sizes
        :param pre_crop: Crop images before image augmentation to this size
        :param post_crop: Side length of square images after image augmentation
        :param load_gaps: If True, loads gaps as specified in samples
        """
        self.multiplicity = multiplicity

        self.samples = samples_df
        self.observed = []
        self.segmented = []

        self.gaps = None
        if load_gaps:
            self.gaps = []

        # load samples
        for _, paths in self.samples.iterrows():
            obs = self.load_image(paths.observation)
            seg = self.load_image(paths.segmentation)

            obs = A.center_crop(obs, *pre_crop)
            seg = A.center_crop(seg, *pre_crop)

            self.observed.append(obs)
            self.segmented.append(seg)

            if load_gaps:
                gaps = self.load_image(paths.gaps)
                gaps = A.center_crop(gaps, *pre_crop)
                self.gaps.append(gaps)

        # normalize observations and convert to float
        self.max_value = np.max(self.observed)
        for i, obs in enumerate(self.observed):
            self.observed[i] = A.to_float(obs, self.max_value)

        self.transform = A.Compose(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Transpose(),
                A.RandomRotate90(),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.01, p=0.8),
                A.RandomCrop(post_crop, post_crop),
                ToTensorV2(),
            ]
        )

    def load_image(self, filepath: str):
        """ Read an image """
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return image[..., None]

    def __getitem__(self, i):
        if not isinstance(i, int):
            raise ValueError(f"{i} must be int")

        i = i % len(self.observed)
        obs, seg = self.observed[i], self.segmented[i]

        # image augmentation
        transformed = self.transform(image=obs, mask=seg)
        obs = transformed["image"]
        seg = transformed["mask"]

        # convert mask
        seg[seg == 255] = 1
        seg = seg[..., 0].float()

        return obs, seg

    def __len__(self):
        return len(self.observed) * self.multiplicity


def load_datasets_from_tsv(samples_tsv, run_id, **kwargs):
    """
    Read relative paths to samples from a tab-separated file,
    shuffle and split the data,
    and return train/val/test datasets.
    Additional kwargs are passed to the MuscleFiberDataset constructor.

    :param samples_tsv: Path to tsv with columns `observation` and `segmentation` containing
                        the paths to images relative to the tsv path.
                        Additionally must contain the column specified by run_id with
                        the values train/val/test, which defines the split.
    """
    # load samples and make paths absolute
    samples = pd.read_csv(samples_tsv, sep="\t")
    basepath = os.path.dirname(samples_tsv)
    samples.observation = samples.observation.map(lambda fn: os.path.join(basepath, fn))
    samples.segmentation = samples.segmentation.map(
        lambda fn: os.path.join(basepath, fn)
    )

    # perform precomputed split
    samples_train = samples.loc[samples[str(run_id)] == "train"]
    samples_val = samples.loc[samples[str(run_id)] == "val"]
    samples_test = samples.loc[samples[str(run_id)] == "test"]

    # make datasets
    train_set = MuscleFiberDataset(samples_train, **kwargs)
    val_set = MuscleFiberDataset(samples_val, **kwargs)
    test_set = MuscleFiberDataset(samples_test, **kwargs)

    return train_set, val_set, test_set
