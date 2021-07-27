"""
Evaluation methods and test loop

Jaccard: useful to see if model learned to segment the image, but not sensitive to gap coverage
Gap Cover: extracts gaps as difference between target and observed image,
           is 1.0 if gap is completely filled
"""

import numpy as np
import torch


def calc_jaccard(
    preds: torch.Tensor, target: torch.Tensor, logits=True, pool_batch=False
):
    """
    Calculate jaccard index.
    :param logits: Set False if preds are in range [0, 1]
                   or True if preds are in range [-infty, +infty]
    :param pool_batch: If True, computes jaccard as if batch of preds are one image,
                       otherwise returns array of jaccards per image
    """
    batch_size = preds.shape[0]

    if logits:
        preds = torch.sigmoid(preds)

    true = target.reshape(batch_size, -1).bool()
    preds = preds.reshape(batch_size, -1) > 0.5

    intersection = (preds & true).sum(1)
    union = (preds | true).sum(1)

    if pool_batch:
        intersection = intersection.sum()
        union = union.sum()

    jaccard = intersection / union
    jaccard = jaccard.detach().numpy()

    if pool_batch:
        jaccard = jaccard.item()

    return jaccard


def calc_gap_cover_precomputed(preds: torch.Tensor, gaps: torch.Tensor, logits=True):
    """
    Calculate ratio of gaps covered
    :param gaps: precomputed binary mask of gaps where 1 indicates presence of gap
    """
    batch_size = preds.shape[0]

    if gaps.dim() == 2:
        gaps = gaps[None, :, :]

    assert preds.shape == gaps.shape

    if logits:
        preds = torch.sigmoid(preds)

    gap_mask = gaps.reshape(batch_size, -1)
    gap_mask = gap_mask.detach().numpy().astype(bool)
    no_gap = ~gap_mask.any(axis=1)
    preds = preds.reshape(batch_size, -1).detach().numpy()

    preds_masked = np.ma.array(preds, mask=~gap_mask)
    gap_cover = preds_masked.mean(axis=1).filled()
    gap_cover[no_gap] = 1.0
    return gap_cover
