"""
Misc. utilities
"""

import numpy as np
import torch


def select_torch_device(cudnn_benchmark=True):
    """ Set the torch device """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        # enable CUDNN performance optimization
        torch.backends.cudnn.benchmark = cudnn_benchmark

    return device


def center_crop(labels, out):
    """ Crop labels to match out shape """
    if labels.shape[-2:] == out.shape[-2:]:
        return labels

    lefts = []
    rights = []
    for dim in (-2, -1):
        left = (labels.shape[dim] - out.shape[dim]) // 2
        right = left + out.shape[dim]
        lefts.append(left)
        rights.append(right)

    return labels[..., lefts[0] : rights[0], lefts[1] : rights[1]]


def count_parameters(model):
    """ Count parameters of a pytorch model """
    params = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in params])
