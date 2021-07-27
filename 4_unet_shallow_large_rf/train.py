#!/usr/bin/env python3

"""
Train shallow and deep U-Net
"""

import click
import numpy as np
import torch

from unet.dataset import TessellationDataset, get_data_loaders
from unet.unet import make_unet
from unet.train import training_loop, test_unet
from unet.utils import select_torch_device


def load_dataset(basepath):
    """
    Load the tessellation dataset and return dataloaders
    """
    dataset = TessellationDataset(basepath)
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset, val_ratio=0.2, test_ratio=0.2, batch_size=32
    )
    return train_loader, val_loader, test_loader


def train_models(architectures, train_loader, val_loader, test_loader, device):
    """ Train U-Nets with different depths """
    patience = 20

    for condition in architectures:
        label = condition["label"]
        width = condition["width"]
        depth = condition["depth"]
        kernel_size = condition["kernel_size"]

        unet, criterion, optim = make_unet(
            depth, width, kernel_size, padding=True, batch_norm=True, device=device
        )

        train_results = training_loop(
            patience,
            unet,
            criterion,
            optim,
            train_loader,
            val_loader,
            verbose=True,
            device=device,
        )
        test_results = test_unet(unet, criterion, test_loader, device)

        results = {
            "width": width,
            "depth": depth,
            "kernel_size": kernel_size,
            "unet": unet.state_dict(),
            "train": train_results,
            "test": test_results,
        }
        torch.save(results, f"unet-{label}.torch")


@click.command()
@click.option("--datapath", "-d", required=True)
@click.option("--device")
def main(datapath, device):
    """ Main """
    if device is None:
        device = select_torch_device()
    print(f"Torch device: {device}")

    dataloaders = load_dataset(datapath)
    print("Dataset loaded")

    deep_params = {"label": "deep", "width": 3, "depth": 3, "kernel_size": 3}
    shallow_params = {"label": "shallow", "width": 5, "depth": 1, "kernel_size": 15}
    train_models([deep_params, shallow_params], *dataloaders, device)


if __name__ == "__main__":
    main()
