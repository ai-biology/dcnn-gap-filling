#!/usr/bin/env python3

"""
Train U-Nets using different depths
"""
import os

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


def train_models(depths, train_loader, val_loader, test_loader, device, outpath):
    """ Train U-Nets with different depths """
    width = 3
    patience = 20

    for depth in depths:
        unet, criterion, optim = make_unet(
            depth, wf=width, padding=True, batch_norm=True, device=device
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
        test_results = test_unet(unet, criterion, test_loader, device=device)

        results = {
            "width": width,
            "depth": depth,
            "unet": unet.state_dict(),
            "train": train_results,
            "test": test_results,
        }
        torch.save(results, os.path.join(outpath, f"unet-{depth}.torch"))


@click.command()
@click.option("--datapath", "-d", required=True)
@click.option("--outpath", "-o", default="./")
@click.option("--device")
def main(datapath, outpath, device):
    """ Main """
    if device is None:
        device = select_torch_device()
    print(f"Torch device: {device}")

    dataloaders = load_dataset(datapath)
    print("Dataset loaded")

    depths = np.arange(1, 5)
    train_models(depths, *dataloaders, device=device, outpath=outpath)


if __name__ == "__main__":
    main()
