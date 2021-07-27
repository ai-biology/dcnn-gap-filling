#!/usr/bin/env python3

"""
Train U-Nets on muscle fiber dataset
with different widths
"""

from itertools import product
import json
import os

import click
import numpy as np
import torch
from torch.utils.data import DataLoader

from unet.muscle_fiber_dataset import load_datasets_from_tsv
from unet.unet import make_unet
from unet.train import training_loop
from unet.utils import select_torch_device


def load_dataset(samples_tsv, run_id, device, batch_size, img_size=200):
    """
    Load the muscle fiber dataset and return train and val dataloaders
    and the pos_weight for the BCE criterion
    """
    pin_memory = device != "cpu"

    train_set, val_set, _ = load_datasets_from_tsv(
        samples_tsv,
        run_id=run_id,
        post_crop=img_size,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        pin_memory=pin_memory,
    )

    neg = (np.stack(train_set.segmented) == 0).sum()
    pos = (np.stack(train_set.segmented) > 0).sum()
    pos_weight = neg / pos

    return train_loader, val_loader, pos_weight


def train_models(
    outdir,
    run_id,
    patience,
    depths,
    widths,
    train_loader,
    val_loader,
    pos_weight,
    device,
):
    """ Train U-Nets with different depths and widths """
    for depth, width in product(depths, widths):
        print(
            f"Run {run_id}: Start training with depth={depth} and width={width}",
            flush=True,
        )
        unet, criterion, optim = make_unet(
            depth,
            wf=width,
            padding=True,
            batch_norm=True,
            pos_weight=pos_weight,
            device=device,
        )

        train_results = training_loop(
            patience,
            unet,
            criterion,
            optim,
            train_loader,
            val_loader,
            eval_gap_covers=False,
            verbose=True,
            device=device,
        )

        results = {
            "run_id": run_id,
            "width": width,
            "depth": depth,
            "unet": unet.state_dict(),
            "train": train_results,
        }
        torch.save(results, f"{outdir}/unet-{run_id}-depth-{depth}-width-{width}.torch")


@click.command()
@click.option("--samples-tsv", "-s", required=True)
@click.option("--params", "-p", required=True)
@click.option(
    "--experiment",
    "-e",
    required=True,
    help="Key in params to use for experiment parameters",
)
@click.option(
    "--skip-runs",
    default=0,
    help="Skip runs from the start, useful to train more networks",
)
@click.option("--run-ids", "-r", default=None)
@click.option("--device")
def main(samples_tsv, params, experiment, skip_runs, run_ids, device):
    """ Main """
    if device is None:
        device = select_torch_device()
    print(f"Torch device: {device}")

    with open(params) as f:
        params = json.load(f)
    n_runs = params[experiment].pop("n_runs")

    outdir = f"unets-{experiment}"
    os.makedirs(outdir, exist_ok=True)

    if run_ids is None:
        run_ids = range(skip_runs, n_runs)
    else:
        run_ids = [int(r) for r in run_ids.split(",")]

    for run_id in run_ids:
        print(f"Start run {run_id}")
        train_loader, val_loader, pos_weight = load_dataset(
            samples_tsv, run_id=run_id, device=device, **params["dataset"]
        )
        print(f"Run {run_id}: Dataset loaded")

        train_models(
            outdir=outdir,
            run_id=run_id,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weight=pos_weight,
            device=device,
            **params[experiment],
        )


if __name__ == "__main__":
    main()
