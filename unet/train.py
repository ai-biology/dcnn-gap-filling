"""
Training procedure for unet
"""

import copy
from itertools import count

import numpy as np
import torch
from torch import nn

from .evaluation import calc_jaccard, calc_gap_cover_precomputed
from .utils import center_crop


def train_epoch(unet, criterion, optim, train_loader, device="cpu"):
    """ Train the unet for one epoch """
    unet.train()

    running_loss = 0
    grad_norm = 0
    n_samples = 0

    for x, y, *_ in train_loader:
        x = x.to(device)
        y = y.to(device)

        out = unet(x)

        y = center_crop(y, out)
        loss = criterion(out, y)

        optim.zero_grad()
        loss.backward()

        grad_norm += nn.utils.clip_grad_norm_(unet.parameters(), np.infty).item()

        optim.step()

        running_loss += loss.item()
        n_samples += len(x)

    running_loss /= n_samples
    grad_norm /= n_samples

    return running_loss, grad_norm


def val_epoch(unet, criterion, val_loader, eval_gap_covers=True, device="cpu"):
    """ Validate the unet using criterion as loss and val_loader """
    unet.eval()

    running_loss = 0
    n_samples = 0

    jaccards = []
    gap_covers = []

    for batch in val_loader:
        if eval_gap_covers:
            x, y, gaps = batch
        else:
            x, y, *_ = batch

        x = x.to(device)
        y = y.to(device)

        out = unet(x)

        y = center_crop(y, out)
        loss = criterion(out, y)

        running_loss += loss.item()
        n_samples += len(x)

        out = out.cpu()
        y = y.cpu()
        jaccards.append(calc_jaccard(out, y.bool()))

        if eval_gap_covers:
            gaps_crop = center_crop(gaps, out)
            gap_covers.append(calc_gap_cover_precomputed(out, gaps_crop))

    mean_jaccard = np.concatenate(jaccards).mean()
    mean_loss = running_loss / n_samples

    if eval_gap_covers:
        mean_gap_cover = np.concatenate(gap_covers).mean()
    else:
        mean_gap_cover = None

    return mean_loss, mean_jaccard, mean_gap_cover


def log_result(**result):
    """ Print kwargs """
    if "epoch" in result and result["epoch"] == 0:
        # print header in first epoch
        header = " | ".join(result.keys())
        print(header)

    msg = " | ".join(
        format(v, ".4f") if isinstance(v, float) else str(v) for v in result.values()
    )
    print(msg)


def training_loop(
    patience,
    unet,
    criterion,
    optim,
    train_loader,
    val_loader,
    eval_gap_covers=True,
    verbose=False,
    device="cpu",
):
    """
    Main training loop for unet, with early stopping.
    :param patience: epochs of no improvement before stopping early
    :returns: results as epoch list
    """
    results = []
    best_epoch = None
    best_model = None

    for epoch in count():
        train_loss, grad_norm = train_epoch(
            unet, criterion, optim, train_loader, device
        )

        with torch.no_grad():
            val_loss, jaccard, gap_cover = val_epoch(
                unet,
                criterion,
                val_loader,
                eval_gap_covers=eval_gap_covers,
                device=device,
            )

        results.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "grad_norm": grad_norm,
                "val_loss": val_loss,
                "jaccard": jaccard,
            }
        )
        if eval_gap_covers:
            results[-1]["gap_cover"] = gap_cover

        if verbose:
            log_result(**results[-1])

        if best_epoch is None:
            # first round
            best_epoch = 0
            continue

        best_val_loss = results[best_epoch]["val_loss"]
        if val_loss < best_val_loss:
            # new best
            best_epoch = epoch
            best_model = copy.deepcopy(unet.state_dict())
        elif epoch - best_epoch > patience:
            # stop
            break

    unet.load_state_dict(best_model)
    return results


def test_unet(unet, criterion, test_loader, device):
    """ Test a unet and return results as dict """
    unet.to(device)
    unet.eval()

    loss, jaccard, gap_cover = val_epoch(
        unet, criterion=criterion, val_loader=test_loader, device=device
    )
    return {"loss": loss, "jaccard": jaccard, "gap_cover": gap_cover}
