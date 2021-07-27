"""
The standard UNet for medical image segmentation

Adapted from https://github.com/jvanvugt/pytorch-unet
"""

import torch
from torch import nn


def make_unet(
    depth,
    wf,
    kernel_size=3,
    in_channels=1,
    padding=False,
    batch_norm=False,
    train=True,
    lr=1e-3,
    weight_decay=0,
    pos_weight=None,
    device="cpu",
):
    """ Helper function to easily build a UNet + BCELoss + ADAM """
    unet = UNet(
        in_channels=in_channels,
        depth=depth,
        wf=wf,
        kernel_size=kernel_size,
        batch_norm=batch_norm,
        up_mode="upconv",
        padding=padding,
    ).to(device)

    if train:
        if pos_weight is not None:
            pos_weight = torch.Tensor([pos_weight])

        criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight).to(
            device
        )
        optim = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=weight_decay)
        return unet, criterion, optim
    else:
        return unet


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 1,
        depth: int = 5,
        wf: int = 6,
        kernel_size: int = 3,
        padding: bool = False,
        batch_norm: bool = False,
        up_mode: str = "upconv",
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ("upconv", "upsample")

        self.padding = padding
        self.depth = depth

        self.down_path = nn.ModuleList()
        prev_channels = in_channels
        for i in range(depth - 1):
            self.down_path.append(
                UNetConvBlock(
                    prev_channels, 2 ** (wf + i), kernel_size, padding, batch_norm
                )
            )
            self.down_path.append(nn.MaxPool2d(2))
            prev_channels = 2 ** (wf + i)

        # bottom down conv
        self.down_path.append(
            UNetConvBlock(
                prev_channels, 2 ** (wf + depth - 1), kernel_size, padding, batch_norm
            )
        )
        prev_channels = 2 ** (wf + depth - 1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(
                    prev_channels,
                    2 ** (wf + i),
                    kernel_size,
                    up_mode,
                    padding,
                    batch_norm,
                )
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i % 2 == 0 and i != len(self.down_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x).reshape(-1, *x.shape[-2:])


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=int(padding))
        )
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(
            nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=int(padding))
        )
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(
            in_size, out_size, kernel_size, padding, batch_norm
        )

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
