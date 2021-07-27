"""
Calculate the theoretical receptive field.
Example:
>>> unet = UNet()
>>> get_rf_for_model(unet)
"""

from dataclasses import dataclass

CONV_OPS = {"Conv2d", "MaxPool2d", "AvgPool2d"}
UP_OPS = {"UpsamplingBilinear2d", "ConvTranspose2d"}


@dataclass
class Layer:
    """ Simplified layer representation """

    kernel_size: int
    stride: float

    @classmethod
    def from_conv(cls, module):
        """
        Initialize layer from downsampling operation,
        i.e. convolution or pooling
        """
        kernel_size = unpack_all_equal_tuple(module.kernel_size)
        stride = unpack_all_equal_tuple(module.stride)
        return cls(kernel_size, stride)

    @classmethod
    def from_up(cls, module):
        """
        Initialize layer from upsampling operation,
        i.e. transposed convolution or upsampling
        """
        stride = unpack_all_equal_tuple(module.stride)
        return cls(1, 1 / stride)


def unpack_all_equal_tuple(t):
    """
    Helper to unpack a tuple, requiring all entries are equal.
    If t is not a tuple, it is returned
    """
    if not isinstance(t, tuple):
        return t

    assert all(x == t[0] for x in t)
    return t[0]


def get_layers(model):
    """ Returns list of all layers that change the receptive field """
    layers = []
    for child in model.children():
        layer_name = child.__class__.__name__
        if layer_name in CONV_OPS:
            layers.append(Layer.from_conv(child))
        elif layer_name in UP_OPS:
            layers.append(Layer.from_up(child))
        else:
            layers.extend(get_layers(child))
    return layers


def compute_receptive_field(layers):
    """ Compute the receptive field for extracted conv. layers """
    rf = 1
    for layer in reversed(layers):
        rf = layer.stride * rf + (layer.kernel_size - layer.stride)
    return rf


def get_rf_for_model(model):
    """ Calculate the receptive field radius for the model """
    # go forward through the network and aggregate the layers
    layers = get_layers(model)

    # go backward through the layers and compute receptive field
    return compute_receptive_field(layers)
