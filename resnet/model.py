"""
Implementation according to Deep Residual Learning for Image Recognition He. et al (2015)
https://arxiv.org/pdf/1512.03385
"""

import torch
import torch.nn as nn


class ShortcutPadLayer(nn.Module):
    """Pad the channels with zeros when shortcut"""

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.avg_pool2d(x, kernel_size=1, stride=2)
        batch, channels, h, w = x.shape
        padded = torch.zeros(batch, self.out_channels, h, w, dtype=x.dtype, device=x.device)
        # Copy original channels to first part
        padded[:, :channels, :, :] = x
        return padded


class ResidualCNNLayer(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel: int,
        layers: int,
        downsample: bool = False,
        shortcut_method: str = "projection",
        batch_normalise: bool = True,
    ):
        """
        in_channel : Number of input channels
        out_channels : Number of output channels
        kernel : Kernel size for the convolution
        layers : Number of convolution layers to stack
        downsample : Reduce the feature size map by half applying stride of 2
        shortcut_method : Method for the shortcut connection when dimension changes, either "projection" or "pad"
        """
        super().__init__()
        self.n_layers = layers

        assert shortcut_method in ["projection", "pad"]
        self.shortcut_method = shortcut_method
        self.layers = []

        # First layer
        in_channels = channels
        channels = channels * 2 if downsample else channels
        self.layers.append(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=kernel,
                stride=2 if downsample else 1,
                bias=False,
                padding=1,
            )
        )
        if batch_normalise:
            self.layers.append(nn.BatchNorm2d(channels))

        for i in range(1, layers):
            if i < layers - 1:
                self.layers.append(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel,
                        bias=False,
                        padding=1,
                    )
                )
                if batch_normalise:
                    self.layers.append(nn.BatchNorm2d(channels))
                self.layers.append(nn.ReLU())
            else:
                # Don't add ReLU to final layer. This happens after residual is added
                self.layers.append(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel,
                        padding=1,
                        bias=False,
                    )
                )
                if batch_normalise:
                    self.layers.append(nn.BatchNorm2d(channels))

        self.layers = nn.ModuleList(self.layers)
        self.residual_projection = nn.Identity()
        if downsample:
            if shortcut_method == "projection":
                # Reduce feature map size of input
                self.residual_projection = nn.Conv2d(in_channels, channels, kernel_size=1, stride=2)
            else:
                self.residual_projection = ShortcutPadLayer(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_projection(x)
        for layer in self.layers:
            x = layer(x)
        return nn.functional.relu(x + residual)


class Resnet(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):

        return x
