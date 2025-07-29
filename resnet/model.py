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
        in_channels: int,
        out_channels: int,
        kernel: int,
        layers: int,
        stride: int,
        shortcut_method: str = "projection",
        batch_normalise: bool = True,
    ):
        """
        channels : Number of input channels
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
        self.layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                bias=False,
                padding=1,
            )
        )
        if batch_normalise:
            self.layers.append(nn.BatchNorm2d(out_channels))

        for i in range(1, layers):
            if i < layers - 1:
                self.layers.append(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=kernel,
                        bias=False,
                        padding=1,
                    )
                )
                if batch_normalise:
                    self.layers.append(nn.BatchNorm2d(out_channels))
                self.layers.append(nn.ReLU())
            else:
                # Don't add ReLU to final layer. This happens after residual is added
                self.layers.append(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=kernel,
                        padding=1,
                        bias=False,
                    )
                )
                if batch_normalise:
                    self.layers.append(nn.BatchNorm2d(out_channels))

        self.layers = nn.ModuleList(self.layers)
        self.residual_projection = nn.Identity()
        if out_channels > in_channels:
            if shortcut_method == "projection":
                # Reduce feature map size of input
                self.residual_projection = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=2
                )
            else:
                self.residual_projection = ShortcutPadLayer(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_projection(x)
        for layer in self.layers:
            x = layer(x)
        return nn.functional.relu(x + residual)


class Resnet(nn.Module):

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        out_channels: list[int],
        blocks: int,
        input_kernel_size: int,
        n_classes: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(
            in_channels, out_channels[0], kernel_size=input_kernel_size, stride=1, padding=1
        )
        self.act1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(out_channels[0])
        self.residual_blocks = []
        in_channels = out_channels[0]
        for c in out_channels:
            for _ in range(blocks):
                stride = 2 if c > in_channels else 1
                self.residual_blocks.append(
                    ResidualCNNLayer(in_channels, c, kernel=3, layers=2, stride=stride)
                )
                in_channels = c

        self.residual_blocks = nn.ModuleList(self.residual_blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.LazyLinear(n_classes)

    def forward(self, x: torch.Tensor):
        x = self.act1(self.norm1(self.conv1(x)))
        for layer in self.residual_blocks:
            x = layer(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.out(x)
