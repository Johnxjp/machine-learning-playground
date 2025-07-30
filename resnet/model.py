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
        self.pool = nn.AvgPool2d(kernel_size=1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        batch, input_channels, h, w = x.shape
        # Copy original channels to first part
        padded = torch.zeros(batch, self.out_channels, h, w, dtype=x.dtype, device=x.device)
        padded[:, :input_channels, :, :] = x
        return padded


class ResidualCNNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        batch_normalise: bool,
        add_activation: bool,
    ):
        """
        in_channels : Number of input channels
        out_channels : Number of output channels
        kernel : Kernel size for the convolution
        stride : Stride for the convolution
        batch_normalise : Add batch normalisation after the convolution
        add_activation : Add activation function after covolution or batch normalisation
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            bias=False,
            padding=1,
        )
        self.norm = nn.BatchNorm2d(out_channels) if batch_normalise else nn.Identity()
        self.act = nn.ReLU() if add_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):

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
        in_channels : Number of input channels
        out_channels : Number of output channels
        kernel : Kernel size for the convolution
        layers : Number of layers in the block
        stride : Stride for the first convolution
        shortcut_method : Method to handle the residual connection, either "projection" or "pad"
        batch_normalise : Whether to add batch normalisation after each convolution
        """
        super().__init__()
        assert shortcut_method in ["projection", "pad"]

        self.layers = []
        self.layers.append(
            ResidualCNNLayer(
                in_channels,
                out_channels,
                kernel=kernel,
                stride=stride,
                batch_normalise=batch_normalise,
                add_activation=True,
            )
        )

        for _ in range(1, layers - 1):
            self.layers.append(
                ResidualCNNLayer(
                    out_channels,
                    out_channels,
                    kernel=kernel,
                    stride=1,
                    batch_normalise=batch_normalise,
                    add_activation=True,
                )
            )

        # Don't add ReLU to final layer. This happens after residual is added
        self.layers.append(
            ResidualCNNLayer(
                out_channels,
                out_channels,
                kernel=kernel,
                stride=1,
                batch_normalise=batch_normalise,
                add_activation=False,
            )
        )

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

        self.final_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_projection(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_act(x + residual)


class Resnet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        blocks: int,
        input_kernel_size: int,
        block_kernel_size: int,
        block_layers: int,
        shortcut_method: str,
        n_classes: int,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels[0], kernel_size=input_kernel_size, stride=1, padding=1
        )
        self.act1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(out_channels[0])
        self.residual_blocks = []
        in_channels = out_channels[0]
        for i, out_channel in enumerate(out_channels):
            self.residual_blocks.append(
                ResidualBlock(
                    in_channels,
                    out_channel,
                    kernel=block_kernel_size,
                    layers=block_layers,
                    stride=2 if i > 0 else 1,  # First set of blocks do not change size
                    shortcut_method=shortcut_method,
                    batch_normalise=batch_norm,
                )
            )
            in_channels = out_channel
            for _ in range(blocks - 1):
                self.residual_blocks.append(
                    ResidualBlock(
                        in_channels,
                        out_channel,
                        kernel=block_kernel_size,
                        layers=block_layers,
                        stride=1,
                        shortcut_method=shortcut_method,
                        batch_normalise=batch_norm,
                    )
                )

        self.residual_blocks = nn.ModuleList(self.residual_blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor):
        x = self.act1(self.norm1(self.conv1(x)))
        for layer in self.residual_blocks:
            x = layer(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
