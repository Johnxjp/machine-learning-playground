import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        swish = x * torch.sigmoid(x)  # Set beta = 1
        return swish * self.linear2(x)
