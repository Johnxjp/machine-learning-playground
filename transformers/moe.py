import torch
import torch.nn as nn

from swiglu.swiglu import SwiGLU


class BasicExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class SparseMOE(nn.Module):
    """Following Mixtral of Experts (2022)"""

    def __init__(self, num_experts: int, top_k: int, in_features: int, out_features: int, include_noise: bool):
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [SwiGLU(in_features, out_features) for _ in range(num_experts)]
        )
        self.wg = nn.Linear(in_features, num_experts, bias=False)
        self.include_noise = include_noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x.shape = (batch_size, seq_length, in_features) """
        logits = self.wg(x)
        if self.include_noise:
            noise = torch.randn_like(logits)
            logits += noise

        topk_values, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        out = torch.full_like(logits, -torch.inf)
        out.scatter_(-1, topk_indices, topk_values)
        out = torch.softmax(out, dim=-1) # shape (batch_size, seq_length, num_experts)

        expert_values = [expert(x) for expert in self.experts]
        expert_values = torch.stack(expert_values, dim=1) # shape (batch_size, num_experts, seq_length, out_features)
        return torch.einsum("ijk,ikjl->ijl", out, expert_values) # shape (batch_size, seq_length, out_features)
