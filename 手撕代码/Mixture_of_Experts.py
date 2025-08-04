import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        gate_logits = self.gate(x)  # [B, E]
        gate_scores = F.softmax(gate_logits, dim=-1)  # [B, E]
        top1_idx = torch.argmax(gate_scores, dim=-1)  # [B]

        # Prepare output tensor
        output = torch.zeros_like(x)  # [B, input_dim]

        # Top-1 routing (for each sample only 1 expert)
        for i in range(self.num_experts):
            mask = top1_idx == i  # [B], boolean mask
            if mask.sum() == 0:
                continue
            x_i = x[mask]  # [B_i, input_dim]
            out_i = self.experts[i](x_i)  # [B_i, input_dim]
            output[mask] = out_i  # write back to output

        return output


moe = MoE(input_dim=16, hidden_dim=32, num_experts=4)
x = torch.randn(8, 16)  # batch_size=8
out = moe(x)
print(out.shape)  # [8, 16]
