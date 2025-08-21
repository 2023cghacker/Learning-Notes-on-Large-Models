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

    def forward(self, x, k=2):
        B, D = x.shape

        # 1. gate
        gate_logits = self.gate(x)  # [B, E]
        gate_scores = F.softmax(gate_logits, dim=-1)
        topk_w, topk_idx = torch.topk(gate_scores, k, dim=-1)  # [B, k]

        # 2. 初始化输出
        y = torch.zeros_like(x)

        # 3. 遍历 k 个位置（而不是遍历 E 个专家）
        for pos in range(k):
            expert_idx = topk_idx[:, pos]  # [B]
            weight = topk_w[:, pos]  # [B]
            print(f"expert_idx={expert_idx}, weight={weight}")

            # 每个 token 要去的专家号
            for e in range(self.num_experts):
                mask = expert_idx == e  # [B]
                print(f"mask={mask}")
                if mask.sum() == 0:  # 如果这个专家没有一个样本激活，则跳过
                    continue
                y[mask] += weight[mask, None] * self.experts[e](x[mask])

        return y


moe = MoE(input_dim=16, hidden_dim=32, num_experts=4)  # 4个专家
x = torch.randn(8, 16)  # batch_size=8
out = moe(x, k=2)
print(out.shape)  # [8, 16]
