import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention的核心思想是将查询（Query）分组，
    每个组内的查询共享一部分计算资源，从而减少计算量。这在处理大规模数据时尤其有用，可以显著降低计算复杂度。
    """

    def __init__(self, hidden_size, num_q_heads, num_kv_heads, head_dim):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.num_q = num_q_heads
        self.num_kv = num_kv_heads
        self.head_dim = head_dim
        self.group_factor = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)

    def forward(self, x):
        B, T, _ = x.size()
        q = self.q_proj(x).view(B, T, self.num_q, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv, self.head_dim)

        # 扩展K/V头到与Q相同
        k = k.repeat_interleave(self.group_factor, dim=2)  # (B,T,Hq,D)
        v = v.repeat_interleave(self.group_factor, dim=2)

        # (B,H,T,D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = F.softmax(attn, dim=-1)  # (B,H,T,T)
        out = torch.matmul(attn, v)  # (B,H,T,D)

        out = out.permute(0, 2, 1, 3).reshape(B, T, self.num_q * self.head_dim)
        return self.o_proj(out)


# demo
if __name__ == "__main__":
    x = torch.randn(2, 16, 512)
    gqa = GroupedQueryAttention(
        hidden_size=512, num_q_heads=8, num_kv_heads=2, head_dim=64
    )
    y = gqa(x)
    print(y.shape)  # (2, 16, 512)
