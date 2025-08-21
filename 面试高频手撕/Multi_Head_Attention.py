import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    vanilla MHA，无偏置，与《Attention Is All You Need》保持一致
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 三个大矩阵一次性完成所有头的投影
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None):
        """
        x:     [batch, seq_len, d_model]
        mask:  [batch, 1, seq_len, seq_len] 下三角为 0 上三角为 -∞（causal）
        return [batch, seq_len, d_model]
        """
        B, S, D = x.shape
        # 1. 线性投影
        qkv = self.w_qkv(x)                       # [B, S, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)            # 3×[B, S, D]

        # 2. reshape 成多头的形状, 把D展开成head*d_k
        q = q.view(B, S, self.n_heads, self.d_k).transpose(1, 2)  # [B, h, S, d_k]
        k = k.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # 3. 缩放点积注意力: 将k最后两个维度互换
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, h, S, S]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 4. 加权求和
        out = torch.matmul(attn, v)               # [B, h, S, d_k]
        out = out.transpose(1, 2).contiguous().view(B, S, D) # 再拼回去 [B, S, D]
        out = self.w_out(out)

        # 5. 输出投影
        return out


# ===== 单元测试 =====
if __name__ == "__main__":
    B, S, D, H = 2, 5, 512, 8 # batch, seq_len, dim, head_nums
    mha = MultiHeadAttention(D, H)
    x = torch.randn(B, S, D)
    mask = torch.tril(torch.ones(S, S)).unsqueeze(0).unsqueeze(0)  # causal mask
    y = mha(x, mask)
    print(y.shape)  # 期望 [2, 5, 512]
