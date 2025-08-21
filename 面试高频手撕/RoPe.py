import torch
import math


def build_rope_cache(seq_len, dim, base=10000, device=None):
    """
    生成 cos/sin 缓存
    seq_len: 最大序列长度
    dim: 每个 head 的维度 (必须是偶数)
    return: cos, sin 形状 (seq_len, dim//2)
    """
    half_dim = dim // 2
    theta = 1.0 / (
        base ** (torch.arange(0, half_dim, device=device).float() / half_dim)
    )
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("n,d->nd", pos, theta)  # (seq_len, half_dim)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """
    x: (..., seq_len, dim)
    cos, sin: (seq_len, dim/2)
    return: same shape as x
    """
    dim = x.size(-1)
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    cos = cos.unsqueeze(0)  # (1,T,D/2)
    sin = sin.unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ----------------------
# demo
if __name__ == "__main__":
    B, T, D = 2, 5, 8  # batch, seq_len, head_dim (偶数)
    x = torch.randn(B, T, D)

    cos, sin = build_rope_cache(seq_len=T, dim=D, base=10000, device=x.device)
    print("cos:", cos.shape, "sin:", sin.shape)  # (T, D/2)
    y = apply_rope(x, cos, sin)
    print("input:", x.shape, "output:", y.shape)
