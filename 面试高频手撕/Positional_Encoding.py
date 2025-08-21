import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 预先计算好 [max_len, d_model] 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维 cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)  # 不训练，固定参数

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len]
        print(pe)
        return x + pe


# -------------------------
# demo
if __name__ == "__main__":
    x = torch.zeros(2, 10, 16)  # (B=2, T=10, d_model=16)
    pe = PositionalEncoding(d_model=16, max_len=100)
    y = pe(x)
    print(y.shape)  # torch.Size([2, 10, 16])
