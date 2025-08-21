import torch
import torch.nn as nn


class LayerNormManual(nn.Module):
    """
    手撕 LayerNorm（最后一个维度归一化）
    输入: (B, L, C) 或任意 shape，对最后一维做归一化
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        # x: (..., C)
        assert x.size(-1) == self.normalized_shape, "LayerNorm last dim mismatch"

        mean = x.mean(dim=-1, keepdim=True)  # [B, L, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B, L, 1]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        y = self.weight * x_norm + self.bias
        return y


class BatchNorm2dManual(nn.Module):
    """
    手撕 BatchNorm2d（对 N×H×W 做归一化）
    输入: (B, C, H, W)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # 推理阶段用 running stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        B, C, H, W = x.shape

        if self.training:
            # 计算当前 batch 的均值方差（N×H×W）
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # 计算均值 [1, C, 1, 1]
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)  # 计算方差

            # 更新 running_mean 和 running_var（用于推理）
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()

        else:
            mean = self.running_mean.view(1, C, 1, 1)
            var = self.running_var.view(1, C, 1, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = self.weight.view(1, C, 1, 1) * x_norm + self.bias.view(1, C, 1, 1)

        return x_norm


if __name__ == '__main__':
    # 假设 LayerNormManual 和 BatchNorm2dManual 已定义
    x = torch.randn(2, 4, 4)  # (B, L, C)
    ln = LayerNormManual(4)
    print("LayerNorm out:", ln(x).shape)

    y = torch.randn(3, 5, 8, 8)  # (B, C, H, W)
    bn = BatchNorm2dManual(5)
    print("BatchNorm out:", bn(y).shape)
