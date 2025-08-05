import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=32, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha # 可以控制Lora对原始权重的影响程度
        self.scaling = alpha / rank # 缩放因子

        # 低秩矩阵 A 和 B (A: 随机初始化, B: 零初始化)
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)


        # 初始化权重
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        # 计算低秩适应: h = x * A * B * scaling
        x = self.A(x)
        x = self.B(x)
        return x * self.scaling


class LinearWithLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=32, dropout=0.1):
        super().__init__()
        # 原始线性层 (通常会冻结)
        self.linear = nn.Linear(in_features, out_features)
        # LoRA 适配层
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)

        # 默认冻结原始线性层
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 原始输出 + LoRA 输出
        return self.linear(x) + self.lora(x)


# 测试代码
if __name__ == "__main__":
    # 创建一个带 LoRA 的线性层
    lora_linear = LinearWithLoRA(in_features=512, out_features=256, rank=8)

    # 生成随机输入
    x = torch.randn(32, 512)  # 32个样本，每个512维

    # 前向传播
    output = lora_linear(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 查看可训练参数数量
    total_params = sum(p.numel() for p in lora_linear.parameters())
    trainable_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    print(f"可训练参数/总参数: {trainable_params}/{total_params}")  # 仅 LoRA 部分的参数
    print(f"可训练比例: {trainable_params / total_params:.2%}")


"""
知识点补充：
矩阵 A 的初始化：随机正态分布
    原因：
    矩阵 A 负责将高维输入投影到低维空间（秩为 rank），需要引入随机初始化的 "扰动" 来捕捉新任务的特征。
    使用较小的标准差（0.01）是为了避免初始扰动过大，防止干扰预训练模型的原有知识。
    随机初始化确保模型有足够的表达能力去学习新任务的模式。
矩阵 B 的初始化：全零初始化
    原因：
    这是 LoRA 最重要的设计之一。当 B 初始化为全零时，LoRA 模块的初始输出为 0（因为 A×B=0），此时模型等价于完全使用预训练权重，保证了训练起点的稳定性。
    这种 "零初始化" 使得模型在训练初期不会破坏预训练模型学到的知识，微调过程是 "温和地" 引入新任务信息。
    随着训练进行，B 会逐渐学习到适配新任务的映射，而 A 则提供多样化的低维特征空间。
"""
