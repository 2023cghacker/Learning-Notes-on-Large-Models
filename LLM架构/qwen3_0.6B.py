import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    RMSNorm归一化层，用于Qwen模型中替代传统的LayerNorm
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均方根
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class MaskedGroupedQueryAttention(nn.Module):
    """
    带掩码的分组查询注意力机制，Qwen模型的核心组件
    """

    def __init__(
        self, embed_dim: int, num_heads: int, num_groups: int, max_seq_len: int = 4192
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_groups = num_groups  # 分组数

        # 确保维度合法
        assert self.head_dim * num_heads == embed_dim, "嵌入维度必须能被头数整除"
        assert num_heads % num_groups == 0, "头数必须能被分组数整除"

        # Q/K/V投影
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 旋转位置编码（RoPE）缓存
        self.register_buffer(
            "rope", self.build_rope_cache(max_seq_len, self.head_dim), persistent=False
        )

    def build_rope_cache(self, max_seq_len: int, head_dim: int):
        """构建RoPE位置编码缓存"""
        theta = 10000 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        seq_pos = torch.arange(max_seq_len, dtype=torch.float32)[:, None]
        # 计算频率
        freqs = seq_pos * theta
        # 堆叠余弦和正弦分量
        rope_cache = torch.stack([freqs.cos(), freqs.sin()], dim=-1)
        return rope_cache

    def apply_rope(self, x: torch.Tensor, seq_len: int):
        """应用RoPE位置编码到查询和键"""
        # x形状: [B, groups, heads_per_group, seq_len, head_dim]
        B, G, H, T, D = x.shape

        # 分离实部和虚部
        x_real, x_imag = x[..., : D // 2], x[..., D // 2 :]

        # 获取对应的RoPE编码
        rope = self.rope[:seq_len]  # [T, D//2, 2]
        cos, sin = rope[..., 0], rope[..., 1]  # [T, D//2]

        # 应用旋转
        x_rot_real = x_real * cos - x_imag * sin
        x_rot_imag = x_real * sin + x_imag * cos

        # 合并回原形状
        return torch.cat([x_rot_real, x_rot_imag], dim=-1)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        B, L, E = x.shape  # [batch, seq_len, embed_dim]

        # 计算QKV并拆分
        qkv = (
            self.qkv_proj(x)
            .reshape(B, L, 3, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        q, k, v = qkv.unbind(1)  # [B, num_heads, L, head_dim]

        # 按组划分注意力头
        heads_per_group = self.num_heads // self.num_groups
        q = q.reshape(B, self.num_groups, heads_per_group, L, self.head_dim)
        k = k.reshape(B, self.num_groups, heads_per_group, L, self.head_dim)
        v = v.reshape(B, self.num_groups, heads_per_group, L, self.head_dim)

        # 应用RoPE位置编码
        q = self.apply_rope(q, L)
        k = self.apply_rope(k, L)

        # 计算注意力分数 (缩放点积)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        # 应用注意力掩码
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # 计算注意力概率
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 加权求和
        attn_output = attn_probs @ v  # [B, groups, heads_per_group, L, head_dim]

        # 重组注意力头并投影输出
        attn_output = attn_output.transpose(2, 3).reshape(B, L, E)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """前馈神经网络，Qwen模型中的MLP组件"""

    def __init__(self, embed_dim: int, hidden_dim: int, act_layer=nn.SiLU):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class QwenTransformerLayer(nn.Module):
    """Qwen模型的Transformer层"""

    def __init__(
        self, embed_dim: int, num_heads: int, num_groups: int, hidden_dim: int
    ):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MaskedGroupedQueryAttention(embed_dim, num_heads, num_groups)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # 自注意力分支 (残差连接)
        x = x + self.attn(self.norm1(x), attn_mask)

        # 前馈分支 (残差连接)
        x = x + self.ffn(self.norm2(x))
        return x


class QwenModel(nn.Module):
    """完整的Qwen模型实现"""

    def __init__(
        self,
        vocab_size: int = 151200,  # Qwen的词汇表大小
        embed_dim: int = 1024,  # 嵌入维度
        num_layers: int = 28,  # Transformer层数
        num_heads: int = 16,  # 注意力头数
        num_groups: int = 2,  # 分组查询注意力的组数
        hidden_dim: int = 3072,  # 前馈网络隐藏层维度
        max_seq_len: int = 4192,  # 最大序列长度
        eos_token_id: int = 151199,  # EOS token ID，默认设为词汇表最后一个
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id  # 在初始化时设置EOS token ID

        # Token嵌入层
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformer层堆叠
        self.layers = nn.ModuleList(
            [
                QwenTransformerLayer(embed_dim, num_heads, num_groups, hidden_dim)
                for _ in range(num_layers)
            ]
        )

        # 最终归一化和输出层
        self.final_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向传播
        input_ids: 输入token的ID，形状为[batch_size, seq_len]
        attn_mask: 注意力掩码，形状为[batch_size, 1, seq_len, seq_len]
        """
        B, L = input_ids.shape

        # 检查序列长度是否超过最大限制
        if L > self.max_seq_len:
            raise ValueError(f"输入序列长度({L})超过模型最大限制({self.max_seq_len})")

        # Token嵌入
        x = self.token_embedding(input_ids)  # [B, L, embed_dim]

        # 构建因果掩码（如果没有提供）
        if attn_mask is None:
            device = x.device
            # 上三角矩阵作为掩码，使得每个位置只能关注自身和前面的位置
            attn_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=device), diagonal=1
            )
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        # 通过所有Transformer层
        for layer in self.layers:
            x = layer(x, attn_mask)

        # 最终处理和输出
        x = self.final_norm(x)
        logits = self.head(x)  # [B, L, vocab_size]
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        批量生成函数，支持贪心解码
        """
        self.eval()  # 切换到评估模式
        with torch.no_grad():
            # 跟踪每个序列是否已生成EOS token
            batch_size = input_ids.shape[0]
            finished = torch.zeros(
                batch_size, dtype=torch.bool, device=input_ids.device
            )

            for _ in range(max_new_tokens):
                # 检查是否所有序列都已生成EOS
                if finished.all():
                    break

                # 获取当前输出
                logits = self(input_ids)
                # 取最后一个token的logits
                next_token_logits = logits[:, -1, :] / temperature
                # 应用softmax获取概率
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                # 贪心选择概率最高的token
                next_token_id = torch.argmax(next_token_probs, dim=-1, keepdim=True)

                # 对于已完成的序列，不再改变其输出
                next_token_id = torch.where(
                    finished.unsqueeze(1),  # [batch_size, 1]
                    torch.zeros_like(next_token_id),  # 已完成的序列填充0或保持不变
                    next_token_id,
                )

                # 拼接结果
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)

                # 更新已完成序列的标记
                finished = finished | (next_token_id.squeeze(1) == self.eos_token_id)

        self.train()  # 切换回训练模式
        return input_ids


# 示例用法
if __name__ == "__main__":
    # 创建一个Qwen 0.6B模型实例
    model = QwenModel(
        vocab_size=151200,
        embed_dim=1024,
        num_layers=28,
        num_heads=16,
        num_groups=2,
        hidden_dim=3072,
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f} M")

    # 模拟输入 (batch_size=2, sequence_length=32)
    input_ids = torch.randint(0, 151200, (2, 32))

    # 前向传播
    output_logits = model(input_ids)
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {output_logits.shape}")  # 应该是 [2, 32, 151200]

    # 生成示例
    generated_ids = model.generate(input_ids, max_new_tokens=10)
    print(f"生成序列形状: {generated_ids.shape}")  # 应该是 [2, 42]
