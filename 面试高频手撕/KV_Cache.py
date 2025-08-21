import torch
import torch.nn as nn

class KVCache:
    def __init__(self, batch_size, num_heads, head_dim, max_seq_len, device="cuda"):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device

        # 初始化缓存（初始为空）
        self.past_keys = torch.zeros(
            (batch_size, num_heads, 0, head_dim), device=device
        )
        self.past_values = torch.zeros(
            (batch_size, num_heads, 0, head_dim), device=device
        )

    def update(self, new_key, new_value):
        """
        更新 KV Cache：
        - new_key:   [batch_size, num_heads, 1, head_dim]
        - new_value: [batch_size, num_heads, 1, head_dim]
        """
        # 将新 KV 追加到缓存
        self.past_keys = torch.cat([self.past_keys, new_key], dim=2)
        self.past_values = torch.cat([self.past_values, new_value], dim=2)

        # 如果超过最大长度，截断（FIFO）
        if self.past_keys.size(2) > self.max_seq_len:
            self.past_keys = self.past_keys[:, :, -self.max_seq_len:, :]
            self.past_values = self.past_values[:, :, -self.max_seq_len:, :]

    def get(self):
        """返回当前的 KV 缓存"""
        return self.past_keys, self.past_values

    def clear(self):
        """清空缓存"""
        self.past_keys = torch.zeros(
            (self.batch_size, self.num_heads, 0, self.head_dim), device=self.device
        )
        self.past_values = torch.zeros(
            (self.batch_size, self.num_heads, 0, self.head_dim), device=self.device
        )

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # QKV 投影层
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # 输出投影层
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # KV Cache
        self.kv_cache = None

    def forward(self, x, use_cache=False):
        """
        x: [batch_size, seq_len, hidden_size]
        use_cache: 是否使用 KV Cache
        """
        batch_size, seq_len, _ = x.shape

        # 计算 QKV
        q = self.q_proj(x)  # [batch_size, seq_len, hidden_size]
        k = self.k_proj(x)  # [batch_size, seq_len, hidden_size]
        v = self.v_proj(x)  # [batch_size, seq_len, hidden_size]

        # 拆分多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 使用 KV Cache
        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = KVCache(batch_size, self.num_heads, self.head_dim, max_seq_len=2048)

            # 更新缓存
            self.kv_cache.update(k, v)

            # 获取完整的 Key 和 Value（历史 + 当前）
            k, v = self.kv_cache.get()

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 注意力加权求和
        attn_output = torch.matmul(attn_probs, v)  # [batch_size, num_heads, seq_len, head_dim]

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 输出投影
        output = self.out_proj(attn_output)
        return output

if __name__ == "__main__":
    batch_size = 2
    hidden_size = 768
    num_heads = 12
    seq_len = 1  # 自回归生成，每次输出 1 个 token

    model = CausalSelfAttention(hidden_size, num_heads, max_seq_len=2048).to("cuda")
    x = torch.randn(batch_size, seq_len, hidden_size).to("cuda")

    # 第一次生成（无缓存）
    output1 = model(x, use_cache=True)
    print(output1.shape)

    # 第二次生成（使用缓存）
    output2 = model(x, use_cache=True)
    print(output2.shape)

    # 清空缓存
    model.kv_cache.clear()
