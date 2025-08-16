import numpy as np


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        初始化Adam优化器
        params: 待优化的参数（如numpy数组）
        lr: 学习率
        beta1: 一阶矩衰减率
        beta2: 二阶矩衰减率
        eps: 数值稳定性参数
        """
        self.params = params  # 待优化参数（如权重矩阵）
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # 初始化一阶矩和二阶矩（与参数同形状，初始为0）
        self.m = np.zeros_like(params)  # 一阶矩（动量项）
        self.v = np.zeros_like(params)  # 二阶矩（自适应项）
        self.t = 0  # 时间步（从0开始，迭代时+1）

    def step(self, grad):
        """
        执行一次参数更新
        grad: 当前步骤的梯度（与params同形状）
        """
        self.t += 1  # 时间步+1

        # 1. 更新一阶矩（动量项）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad

        # 2. 更新二阶矩（自适应项）
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        # 3. 偏差修正
        m_hat = self.m / (1 - self.beta1**self.t)  # 修正一阶矩
        v_hat = self.v / (1 - self.beta2**self.t)  # 修正二阶矩

        # 4. 更新参数
        self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return self.params  # 返回更新后的参数
