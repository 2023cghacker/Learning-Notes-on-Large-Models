import numpy as np


class FFN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        # 参数初始化（Xavier）：通过输入维度缩放随机权重，加速训练收敛
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(
            input_dim
        )  # 输入层到隐藏层权重
        self.b1 = np.zeros((1, hidden_dim))  # 隐藏层偏置
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(
            hidden_dim
        )  # 隐藏层到输出层权重
        self.b2 = np.zeros((1, output_dim))  # 输出层偏置
        self.lr = lr  # 学习率
        self.y = None  # 存储真实标签用于反向传播
        self.probs = None  # 存储softmax概率用于反向传播

    def relu(self, x):
        # ReLU激活函数：引入非线性，输出max(0, x)
        return np.maximum(0, x)

    def relu_grad(self, x):
        # ReLU导数：输入大于0时为1，否则为0
        return (x > 0).astype(float)

    def forward(self, X):
        # 前向传播：三层网络（输入→隐藏→输出）
        self.z1 = X @ self.W1 + self.b1  # 隐藏层线性输出（z1 = W1·X + b1）
        self.a1 = self.relu(self.z1)  # 隐藏层激活输出（a1 = ReLU(z1)）
        self.z2 = self.a1 @ self.W2 + self.b2  # 输出层线性输出（z2 = W2·a1 + b2）
        return self.z2  # 返回原始logits（未经过softmax）

    def backward(self, X, loss):
        """
        从损失开始计算反向传播
        loss: 损失值（用于启动反向传播）
        """
        # 从损失计算输出层梯度（dL/dz2）
        grad_z2 = self.probs
        grad_z2[np.arange(len(self.y)), self.y] -= 1
        grad_z2 /= len(self.y)  # 平均到每个样本

        # 计算输出层参数梯度
        dW2 = self.a1.T @ grad_z2  # dL/dW2 = a1^T · grad_z2
        db2 = np.sum(grad_z2, axis=0, keepdims=True)  # dL/db2 = 求和(grad_z2)

        # 计算隐藏层梯度（链式法则）
        grad_a1 = grad_z2 @ self.W2.T  # dL/da1 = grad_z2 · W2^T
        grad_z1 = grad_a1 * self.relu_grad(self.z1)  # dL/dz1 = dL/da1 · ReLU'(z1)

        # 计算输入层参数梯度
        dW1 = X.T @ grad_z1  # dL/dW1 = X^T · grad_z1
        db1 = np.sum(grad_z1, axis=0, keepdims=True)  # dL/db1 = 求和(grad_z1)

        # SGD更新参数：权重 = 权重 - 学习率×梯度
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train_step(self, X, y):
        # 存储真实标签用于反向传播
        self.y = y

        # 前向传播得到logits
        logits = self.forward(X)

        # 计算softmax概率分布（数值稳定版：减去每行最大值）
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 计算交叉熵损失（加微小值避免log(0)）
        y_probs = self.probs[np.arange(len(y)), y]  # 真实类别对应的预测概率
        loss = -np.mean(np.log(y_probs + 1e-12))

        # 从损失开始反向传播更新参数
        self.backward(X, loss)

        return loss


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(5, 4)  # 5个样本，每个4维特征
    y = np.array([0, 2, 1, 2, 0])  # 类别标签（共3类）

    # 初始化网络：输入4维，隐藏层8维，输出3类，学习率0.1
    net = FFN(input_dim=4, hidden_dim=8, output_dim=3, lr=0.1)

    # 训练10轮
    for epoch in range(10):
        loss = net.train_step(X, y)
        print(f"Epoch {epoch+1}, Loss={loss:.4f}")
