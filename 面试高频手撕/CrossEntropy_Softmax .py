import numpy as np


# Softmax函数
def softmax(z):
    z_max = np.max(z, axis=-1, keepdims=True)  # 减去最大值放置溢出
    z_exp = np.exp(z - z_max)
    z_sum = np.sum(z_exp, axis=-1, keepdims=True)
    return z_exp / z_sum


# 交叉熵损失函数
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    loss = -np.sum(y_true * np.log(y_pred + epsilon), axis=-1).mean()

    return loss


# 测试数据
z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y_true = np.array([[0, 1, 0], [1, 0, 0]])

# 计算Softmax输出
y_pred = softmax(z)
print("Softmax output:")
print(y_pred)

# 计算交叉熵损失
loss = cross_entropy_loss(y_true, y_pred)
print("Cross-Entropy Loss:")
print(loss)
