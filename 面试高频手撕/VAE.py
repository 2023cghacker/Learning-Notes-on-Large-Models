import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# 1. VAE模型定义
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32, hidden_dim=256):
        super().__init__()

        # 编码器：输入→隐藏层→均值+方差
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器：潜在向量→隐藏层→输出（与输入维度一致）
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # 图像像素归一化到0-1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, 784)  # 展平图像（[batch,1,28,28]→[batch,784]）
        enc_out = self.enc(x)
        mu = self.fc_mu(enc_out)
        logvar = self.fc_logvar(enc_out)
        z = self.reparameterize(mu, logvar)
        dec_out = self.dec(z)
        return dec_out, mu, logvar


# 2. 损失函数（重构损失+KL散度）
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.BCELoss(reduction="sum")(recon_x, x.view(-1, 784))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
