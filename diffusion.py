import torch
import torch.nn as nn
import torch.optim as optim

# 定义前向扩散过程
def forward_diffusion(x0, t, noise_schedule):
    """
    前向扩散过程：逐步添加噪声。
    x0: 初始数据 (原始图像)
    t: 时间步
    noise_schedule: 噪声调度表
    """
    noise = torch.randn_like(x0)
    alpha_t = noise_schedule[t]
    xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
    return xt, noise

# 定义后向扩散过程
class SimpleDiffusionModel(nn.Module):
    def __init__(self):
        super(SimpleDiffusionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
        )
    
    def forward(self, x, t):
        """
        后向扩散过程：预测噪声以恢复原始数据。
        x: 扩散后的数据
        t: 时间步
        """
        return self.network(x)

# 设置噪声调度表
def get_noise_schedule(T):
    """
    生成噪声调度表。
    T: 最大时间步
    """
    return torch.linspace(1e-4, 0.02, T)

# 模型训练示例
def train_diffusion_model():
    T = 1000  # 最大时间步数
    noise_schedule = get_noise_schedule(T)
    model = SimpleDiffusionModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 假设有一个数据集 x0_data
    x0_data = torch.randn(64, 28*28)  # 假设是28x28的图像数据

    for epoch in range(100):  # 训练100个epoch
        for x0 in x0_data:
            t = torch.randint(0, T, (1,)).item()
            xt, noise = forward_diffusion(x0, t, noise_schedule)
            xt = xt.view(1, -1)  # 展平为一维向量
            
            # 预测噪声
            noise_pred = model(xt, t)
            loss = loss_fn(noise_pred, noise.view(1, -1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

if __name__ == "__main__":
    train_diffusion_model()
