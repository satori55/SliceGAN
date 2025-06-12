import os
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import UNet3DConditionModel, DDPMScheduler
from matplotlib import pyplot as plt
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import tifffile as tiff
import cv2
import numpy as np


def eachFile(path_file) -> list:
    """
    get file name from path_file
    :param path_file: target path
    :return: file name list in target path
    """
    fileName = []
    for file in os.listdir(path_file):
        if os.path.isfile(os.path.join(path_file, file)):
            fileName.append(file)
    return fileName

path = "./bittergourd_nitrogen"
filename = eachFile(path)
print(filename)

train_data = []
for file in filename:
    img_file = cv2.imread(path + "/" + file)
    img_resized = cv2.resize(img_file, (128, 128), interpolation=cv2.INTER_AREA)

    train_data.append(img_resized)

print(len(train_data))

# list to array
train_data = np.array(train_data)
print(train_data.shape)

# tiff.imshow(train_data[0])

# data loader
# 将 train_data 转换为 PyTorch 张量，并确保其类型正确
train_data = torch.tensor(train_data, dtype=torch.float32)

train_data = train_data / 255.0

# 调整张量的维度顺序，使其符合模型的输入格式 (N, C, H, W)
train_data = train_data.permute(3, 0, 1, 2)
print(train_data.shape)

data =  train_data[:, :32, :32, :32]
# add batch dimension
data = data.unsqueeze(0)
print(data.shape)

# 创建 TensorDataset
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_data), batch_size=5, shuffle=True
)


# model
model = UNet3DConditionModel(
    sample_size=32,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    down_block_types=(
        "CrossAttnDownBlock3D",  # a regular ResNet downsampling block
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "DownBlock3D",  # a ResNet downsampling block with spatial self-attention,
    ),
    up_block_types=(
        "UpBlock3D",  # a regular ResNet upsampling block
        "CrossAttnUpBlock3D",  # a ResNet upsampling block with spatial self-attention
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D",
    ),
    block_out_channels=(32, 64, 128, 128),
    # num_attention_heads=4,
    attention_head_dim=32,
    cross_attention_dim=32,
)

batch_size = data.shape[0]
sequence_length = 32  # 根据任务或模型调整
hidden_size = 32      # 模型配置的特征维度

encoder_hidden_states = torch.randn(batch_size, sequence_length, hidden_size).to(torch.device("cuda"))


model.train()
model.to(torch.device("cuda"))
out = model(data.to(torch.device("cuda")), 100, encoder_hidden_states)
print(f"{out.sample.shape=}")


# create a scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=500)
noise = torch.randn(train_data.shape)

# train the model
number_of_epochs = 4000
learing_rate = 1e-5
num_of_warmup_steps = 200


optimizer = torch.optim.AdamW(model.parameters(), lr=learing_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_of_warmup_steps,
    num_training_steps=(len(train_loader) * number_of_epochs),
)

# training loop
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
print(f"Using device: {device}")
model.to(device)

#
def get_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02, schedule_type="linear"):
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule_type == "quadratic":
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}")

# 设置时间步数
num_timesteps = 500

# 计算 beta 系数
beta = get_beta_schedule(num_timesteps)

# 计算 alpha 系数
alpha = 1.0 - beta

# 计算累积 alpha 系数
alpha_cumprod = torch.cumprod(alpha, dim=0)

# 计算逆扩散过程中所需的平方根 alpha 系数
rec_sqrt_alpha = torch.sqrt(1.0 / alpha)
sqrt_beta = torch.sqrt(beta)
beta_over_sqrt_1sub_alcum = beta / torch.sqrt(1.0 - alpha_cumprod)


for epoch in range(number_of_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        train_images = batch
        train_images = train_images[0].to(device)

        # 生成噪声
        noise = torch.randn(train_images.shape).to(device)

        # 随机选择时间步
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (train_images.shape[0],), device=device).long()

        # 为每个样本添加噪声
        noisy_images = noise_scheduler.add_noise(train_images, noise, timesteps)

        # 前向传播
        noise_pred = model(noisy_images, timesteps).sample

        # 计算损失
        loss = F.mse_loss(noise_pred, noise)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        train_loss += loss.item()

    # 计算平均训练损失
    avg_train_loss = train_loss / len(train_loader)

    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{number_of_epochs}], Loss: {avg_train_loss:.4f}")

    # 可视化一些训练图像和噪声图像
    if epoch % 50 == 0:
        noisy_image = torch.randn(train_images.shape).to(device)  # 初始化为纯噪声
        original_noise = noisy_image.clone()

        # 设置模型为评估模式
        model.eval()

        with torch.no_grad():
            timesteps = torch.arange(num_timesteps, 0, -1).to(device)  # 从较大的时间步开始，逐步减少时间步
            for t in timesteps:
                # 预测当前时间步的噪声
                t_tensor = torch.full((noisy_image.size(0),), t, dtype=torch.long, device=device)
                noise_pred = model(noisy_image, t_tensor).sample

                # 从 precomputed 的数组中获取当前时间步的 alpha 和 beta 系数
                alpha_t = alpha[t - 1]
                alpha_t_minus_1 = alpha[t - 2] if t > 1 else alpha[0]
                beta_t = beta[t - 1]

                # 计算当前去噪图像
                noisy_image = (1 / torch.sqrt(alpha_t_minus_1)) * (
                        noisy_image - (beta_over_sqrt_1sub_alcum[t - 1]) * noise_pred
                )

                # 在每一步后加入少量随机噪声（可以根据实际需求调整）
                if t > 1:
                    noise = torch.randn_like(noisy_image).to(device)
                    noisy_image += noise * torch.sqrt(beta_t)

            denoised_image = noisy_image  # 最后的输出就是去噪后的图像

            plt.figure(figsize=(15, 5))

            # 显示原始图像
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(train_images[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.axis('off')  # 不显示坐标轴

            # 显示带噪声的图像
            plt.subplot(1, 3, 2)
            plt.title("Noisy Image")
            plt.imshow(original_noise[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.axis('off')

            # 显示去噪后的图像
            plt.subplot(1, 3, 3)
            plt.title("Denoised Image")
            plt.imshow(denoised_image[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.axis('off')

            plt.savefig(f"./resultsave/denoised_image_{epoch}.png")
            plt.close()
