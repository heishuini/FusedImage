import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

device="cuda" if torch.cuda.is_available() else "cpu"
class PixelWiseLinearFusion(nn.Module):
    def __init__(self, height=512, width=512, channels=3, num_inputs=32):
        super().__init__()
        self.H = height
        self.W = width
        self.C = channels
        self.K = num_inputs
        # 权重 shape: (H, W, C, K)，每个像素通道一个向量
        self.weights = nn.Parameter(torch.randn(height, width, channels, num_inputs))
    
    def forward(self, inputs):
        """
        inputs: (batch, K, H, W, C) - 32 个版本的图像堆叠
        """
        # 调整维度：→ (batch, H, W, C, K)
        inputs = inputs.permute(0, 2, 3, 4, 1)
        # 权重 normalize 到 softmax 以稳定融合
        soft_weights = torch.softmax(self.weights, dim=-1)  # (H, W, C, K)
        # 融合操作
        fused = torch.sum(inputs * soft_weights.unsqueeze(0), dim=-1)  # → (batch, H, W, C)
        return fused

def batch_fuse_images(input_dirs, output_dir, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件名（假设用第一个目录的文件名为主）
    image_names = sorted(os.listdir(input_dirs[0]))

    # 加载模型
    model = PixelWiseLinearFusion().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for name in tqdm(image_names, desc="Fusing images"):
        # 加载 32 个版本并堆叠成 tensor
        try:
            input_stack = torch.stack([
                torch.tensor(np.array(Image.open(os.path.join(d, name)), dtype=np.float32) / 255.0)
                for d in input_dirs
            ], dim=0).permute(0, 3, 1, 2)  # (K, C, H, W)

            # 变成 (1, K, H, W, C)
            inputs = input_stack.permute(0, 2, 3, 1).unsqueeze(0).to(device)  # (1, K, H, W, C)

            # 推理
            with torch.no_grad():
                fused = model(inputs)  # (1, H, W, C)

            fused_np = fused.squeeze(0).cpu().numpy()  # (H, W, C)
            fused_image_uint8 = (fused_np * 255).clip(0, 255).astype(np.uint8)

            # 保存图像
            Image.fromarray(fused_image_uint8).save(os.path.join(output_dir, name))
        except Exception as e:
            print(f"Failed to process {name}: {e}")

folder_paths = {
        "blur_haze": "../data/RealPhoto60/restore/sr_omini_blur_haze",
        "blur_jpeg": "../data/RealPhoto60/restore/sr_omini_blur_jpeg",
        "blur_lowlight": "../data/RealPhoto60/restore/sr_omini_blur_lowlight",
        "blur_noise": "../data/RealPhoto60/restore/sr_omini_blur_noise",
        "blur_noise_jpeg": "../data/RealPhoto60/restore/sr_omini_blur_noise_jpeg",
        "blur_rain": "../data/RealPhoto60/restore/sr_omini_blur_rain",
        "blur_raindrop": "../data/RealPhoto60/restore/sr_omini_blur_raindrop",
        "deblur": "../data/RealPhoto60/restore/sr_omini_deblur",
        "dehaze": "../data/RealPhoto60/restore/sr_omini_dehaze",
        "denoise": "../data/RealPhoto60/restore/sr_omini_denoise",
        "derain": "../data/RealPhoto60/restore/sr_omini_derain",
        "deraindrop": "../data/RealPhoto60/restore/sr_omini_deraindrop",
        "haze_jpeg": "../data/RealPhoto60/restore/sr_omini_haze_jpeg",
        "haze_lowlight": "../data/RealPhoto60/restore/sr_omini_haze_lowlight",
        "haze_noise": "../data/RealPhoto60/restore/sr_omini_haze_noise",
        "haze_rain": "../data/RealPhoto60/restore/sr_omini_haze_rain",
        "haze_raindrop": "../data/RealPhoto60/restore/sr_omini_haze_raindrop",
        "jpeg": "../data/RealPhoto60/restore/sr_omini_jpeg",
        "lowlight": "../data/RealPhoto60/restore/sr_omini_lowlight",
        "lowlight_blur_jpeg": "../data/RealPhoto60/restore/sr_omini_lowlight_blur_jpeg",
        "lowlight_blur_noise": "../data/RealPhoto60/restore/sr_omini_lowlight_blur_noise",
        "lowlight_jpeg": "../data/RealPhoto60/restore/sr_omini_lowlight_jpeg",
        "lowlight_noise": "../data/RealPhoto60/restore/sr_omini_lowlight_noise",
        "lowlight_rain": "../data/RealPhoto60/restore/sr_omini_lowlight_rain",
        "lowlight_raindrop": "../data/RealPhoto60/restore/sr_omini_lowlight_raindrop",
        "noise_jpeg": "../data/RealPhoto60/restore/sr_omini_noise_jpeg",
        "noise_rain": "../data/RealPhoto60/restore/sr_omini_noise_rain",
        "noise_raindrop": "../data/RealPhoto60/restore/sr_omini_noise_raindrop",
        "rain_jpeg": "../data/RealPhoto60/restore/sr_omini_rain_jpeg",
        "raindrop_jpeg": "../data/RealPhoto60/restore/sr_omini_raindrop_jpeg",
        "sr": "../data/RealPhoto60/restore/sr_omini_sr",
        "universal": "../data/RealPhoto60/restore/sr_omini_universal",
    }

input_dirs = [value for key,value in folder_paths.items()]

# output_dir = "../data/RealPhoto60/result_pixel_mse_lpips"
output_dir = "../data/RealPhoto60/result_pixel_test"
model_path = "checkpoints/pixel_epoch4_decay-2_batch64_mse_lpips.pth"

batch_fuse_images(input_dirs, output_dir, model_path)


# # 单张图片融合测试
# # 权重 shape: (H, W, C, K)，每个像素通道一个向量
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = PixelWiseLinearFusion().to(device)
# model.load_state_dict(torch.load("best_pixelwise_model.pth", map_location=device))
# model.eval()
# print(model.weights.data.shape)

# name = "lq_001.png"

# inputs = torch.stack([
#     torch.tensor(np.array(Image.open(os.path.join(d, name)), dtype=np.float32) / 255.0)
#     for d in input_dirs
# ], dim=0).permute(0, 3, 1, 2)  # (32, 3, H, W)

# # 变成 (1, 32, H, W, 3)
# inputs = inputs.permute(0, 2, 3, 1).unsqueeze(0).to(device)  # (1, 32, H, W, 3)
# with torch.no_grad():
#     fused = model(inputs)  # (1, H, W, C)

# # 移除batch
# fused = fused.squeeze(0).cpu().detach().numpy()  # (H, W, C)

# # 将tensor转为图像保存
# fused_image_uint8 = (fused * 255).astype(np.uint8)  # 转为 0-255 之间的 uint8 类型
# Image.fromarray(fused_image_uint8).save("fused_image.png")