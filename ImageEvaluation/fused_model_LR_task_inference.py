import pyiqa
import os
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from utils import util_image
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

import torch
import torch.nn as nn
import math

class LinearFusionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearFusionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

def weighted_fusion(images, scores, image_id):
    """
    简单加权融合多处理结果图像
    
    参数:
        images: 图像list(0~1的值) [universal_img的像素值, deblur_img, dehaze_img, ...]
        scores: 对应得分列表 [universal_score, deblur_score, ...]
    
    返回:
        融合后的图像
    """
    
    # 将得分转为权重，softmax

    weights = np.array(scores, dtype=np.float32)
    # weights = np.exp(weights - np.max(weights))  # 数值稳定性优化
    # weights /= np.sum(weights)  # Softmax归一化

    weights = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))
    # print("Softmax后权重:", weights) 

    # 初始化结果图像
    result = np.zeros_like(images[0], dtype=np.float32)

    # 加权融合， 逐步进行
    for i,(img, weight) in enumerate(zip(images, weights)):
        # imread时转为float32，又除了255，此时是[0,1]之间
        img_float = img
        result += weight * img_float

        # 要求result是[-1,1]之间
        # normalized_immediate = np.clip(result, -1, 1)
        # util_image.imwrite(normalized_immediate,os.path.join('../data/DIV2K/weighted_x4/immediate',f'{image_id:03d}_step_{i:02d}.png'))
        
    # clip是确保值在0~255
    normalized_img = np.clip(result, -1, 1)
    util_image.imwrite(normalized_img,os.path.join('../data/RealPhoto60/result_task_test',f'result_{image_id:04d}.png'))
    # cv2.imwrite('output.png', normalized_img)  # OpenCV要求uint8类型


if __name__ == "__main__":
    # 定义8个文件夹路径
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

    model = LinearFusionModel(input_dim=32)  # 你是32组图嘛
    model.load_state_dict(torch.load("checkpoints/task_epoch13_decay-4_batch4_block16_mse.pth"))  # 推理可以在CPU
    model.eval()
    # 从linear层拿权重和偏置
    weights = model.linear.weight.data.squeeze(0)  # shape: (32,)
    bias = model.linear.bias.data.item()            # float
    
    weights_list = weights.tolist()  # ---> 变成长度32的普通Python列表

    # 验证所有文件夹存在
    for name, path in folder_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found: {path}")

    # 读取图像
    all_images = {}
    for method, folder_path in folder_paths.items():
        folder = Path(folder_path)
        image_files = sorted([f for f in folder.glob("*.png")])
        all_images[method] = image_files

    num_images = len(all_images[list(all_images.keys())[0]])

    # num_images = 1
    for i in tqdm(range(num_images), desc="Processing images"):
        img_result = {"image_id": i+1}
        im_in_list = []

        # 各类别图像读取
        for method in folder_paths.keys():
            img_path = all_images[method][i]
            
            # 加载图像并融合
            try:
                # 转为numpy，值是0~1
                im_in = util_image.imread(img_path, chn='rgb', dtype='float32')
                # im_out = util_image.imwrite(im_in, os.path.join('../data/DIV2K/weighted_x4/origin',f'origin_{method}_{i:02d}.png'))

                im_in_list.append(im_in)
                # im_in_tensor = util_image.img2tensor(im_in).to(device) 

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                img_result[method] = np.nan

        # 简单的加权融合
        weighted_fusion(im_in_list,weights_list,image_id=i)
        # 

        
        


