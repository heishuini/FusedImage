import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from fused_model_LR_pixel_inference import RegionWiseLinearFusion

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def apply_patchwise_fusion_gpu(
    weights, biases, input_root, image_root, output_dir, patch_size=32, device="cuda"
):
    os.makedirs(output_dir, exist_ok=True)

    # 加载输入图像路径
    input_dirs = sorted([
        os.path.join(input_root, d)
        for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])
    num_inputs = len(input_dirs)

    image_names = sorted([
        f for f in os.listdir(image_root)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    # 加载一张样图确定尺寸
    sample_img = Image.open(os.path.join(input_dirs[0], image_names[0])).convert("RGB")
    width, height = sample_img.size
    h_patches = height // patch_size
    w_patches = width // patch_size
    channels = 3

    input_dim = num_inputs * patch_size * patch_size * channels
    output_dim = patch_size * patch_size * channels
    assert weights.shape == (input_dim, output_dim)
    assert biases.shape == (1, output_dim)

    # 将权重加载到 GPU 上
    W = torch.tensor(weights, dtype=torch.float32, device=device)
    B = torch.tensor(biases, dtype=torch.float32, device=device)

    for name in tqdm(image_names, desc="GPU融合"):
        # 打开所有版本图像并转换为 numpy
        img_versions = [
            np.array(Image.open(os.path.join(d, name)).convert("RGB"), dtype=np.float32) / 255.0
            for d in input_dirs
        ]
        img_versions = np.stack(img_versions, axis=0)  # (num_inputs, H, W, 3)

        # 预分配融合图像
        fused_image = np.zeros((height, width, channels), dtype=np.float32)

        for ph in range(h_patches):
            for pw in range(w_patches):
                # 提取该位置的 patch：→ (num_inputs, patch_size, patch_size, 3)
                patch_stack = img_versions[
                    :,
                    ph * patch_size : (ph + 1) * patch_size,
                    pw * patch_size : (pw + 1) * patch_size,
                    :
                ]

                # reshape 成向量：→ (input_dim,)
                patch_vector = patch_stack.transpose(0, 3, 1, 2).reshape(-1)
                patch_tensor = torch.tensor(patch_vector, device=device).unsqueeze(0)  # (1, input_dim)

                # 加权融合
                fused_tensor = torch.matmul(patch_tensor, W) + B  # (1, output_dim)
                fused_patch = fused_tensor.view(patch_size, patch_size, channels).clamp(0, 1).cpu().numpy()

                # 放回图像
                h_start, h_end = ph * patch_size, (ph + 1) * patch_size
                w_start, w_end = pw * patch_size, (pw + 1) * patch_size
                fused_image[h_start:h_end, w_start:w_end, :] = fused_patch

        # 保存融合图像
        fused_image_uint8 = (fused_image * 255).astype(np.uint8)
        Image.fromarray(fused_image_uint8).save(os.path.join(output_dir, name))

    print(f"✅ 融合完成，结果保存在 {output_dir}")

num_inputs = 32

patch_size = 16

num_patches = (1024 // patch_size) ** 2
input_dim = num_inputs * patch_size * patch_size * 3
output_dim = patch_size * patch_size * 3 

# 模型加载
model = RegionWiseLinearFusion(input_dim, output_dim, num_patches)
model.load_state_dict(torch.load("region_pixel_model.pth", map_location="cpu"))
model.eval()

# 提取参数
weights = model.weights.detach().cpu().numpy()   # (input_dim, output_dim)
biases = model.biases.detach().cpu().numpy()     # (1, output_dim)
print(weights.shape)
print(biases.shape)

# 融合
input_root = "../../data/DIV2K/crop_1024/x2/restore"
image_root = "../../data/DIV2K/crop_1024/x2/restore/sr_omini_blur_haze"
output_dir = "pixel_result"

apply_patchwise_fusion_gpu(
    weights=weights,
    biases=biases,
    input_root=input_root,     # 包含32个子文件夹的路径
    image_root=image_root,  # 任选一个子文件夹用于读取图像名
    output_dir=output_dir,
    patch_size=patch_size
)
