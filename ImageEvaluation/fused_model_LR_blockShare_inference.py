import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from fused_model_LR_blockShare import BlockwiseLinearFusion


def blockwise_weighted_fusion(
    input_root, weights, block_size=16, device="cuda", output_dir="fused_output"
):
    """
    进行 block-wise 融合：每 block_size × block_size 区域使用一组 32 维的权重。

    Args:
        input_root: 包含 32 个版本图像文件夹的根目录
        weights: torch.Tensor, shape (num_blocks_h, num_blocks_w, 3, 32)
        block_size: 每个融合块的大小（例如 16）
        device: 使用设备
        output_dir: 输出文件夹
    """
    os.makedirs(output_dir, exist_ok=True)

    input_dirs = sorted([
        os.path.join(input_root, d)
        for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])
    assert len(input_dirs) == 32, "需要32个输入版本"

    # 读取图像名
    image_names = sorted([
        f for f in os.listdir(input_dirs[0]) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    # 获取图像大小
    sample_img = Image.open(os.path.join(input_dirs[0], image_names[0]))
    W, H = sample_img.size
    assert W % block_size == 0 and H % block_size == 0, "图像大小需能整除 block_size"

    num_blocks_y = H // block_size
    num_blocks_x = W // block_size

    assert weights.shape == (num_blocks_y, num_blocks_x, 3, 32), \
        f"权重形状应为 ({num_blocks_y}, {num_blocks_x}, 3, 32)"

    weights = weights.to(device)

    for name in tqdm(image_names, desc="Block-wise 融合"):
        # 加载所有版本图像
        imgs = [
            np.array(Image.open(os.path.join(d, name)).convert("RGB"), dtype=np.float32) / 255.0
            for d in input_dirs
        ]
        imgs = np.stack(imgs, axis=0)  # (32, H, W, 3)
        imgs_tensor = torch.tensor(imgs, dtype=torch.float32, device=device)  # (32, H, W, 3)
        imgs_tensor = imgs_tensor.permute(3, 0, 1, 2)  # (3, 32, H, W)

        fused = torch.zeros((3, H, W), device=device)

        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                top = i * block_size
                left = j * block_size
                h_slice = slice(top, top + block_size)
                w_slice = slice(left, left + block_size)

                # 获取当前块的所有版本像素 (3, 32, block_size, block_size)
                block_pixels = imgs_tensor[:, :, h_slice, w_slice]  # (3, 32, bh, bw)

                ##### softmax归一化
                # 获取当前块的权重 (3, 32)
                raw_weights = weights[i, j]  # shape: (3, 32)
                # 归一化：对32个版本做 softmax，确保加权合理
                norm_weights = torch.nn.functional.softmax(raw_weights, dim=-1)  # (3, 32)
                # reshape 用于广播
                w_block = norm_weights.view(3, 32, 1, 1)  # (3, 32, 1, 1)
                #####

                # 原训练的 
                w_block = raw_weights.view(3, 32, 1, 1)
                
                ##### min-max 归一化
                # raw_weights = weights[i, j]  # (3, 32)
                # min_vals = raw_weights.min(dim=-1, keepdim=True).values
                # max_vals = raw_weights.max(dim=-1, keepdim=True).values
                # norm_weights = (raw_weights - min_vals) / (max_vals - min_vals + 1e-8)
                # w_block = norm_weights.view(3, 32, 1, 1)
                #####

                ##### 原版 , 要的是相对重要，而不是绝对值意义的。
                # 权重 shape: (3, 32, 1, 1)
                # w_block = weights[i, j].view(3, 32, 1, 1)
                #####

                # 融合: 对 32 维加权求和
                fused_block = (block_pixels * w_block).sum(dim=1)  # → (3, bh, bw)

                fused[:, h_slice, w_slice] = fused_block

        fused_img = (fused.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(fused_img).save(os.path.join(output_dir, name))

    print(f"✅ 所有图像已融合完成，保存于：{output_dir}")

block_size = 16

# 每个block不同权重，block内像素共享一套权重
# weights = torch.rand(num_blocks_y, num_blocks_x, 3, 32)
model = BlockwiseLinearFusion(512, 512, block_size=16)
model.load_state_dict(torch.load("checkpoints/block_epoch52_decay-4_batch4_block16_mse_lpips.pt"))
weights_tensor = model.weights.detach().cpu()
print(weights_tensor.shape) # (64, 64, 3, 32)
input_root = "../data/RealPhoto60/restore"

blockwise_weighted_fusion(
    input_root=input_root,        # 32个版本图像的根目录
    weights=weights_tensor,                     # 权重 tensor
    block_size=16,                       # 每个 block 的大小
    device="cuda",                       # 如果有GPU，使用它
    output_dir="../data/RealPhoto60/result_block_test"            # 输出目录
)
