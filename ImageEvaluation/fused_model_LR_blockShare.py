import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb

import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import lpips


def log_weight_visualizations(model, epoch):
    """
    可视化 BlockwiseLinearFusion 模型的权重选择和不确定性（熵）分布。
    """
    with torch.no_grad():
        soft_weights = torch.softmax(model.weights.detach().cpu(), dim=-1).numpy()  # (Hb, Wb, C, K)
        Hb, Wb, C, K = soft_weights.shape

        # dominant version index 可视化 (以红色通道为例)
        max_indices = soft_weights.argmax(axis=-1)  # (Hb, Wb, C)

        center_idx = C // 2  # 例如取通道 0（红色）
        plt.figure(figsize=(6, 6))
        plt.imshow(max_indices[:, :, center_idx], cmap="viridis")
        plt.title(f"Dominant Version Index Map (Channel {center_idx})")
        plt.colorbar()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({"dominant_version_map": wandb.Image(image)}, step=epoch + 1)
        plt.close()

        # 熵图可视化：通道平均
        entropy = -np.sum(soft_weights * np.log(soft_weights + 1e-8), axis=-1)  # (Hb, Wb, C)
        entropy_map = entropy.mean(axis=-1)  # → (Hb, Wb)

        plt.figure(figsize=(6, 6))
        plt.imshow(entropy_map, cmap="hot")
        plt.title("Entropy Map (avg over channels)")
        plt.colorbar()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({"entropy_map": wandb.Image(image)}, step=epoch + 1)
        plt.close()

        # 统计某通道中各版本的出现频次（柱状图）
        flat_indices = max_indices[:, :, center_idx].flatten()
        version_counts = np.bincount(flat_indices, minlength=K)

        plt.figure(figsize=(8, 4))
        plt.bar(range(K), version_counts)
        plt.title(f"Version Frequency (Channel {center_idx})")
        plt.xlabel("Version Index")
        plt.ylabel("Frequency")

        # 使用 wandb.Image 保存图像
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        wandb.log({"version_distribution": wandb.Image(Image.open(buf))}, step=epoch + 1)
        plt.close()



class BlockwiseFusionDataset(Dataset):
    def __init__(self, input_root, gt_root, transform=None):
        self.degraded_dirs = sorted([
            os.path.join(input_root, d)
            for d in os.listdir(input_root)
            if os.path.isdir(os.path.join(input_root, d))
        ])
        self.gt_root = gt_root
        self.image_names = sorted([
            f for f in os.listdir(self.degraded_dirs[0])
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        imgs = [
            np.array(Image.open(os.path.join(d, name)).convert("RGB"), dtype=np.float32) / 255.0
            for d in self.degraded_dirs
        ]
        gt = np.array(Image.open(os.path.join(self.gt_root, name)).convert("RGB"), dtype=np.float32) / 255.0

        imgs_tensor = torch.tensor(np.stack(imgs), dtype=torch.float32).permute(0, 3, 1, 2)  # (32, 3, H, W)
        gt_tensor = torch.tensor(gt, dtype=torch.float32).permute(2, 0, 1)  # (3, H, W)

        return imgs_tensor, gt_tensor, name

class BlockwiseLinearFusion(nn.Module):
    def __init__(self, height, width, block_size, channels=3, num_versions=32):
        super().__init__()
        self.Hb = height // block_size
        self.Wb = width // block_size
        self.block_size = block_size
        self.weights = nn.Parameter(torch.rand(self.Hb, self.Wb, channels, num_versions))

    def forward(self, x):
        """
        x: (B, 32, 3, H, W)
        输出: (B, 3, H, W)
        """
        B, K, C, H, W = x.shape
        output = torch.zeros((B, C, H, W), device=x.device)

        for i in range(self.Hb):
            for j in range(self.Wb):
                h_slice = slice(i * self.block_size, (i + 1) * self.block_size)
                w_slice = slice(j * self.block_size, (j + 1) * self.block_size)

                block = x[:, :, :, h_slice, w_slice]          # (B, 32, 3, bh, bw)
                block = block.permute(0, 2, 1, 3, 4)          # (B, 3, 32, bh, bw)
                w_block = self.weights[i, j].view(1, C, K, 1, 1)
                fused_block = (block * w_block).sum(dim=2)   # (B, 3, bh, bw)

                output[:, :, h_slice, w_slice] = fused_block

        return output

from torch.utils.data import random_split

def create_dataloaders(input_root, gt_root, val_ratio=0.2, batch_size=4):
    dataset = BlockwiseFusionDataset(input_root, gt_root)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return train_loader, val_loader


def train_blockwise_model(
    input_root,
    gt_dir,
    image_size=(1024, 1024),
    block_size=16,
    batch_size=4,
    num_epochs=30,
    lr=1e-2,
    val_ratio=0.2,
    device="cuda",
    save_every=20,
):
    # Initialize wandb for logging
    wandb.init(project="blockwise_fusion", config={
        "image_size": image_size,
        "block_size": block_size,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "val_ratio": val_ratio,
    })

    train_loader, val_loader = create_dataloaders(input_root, gt_dir, val_ratio, batch_size)

    model = BlockwiseLinearFusion(*image_size, block_size=block_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_loss = float("inf")
    patience, max_patience = 0, 5

    lpips_fn = lpips.LPIPS(net='vgg').to(device)  # 或者 net='vgg'，根据你需求
    lpips_weight = 0.5  # 假设给LPIPS一个权重，可以根据需要调整

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_lpips_loss = 0.0
        for degraded, gt, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            degraded = degraded.to(device)  # (B, 32, 3, H, W)
            gt = gt.to(device)              # (B, 3, H, W)

            output = model(degraded)
            mse_loss = F.mse_loss(output, gt)
            
            # 添加lpips损失
            lpips_loss = lpips_fn(output, gt).mean()

            loss = mse_loss + lpips_weight * lpips_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse_loss += mse_loss.item()
            train_lpips_loss += lpips_loss.item()
            
        # avg_train_loss = train_loss / len(train_loader)
        # Log training loss to wandb
        # wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})

        # -------- 验证 --------
        model.eval()
        val_loss = 0.0
        val_mse_loss = 0.0
        val_lpips_loss = 0.0
        with torch.no_grad():
            for degraded, gt, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                degraded = degraded.to(device)
                gt = gt.to(device)
                output = model(degraded)
                # loss = F.mse_loss(output, gt)

                mse_loss = F.mse_loss(output, gt)
                lpips_loss = lpips_fn(output, gt).mean()

                v_loss = mse_loss + lpips_weight * lpips_loss

                val_loss += v_loss.item()
                val_mse_loss += mse_loss.item()
                val_lpips_loss += lpips_loss.item()

        # avg_val_loss = val_loss / len(val_loader)
        # Log validation loss to wandb
        # wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})

        avg_train_loss = train_loss / len(train_loader)
        avg_train_mse_loss = train_mse_loss / len(train_loader)
        avg_train_lpips_loss = train_lpips_loss / len(train_loader)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse_loss = val_mse_loss / len(val_loader)
        avg_val_lpips_loss = val_lpips_loss / len(val_loader)

        wandb.log({
            "train_loss": avg_train_loss,
            "train_mse_loss": avg_train_mse_loss,
            "train_lpips_loss": avg_train_lpips_loss,
            "epoch": epoch + 1
        })

        wandb.log({
            "val_loss": avg_val_loss,
            "val_mse_loss": avg_val_mse_loss,
            "val_lpips_loss": avg_val_lpips_loss,
            "epoch": epoch + 1
        })

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), "best_blockwise_model.pt")
            print("✅ 验证损失下降，保存模型")
        else:
            patience += 1
            if patience >= max_patience:
                print("🛑 Early stopping.")
                break
        
        # 间隔保存模型
                # 每隔 save_every 保存一次模型
        if (epoch + 1) % save_every == 0:
            ckpt_path = f"BlockFuse_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"📦 Checkpoint saved: {ckpt_path}")
        
        # 在验证日志之后添加
        # log_weight_visualizations(model, epoch)

     # Final model save and wandb logging
    # wandb.log({"final_val_loss": best_val_loss})
    wandb.finish()

    return model

if __name__ == "__main__":

    model = train_blockwise_model(
        input_root="../../data/DIV2K_Flickr_LSDIR5000/restore",
        gt_dir = "../../data/DIV2K_Flickr_LSDIR5000/gt_1000name",
        image_size=(512, 512),
        block_size=16,
        batch_size=4,
        num_epochs=1000,
        lr=1e-4,
        val_ratio=0.1,
        device="cuda"
    )

# 得到的weights是(num_blocks_y, num_blocks_x, C, num_prompts)
