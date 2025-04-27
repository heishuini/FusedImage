import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

import matplotlib.pyplot as plt
import io
import lpips

# EarlyStopping 类
class EarlyStopping:
    def __init__(self, patience=5, delta=0, model_path="best_pixelwise_model.pth"):
        self.patience = patience  # 多少轮没有改善后停止
        self.delta = delta  # 每次最小改进
        self.best_loss = None  # 最好的验证损失
        self.counter = 0  # 没有改善的轮次数
        self.early_stop = False  # 是否早停
        self.model_path = model_path  # 模型保存路径

    def __call__(self, val_loss, model):
        """检查验证损失并决定是否早停"""
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        """保存当前验证损失最小的模型"""
        torch.save(model.state_dict(), self.model_path)
        print(f"Checkpoint saved with validation loss: {val_loss:.4f}")


class PixelWiseLinearFusion(nn.Module):
    def __init__(self, height=512, width=512, channels=3, num_inputs=32):
        super().__init__()
        self.H = height
        self.W = width
        self.C = channels
        self.K = num_inputs
        # 权重 shape: (H, W, C, K)，每个像素通道一个向量
        # 初始化为接近均匀分布，但加入轻微扰动，便于打破对称性
        uniform_weight = torch.ones(height, width, channels, num_inputs) / num_inputs
        noise = torch.randn_like(uniform_weight) * 0.01
        self.weights = nn.Parameter(uniform_weight + noise)

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
   
class FullImageFusionDataset(Dataset):
    def __init__(self, input_dirs, gt_dir, image_names=None):
        self.input_dirs = input_dirs
        self.gt_dir = gt_dir
        self.image_names = image_names or sorted(os.listdir(gt_dir))
        self.num_inputs = len(input_dirs)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        inputs = []

        for d in self.input_dirs:
            img = Image.open(os.path.join(d, name)).convert("RGB")
            img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
            inputs.append(img_np)

        inputs = np.stack(inputs, axis=0)  # (K, H, W, 3)
        gt_img = Image.open(os.path.join(self.gt_dir, name)).convert("RGB")
        gt_np = np.array(gt_img, dtype=np.float32) / 255.0  # (H, W, 3)

        return (
            torch.tensor(inputs, dtype=torch.float32),     # (K, H, W, 3)
            torch.tensor(gt_np, dtype=torch.float32)       # (H, W, 3)
        )

def train_pixel_fusion(
    input_dirs, gt_dir, height=512, width=512, channels=3, num_inputs=32,
    batch_size=2, epochs=10, lr=1e-4, val_ratio=0.1, device="cuda",project_name="PixelFusion", run_name="pixelwise_run",
    save_every=20, weight_decay=1e-2, l2_lambda=0, precision_limit=3, perceptual_lambda = 0.1
):
    # 初始化 wandb
    wandb.init(project=project_name, name=run_name, config={
        "height": height,
        "width": width,
        "channels": channels,
        "num_inputs": num_inputs,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "val_ratio": val_ratio
    })


    image_names = sorted(os.listdir(gt_dir))
    train_names, val_names = train_test_split(image_names, test_size=val_ratio, random_state=42)

    train_set = FullImageFusionDataset(input_dirs, gt_dir, image_names=train_names)
    val_set = FullImageFusionDataset(input_dirs, gt_dir, image_names=val_names)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = PixelWiseLinearFusion(height, width, channels, num_inputs).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    best_val_loss = float("inf")
    early_stopping = EarlyStopping(patience=3, delta=0.0001)  # 设置早停的容忍度

    lpips_loss = lpips.LPIPS(net='alex').to(device)  # or 'vgg'

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, gt in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            inputs, gt = inputs.to(device), gt.to(device)
            output = model(inputs)  # (B, H, W, C)

            # loss = criterion(output, gt)
            # 加入 L2 正则项， 此处是双重正则化了
            # 上面weight_decay是对所有参数加，此处是针对weight，但本模型参数只有weight，相当于翻倍惩罚。
            # l2_reg = torch.norm(model.weights, p=2)
            # loss = criterion(output, gt) + l2_lambda * l2_reg

            # 加入感知loss
            mse_loss = criterion(output, gt)
            perceptual = lpips_loss(output.permute(0,3,1,2), gt.permute(0,3,1,2)).mean()
            loss = mse_loss + perceptual_lambda * perceptual
            # loss = mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        # 每个epoch更新学习率
        scheduler.step()

        val_loss = 0.0
        val_perceptual_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, gt in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}"):
                inputs, gt = inputs.to(device), gt.to(device)
                output = model(inputs)
                mse = criterion(output, gt)
                val_loss += mse.item()

                perceptual = lpips_loss(output.permute(0,3,1,2), gt.permute(0,3,1,2)).mean()
                val_perceptual_loss += perceptual.item()

        # 计算平均训练损失和验证损失，并限制精度
        train_loss_avg = round(train_loss / len(train_loader), precision_limit)
        val_loss_avg = round(val_loss / len(val_loader), precision_limit)
        val_perceptual_loss_avg = round(val_perceptual_loss / len(val_loader), precision_limit)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss_avg :.4f}, "
              f"Val Loss = {val_loss_avg :.4f},"
              f"Val perceptual Loss = {val_perceptual_loss_avg :.4f},"
              )

        # wandb logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            # "val_perceptual_loss": val_perceptual_loss_avg,
        })

        # 可视化权重分布，观察最大的权重，看图像的区域依赖于哪一个版本
        # (H, W, C, K) → (H, W, C)
        soft_weights = torch.softmax(model.weights.detach(), dim=-1).cpu().numpy()  # (H, W, C, K)
        max_indices = soft_weights.argmax(axis=-1)
        # 可视化为一张图：不同版本 index 映射为颜色
        # 比如只取红色通道的权重 index 可视化：
        plt.imshow(max_indices[:, :, 0], cmap="viridis")
        plt.title("Dominant Version Index (Channel 0)")
        plt.colorbar()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({"dominant_version_map": wandb.Image(image)}, step=epoch+1)
        plt.close()

        # 可视化模型在融合的时候对哪一个区域是不确定的。
        # 熵越高说明大家的权重差不多，犹豫选取哪一个，反之果断。
        entropy = -np.sum(soft_weights * np.log(soft_weights + 1e-8), axis=-1)  # shape: (H, W, C)
        entropy_map = entropy.mean(axis=-1)  # 按通道平均 → (H, W)

        plt.imshow(entropy_map, cmap="hot")
        plt.title("Weight Entropy Map (avg over channels)")
        plt.colorbar()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({"entropy_map": wandb.Image(image)}, step=epoch+1)
        plt.close()

         # Save best model
        # if val_loss_avg + val_perceptual_loss_avg * perceptual_lambda < best_val_loss:
        #     best_val_loss = val_loss_avg + val_perceptual_loss_avg * perceptual_lambda
        #     model_path = "best_pixelwise_model.pth"
        #     torch.save(model.state_dict(), model_path)
        #     print("✅ Best model saved.")

        if val_loss_avg  < best_val_loss:
            best_val_loss = val_loss_avg 
            model_path = "best_pixelwise_model.pth"
            torch.save(model.state_dict(), model_path)
            print("✅ Best model saved.")

            # log model artifact to wandb
            # artifact = wandb.Artifact("best_pixelwise_model", type="model")
            # artifact.add_file(model_path)
            # wandb.log_artifact(artifact)

        # # Early stopping check
        # if early_stopping(val_loss_avg + val_perceptual_loss_avg * perceptual_lambda, model):
        #     print(f"Early stopping at epoch {epoch+1}.")
        #     break

        if early_stopping(val_loss_avg , model):
            print(f"Early stopping at epoch {epoch+1}.")
            break

        # 每隔 save_every 保存一次模型
        if (epoch + 1) % save_every == 0:
            ckpt_path = f"pixelFuse_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"📦 Checkpoint saved: {ckpt_path}")
        

    wandb.finish()
    return model

if __name__ == "__main__":
    folder_paths = {
        "blur_haze": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_haze",
        "blur_jpeg": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_jpeg",
        "blur_lowlight": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_lowlight",
        "blur_noise": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_noise",
        "blur_noise_jpeg": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_noise_jpeg",
        "blur_rain": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_rain",
        "blur_raindrop": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_raindrop",
        "deblur": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_deblur",
        "dehaze": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_dehaze",
        "denoise": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_denoise",
        "derain": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_derain",
        "deraindrop": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_deraindrop",
        "haze_jpeg": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_jpeg",
        "haze_lowlight": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_lowlight",
        "haze_noise": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_noise",
        "haze_rain": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_rain",
        "haze_raindrop": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_raindrop",
        "jpeg": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_jpeg",
        "lowlight": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight",
        "lowlight_blur_jpeg": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_blur_jpeg",
        "lowlight_blur_noise": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_blur_noise",
        "lowlight_jpeg": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_jpeg",
        "lowlight_noise": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_noise",
        "lowlight_rain": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_rain",
        "lowlight_raindrop": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_raindrop",
        "noise_jpeg": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_jpeg",
        "noise_rain": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_rain",
        "noise_raindrop": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_raindrop",
        "rain_jpeg": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_rain_jpeg",
        "raindrop_jpeg": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_raindrop_jpeg",
        "sr": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_sr",
        "universal": "../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_universal",
    }

    # 定义输入图像路径
    input_dirs = [value for key, value in folder_paths.items()]
    gt_dir = "../../data/DIV2K_Flickr_LSDIR5000/gt_1000name"

    # 训练模型
    model = train_pixel_fusion(
        input_dirs=input_dirs,
        gt_dir=gt_dir,
        height=512,
        width=512,
        channels=3,
        num_inputs=32,
        batch_size=64,
        epochs=1000,
        lr=1e-5,
        val_ratio=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )