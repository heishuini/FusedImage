import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from utils import util_image
from sklearn.model_selection import train_test_split
import logging
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import Dataset, DataLoader

PATCH_SIZE = (16, 16)
BATCH_SIZE = 4
VAL_RATIO = 0.1
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def readImagePath(folder_paths):
    all_images = {}
    for method, folder_path in folder_paths.items():
        folder = Path(folder_path)
        image_files = sorted([f for f in folder.glob("*.png")])
        all_images[method] = image_files
    return all_images


def extract_patches(image_numpy, patch_size=(64, 64)):
    H, W, C = image_numpy.shape
    ph, pw = patch_size
    patches = []

    for i in range(0, H, ph):
        for j in range(0, W, pw):
            if i + ph <= H and j + pw <= W:
                patch = image_numpy[i:i+ph, j:j+pw, :]
                patches.append(patch)
    return patches

def evaluate_regression(Y_true, Y_pred):
    """计算回归任务评估指标"""
    metrics = {
        'MSE': np.mean((Y_true - Y_pred)**2),
        'PSNR': peak_signal_noise_ratio(Y_true, Y_pred, data_range=1.0),
        'SSIM': structural_similarity(
            Y_true, Y_pred, 
            multichannel=True,  # 对于RGB图像
            data_range=1.0
        )
    }
    return metrics

class PatchFusionDataset(Dataset):
    def __init__(self, folder_paths, all_images, indices, gt_im_in_list=None, patch_size=PATCH_SIZE, transform=None):
        self.folder_paths = folder_paths
        self.gt_im_in_list = gt_im_in_list
        self.all_images = all_images
        self.patch_size = patch_size
        self.transform = transform
        self.indices = indices  # 子集索引

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 读取图片数据
        im_in_list = []
        for method in self.folder_paths.keys():
            img_path = self.all_images[method][idx]
            im = util_image.imread(img_path, chn='rgb', dtype='float32')
            im_in_list.append(im)

        # 读取GT图片
        gt_img = self.gt_im_in_list[idx]

        # 提取patch
        input_patches = list(zip(*[self.extract_patches(im) for im in im_in_list]))
        gt_patches = self.extract_patches(gt_img)

        # patch平均值
        patch_X = []
        patch_Y = []
        
        for p_input, p_gt in zip(input_patches, gt_patches):
            x = np.array([patch.mean() for patch in p_input], dtype=np.float32)
            y = np.array([p_gt.mean()], dtype=np.float32)
            patch_X.append(x)
            patch_Y.append(y)

        # 转换为tensor
        X_tensor = torch.from_numpy(np.array(patch_X, dtype=np.float32)) # (num_patches, n_methods)
        Y_tensor = torch.from_numpy(np.array(patch_Y, dtype=np.float32)) # (num_patches, 1)

        return X_tensor, Y_tensor
    
    def extract_patches(self, img):
        """返回提取的patch"""
        patches = []
        height, width, _ = img.shape
        patch_height, patch_width = self.patch_size

        # 假设你按patch_size遍历图像
        for i in range(0, height - patch_height + 1, patch_height):
            for j in range(0, width - patch_width + 1, patch_width):
                patch = img[i:i + patch_height, j:j + patch_width]
                patches.append(patch)

        return patches

# 创建DataLoader
def create_dataloader(folder_paths, gt_im_in_list, all_images, val_ratio=0.1, batch_size=8, shuffle=True):
    num_images = len(gt_im_in_list)
    indices = list(range(num_images))
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=42)

    train_dataset = PatchFusionDataset(folder_paths, gt_im_in_list=gt_im_in_list,all_images=all_images, indices=train_idx)
    val_dataset = PatchFusionDataset(folder_paths, gt_im_in_list=gt_im_in_list,all_images=all_images, indices=val_idx)

    # batch_size: 将getitem的内容多运行batch_size次，返回得到batch_size个tensor然后拼为一个batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader

class LinearFusionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearFusionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)

# 每个图片分块patches，取patches的平均值做运算，每个patches分batch
def SGD_fit_patchwise(folder_paths, train_loader, val_loader):
    # 初始化模型和优化器
    input_dim = len(folder_paths)  # 方法数
    model = LinearFusionModel(input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    # weight_decay是L2正则化惩罚项
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 评估指标记录器
    history = {
        'train_MSE': [],
        'val_MSE': [],
        'train_PSNR': [],
        'val_PSNR': [],
        'weights': []
    }

    # 🧠 EarlyStopping 配置
    patience = 5
    best_loss = float('inf')
    counter = 0
    best_model_state = None
    model_save_dir = './'
    best_model_path = os.path.join(model_save_dir, "best_model_task.pth")

    
    for epoch in range(EPOCHS):
        epoch_loss = {'train': [], 'val': []}
        epoch_psnr = {'train': [], 'val': []}

        for phase, dataloader in zip(['train', 'val'], [train_loader, val_loader]):
            model.train() if phase == 'train' else model.eval()
            for inputs, targets in tqdm(dataloader, desc=f"[{phase.upper()}] Epoch {epoch}"):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                if phase == 'train':
                    optimizer.zero_grad()
                    pred = model(inputs)
                    loss = criterion(pred, targets)
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        mse = loss.item()
                        psnr = 10 * np.log10(1 / mse)
                        wandb.log({
                            f"{phase}/step_loss": mse,
                            f"{phase}/step_psnr": psnr,
                        })
                        epoch_loss[phase].append(mse)
                        epoch_psnr[phase].append(psnr)

                else:
                    with torch.no_grad():
                        pred = model(inputs)
                        loss = criterion(pred, targets)
                        mse = loss.item()
                        psnr = 10 * np.log10(1 / mse)
                        wandb.log({
                            f'{phase}/step_loss': mse,
                            f'{phase}/step_psnr': psnr,
                        })
                        epoch_loss[phase].append(mse)
                        epoch_psnr[phase].append(psnr)

            mean_loss = np.mean(epoch_loss[phase])
            mean_psnr = np.mean(epoch_psnr[phase])           
            history[f"{phase}_MSE"].append(mean_loss)
            history[f"{phase}_PSNR"].append(mean_psnr)
            wandb.log({
                f"{phase}/epoch_loss": mean_loss,
                f"{phase}/epoch_psnr": mean_psnr,
                f'{phase}/epoch_lr': optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })         

            logging.info(f"[Epoch {epoch} {phase.upper()}] Mean MSE: {mean_loss:.4f} | Mean PSNR: {mean_psnr:.2f}")
            # 记录epoch级别模型参数
            # 提取模型参数
            weights = model.linear.weight.data.cpu().numpy().flatten()
            bias = model.linear.bias.item()

        # 🛑 EarlyStopping & 保存模型
        mean_loss = np.mean(epoch_loss['val'])
        mean_loss = round(mean_loss, 4)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, best_model_path)
            logging.info(f"💾 [BEST] Saved model at epoch {epoch} with Val MSE: {mean_loss:.4f}")

        else:
            counter += 1
            if counter >= patience:
                logging.info(f"🛑 Early stopping triggered at epoch {epoch} — no improvement for {patience} epochs.")
                break

    final_weights = model.linear.weight.data.cpu().numpy().flatten()
    final_bias = model.linear.bias.data.cpu().numpy()[0]

    # 打印
    print("🔍 融合权重（每种增强图像的系数）:")
    for i, method in enumerate(folder_paths.keys()):
        print(f"{method:25s}: {final_weights[i]:.6f}")
    print(f"\n🧮 偏置项(bias): {final_bias:.6f}")

    import datetime
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"fusion_model_task.pth")

    # 训练完成后绘图
    # plt.figure(figsize=(12, 5))

    # # MSE 曲线
    # plt.subplot(1, 2, 1)
    # plt.plot(history['train_MSE'], label='Train MSE', color='blue', alpha=0.7)
    # plt.plot(history['val_MSE'], label='Validation MSE', color='orange', alpha=0.7)
    # plt.xlabel("Step")
    # plt.ylabel("MSE")
    # plt.title("MSE Curve")
    # plt.legend()
    # plt.grid(True)

    # # PSNR 曲线
    # plt.subplot(1, 2, 2)
    # plt.plot(history['train_PSNR'], label='Train PSNR', color='green', alpha=0.7)
    # plt.plot(history['val_PSNR'], label='Validation PSNR', color='red', alpha=0.7)
    # plt.xlabel("Step")
    # plt.ylabel("PSNR (dB)")
    # plt.title("PSNR Curve")
    # plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"loss_curve_{now}.png")
    plt.show()

    return final_weights, final_bias, history

def inference(model):
    # 构建模型结构
    input_dim = 32
    model = LinearFusionModel(input_dim)
    model.load_state_dict(torch.load('fusion_model.pt'))
    model.to(DEVICE)
    model.eval()

if __name__ == '__main__':
    wandb.init(
        project="image-fusion",
        name="patchwise_adam_fusion",
        config={
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "patch_size": PATCH_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "val_ratio": VAL_RATIO,
              "optimizer": "Adam",
              "loss": "MSEloss"
        }
    )

    # wandb.init(
    #     project="image-fusion",
    #     name="imagewise_sgd_fusion",
    #     config={
    #         "learning_rate": 1e-4,
    #         "weight_decay": 1e-5,
    #         "batch_image": IMAGE_BATCH_SIZE,
    #         "epochs": EPOCHS,
    #         "val_ratio": VAL_RATIO,
    #         "optimizer": "SGD",
    #         "loss": "MSEloss"
    #     }
    # )

    # 定义32个文件夹路径
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

    folder_gt_path = {"gt": "../../data/DIV2K_Flickr_LSDIR5000/gt_1000name"}

    # 验证所有文件夹存在
    for name, path in folder_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found: {path}")

    # 读取训练图像路径
    all_images = readImagePath(folder_paths)

    # 一个类别的图片数量
    num_images = len(all_images[list(all_images.keys())[0]])

    # 读取gt图像路径
    gt_images = readImagePath(folder_gt_path)

    for k,v in all_images.items():
        print(k)

    # 读取gt图像
    gt_im_in_list = []
    for i in tqdm(range(num_images), desc="Reading gt"):
        for method in folder_gt_path.keys():
            img_path = gt_images[method][i]
        
            # 加载图像
            try:
                # (H,W,C),值为0~1
                gt_im_in = util_image.imread(img_path, chn='rgb', dtype='float32')
                gt_im_in_list.append(gt_im_in)
                # print(len(im_in_list))
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

    # 生成DataLoader
    train_loader, val_loader = create_dataloader(folder_paths, gt_im_in_list, all_images, val_ratio=VAL_RATIO, batch_size=BATCH_SIZE, shuffle=True)

    weights, bias, history = SGD_fit_patchwise(folder_paths, train_loader, val_loader)

    weight_dict = {method: float(w) for method, w in zip(folder_paths.keys(), weights)}
    wandb.log({
        "final_weights": weight_dict,
        "final_bias": bias
    })

    wandb.finish()