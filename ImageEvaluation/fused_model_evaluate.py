from fused_model_LR_task import LinearFusionModel
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyiqa
from utils import util_image
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def readImagePath(folder_paths):
    all_images = {}
    for method, folder_path in folder_paths.items():
        folder = Path(folder_path)
        image_files = sorted([f for f in folder.glob("*.png")])
        all_images[method] = image_files
    return all_images

def evaluate(in_path, ref_path, ntest=None):
    # No-ref metrics: MUSIQ, CLIPIQA
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(DEVICE)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(DEVICE)

    # For reference-based metrics: PSNR, SSIM, LPIPS
    metric_paired_dict = {}
    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])
        
        if ntest is not None: 
            ref_path_list = ref_path_list[:ntest]
        metric_paired_dict["psnr"] = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(DEVICE)
        metric_paired_dict["lpips"] = pyiqa.create_metric('lpips',pretrained_model_path='LPIPS_v0.1_alex-df73285e.pth').to(DEVICE)
        metric_paired_dict["ssim"] = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(DEVICE)

    # Process input images
    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None:
        lr_path_list = lr_path_list[:ntest]
    print(f'Find {len(lr_path_list)} images in {in_path}')

    # List to store per-image data as dictionaries
    results_data = []

    # Process images
    for i in tqdm(range(len(lr_path_list))):
        current_in_path = lr_path_list[i]
        current_ref_path = ref_path_list[i] if ref_path_list is not None else None

        # Load input image and convert to tensor
        im_in = util_image.imread(current_in_path, chn='rgb', dtype='float32')
        im_in_tensor = util_image.img2tensor(im_in).to(DEVICE)

        # Create a dictionary for this image's metrics
        image_data = {
            'image_path': str(current_in_path),
            'ref_path': str(current_ref_path) if current_ref_path else None,
        }

        # Assess no-reference metrics
        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                metric_value = metric(im_in_tensor).item()
            image_data[key] = metric_value
            print(f"{current_in_path}: {key}: {metric_value:.5f}")

        # Assess reference-based metrics if available
        if current_ref_path is not None:
            
            im_ref = util_image.imread(current_ref_path, chn='rgb', dtype='float32')
            
            # print(type(im_ref))
            im_ref_tensor = util_image.img2tensor(im_ref).to(DEVICE)

            im_in = util_image.imread(current_in_path, chn='rgb', dtype='float32')
            im_in_tensor = util_image.img2tensor(im_in).to(DEVICE)
            
            for key, metric in metric_paired_dict.items():
                metric_value = metric(im_in_tensor, im_ref_tensor).item()
                image_data[key] = metric_value
                print(f"{current_in_path} vs {current_ref_path}: {key}: {metric_value:.5f}")

        # Append this image's data to the results
        results_data.append(image_data)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(results_data)

    # Compute averages
    avg_row = {
        'image_path': 'Average',
        'ref_path': None,
    }
    # Averages for no-reference metrics (always present)
    for key in metric_dict.keys():
        avg_row[key] = df[key].mean()
    # Averages for reference-based metrics (if applicable)
    if ref_path is not None:
        for key in metric_paired_dict.keys():
            avg_row[key] = df[key].mean()

    # Print averages
    print('avg:')
    for key, val in avg_row.items():
        if key not in ['image_path', 'ref_path']:
            print(f"{key}: {val:.5f}")

    # Append the average row to the DataFrame
    avg_df = pd.DataFrame([avg_row])
    df = pd.concat([df, avg_df], ignore_index=True)

    # Save to CSV file
    df.to_csv("../../data/DIV2K/crop_1024/x2/resultEvaluation/crop_1024_evaluation.csv", index=False)
    


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
    print("Softmax后权重:", weights) 
    # 对比线性归一化
    # linear_weights = scores / np.sum(scores)
    # print("线性权重:", linear_weights) 
    # [0.11574811 0.13057238 0.12488096 0.12296596 0.1275074  0.12830867 0.12473699 0.12527954]

    # 初始化结果图像
    result = np.zeros_like(images[0], dtype=np.float32)

    # 加权融合， 逐步进行
    for i,(img, weight) in enumerate(zip(images, weights)):
        # imread时转为float32，又除了255，此时是[0,1]之间
        img_float = img
        result += weight * img_float

        # 要求result是[-1,1]之间
        # normalized_immediate = np.clip(result, -1, 1)
        # util_image.imwrite(normalized_immediate,os.path.join('../../data/DIV2K/weighted_x4/immediate',f'{image_id:03d}_step_{i:02d}.png'))
        
    # clip是确保值在0~255
    normalized_img = np.clip(result, -1, 1)
    util_image.imwrite(normalized_img,os.path.join('../../data/DIV2K/crop_1024/x2/resultImage',f'result_{image_id:03d}.png'))
    # cv2.imwrite('output.png', normalized_img)  # OpenCV要求uint8类型



if __name__ == "__main__":

    folder_paths = {
        "blur_haze": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_blur_haze",
        "blur_jpeg": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_blur_jpeg",
        "blur_lowlight": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_blur_lowlight",
        "blur_noise": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_blur_noise",
        "blur_noise_jpeg": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_blur_noise_jpeg",
        "blur_rain": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_blur_rain",
        "blur_raindrop": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_blur_raindrop",
        "deblur": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_deblur",
        "dehaze": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_dehaze",
        "denoise": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_denoise",
        "derain": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_derain",
        "deraindrop": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_deraindrop",
        "haze_jpeg": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_haze_jpeg",
        "haze_lowlight": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_haze_lowlight",
        "haze_noise": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_haze_noise",
        "haze_rain": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_haze_rain",
        "haze_raindrop": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_haze_raindrop",
        "jpeg": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_jpeg",
        "lowlight": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_lowlight",
        "lowlight_blur_jpeg": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_lowlight_blur_jpeg",
        "lowlight_blur_noise": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_lowlight_blur_noise",
        "lowlight_jpeg": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_lowlight_jpeg",
        "lowlight_noise": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_lowlight_noise",
        "lowlight_rain": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_lowlight_rain",
        "lowlight_raindrop": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_lowlight_raindrop",
        "noise_jpeg": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_noise_jpeg",
        "noise_rain": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_noise_rain",
        "noise_raindrop": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_noise_raindrop",
        "rain_jpeg": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_rain_jpeg",
        "raindrop_jpeg": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_raindrop_jpeg",
        "sr": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_sr",
        "universal": "../../data/DIV2K/crop_1024/x2/restore/sr_omini_universal",
    }

    folder_gt_path = {"gt": "../../data/DIV2K/crop_1024/x2/gt_1024"}

    model = LinearFusionModel(input_dim=len(folder_paths)).to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval()

    # 读取模型的系数
    weights = model.linear.weight.data.cpu().numpy().flatten()
    final_bias = model.linear.bias.data.cpu().numpy()[0]

    for i, method in enumerate(folder_paths.keys()):
        print(f"{method:25s}: {weights[i]:.6f}")
    print(f"\n🧮 偏置项(bias): {final_bias:.6f}")
    
    print(weights.shape)
    ##### 处理输入
    # 读取所有方法处理后的图像路径
    all_images = readImagePath(folder_paths)
    num_images = len(all_images[list(all_images.keys())[0]])

    for i in tqdm(range(num_images), desc="Processing images"):
        img_result = {"image_id": i+1}
        im_in_list = []

        # 各类别图像读取
        for method in folder_paths.keys():
            img_path = all_images[method][i]
            
            # 加载图像并融合
            # 转为numpy，值是0~1
            im_in = util_image.imread(img_path, chn='rgb', dtype='float32')
            im_in_list.append(im_in)

        # 简单的加权融合
        weighted_fusion(im_in_list,weights,image_id=i)
    ####
        
# 评估
in_path = "../../data/DIV2K/crop_1024/x2/resultImage"
ref_path = "../../data/DIV2K/crop_1024/x2/gt_1024"
evaluate(in_path, ref_path)
    



