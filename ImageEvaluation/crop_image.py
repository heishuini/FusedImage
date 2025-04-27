import cv2
import numpy as np
import os
from glob import glob

def center_crop(img: np.ndarray, crop_size: tuple, pad_mode: str = 'reflect') -> np.ndarray:
    """
    标准中心裁剪（自动处理尺寸不足情况）
    
    参数：
        img: 输入图像，uint8或float32格式
        crop_size: 目标尺寸 (height, width)
        pad_mode: 填充方式 ('reflect', 'constant', 'edge')
    
    返回：
        裁剪后图像，保持原始数据类型
    """
    # 参数校验
    assert len(img.shape) in [2, 3], "输入必须为HWC或HW格式"
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    
    h, w = img.shape[:2]
    crop_h, crop_w = min(crop_size[0], h), min(crop_size[1], w)
    
    # 计算裁剪区域
    start_y = max(0, (h - crop_h) // 2)
    start_x = max(0, (w - crop_w) // 2)
    cropped = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    # 尺寸不足时填充
    if cropped.shape[0] != crop_size[0] or cropped.shape[1] != crop_size[1]:
        pad_h = max(0, crop_size[0] - cropped.shape[0])
        pad_w = max(0, crop_size[1] - cropped.shape[1])
        
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_h//2, pad_h - pad_h//2,
            pad_w//2, pad_w - pad_w//2,
            borderType={
                'reflect': cv2.BORDER_REFLECT,
                'constant': cv2.BORDER_CONSTANT,
                'edge': cv2.BORDER_REPLICATE
            }[pad_mode],
            value=0 if img.dtype == np.uint8 else 0.0
        )
    
    return cropped

def process_folder(input_folder, output_folder, crop_size=(1024, 1024)):
    """
    处理整个文件夹中的图像
    
    参数：
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        crop_size: 目标裁剪尺寸
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件（支持常见格式）
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(input_folder, ext)))
    
    # 处理每张图像
    for img_path in image_paths:
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue
            
            # 中心裁剪
            cropped = center_crop(img, crop_size)
            
            # 保存结果
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped)
            
            print(f"已处理: {img_path} -> {output_path}")
            
        except Exception as e:
            print(f"处理 {img_path} 时出错: {str(e)}")

# 使用示例
input_folder = "../../RealESRGAN/val_HR"  # 输入文件夹路径
output_folder = "../../RealESRGAN/crop_val_HR_1024"  # 输出文件夹路径

process_folder(input_folder, output_folder, crop_size=(1024, 1024))