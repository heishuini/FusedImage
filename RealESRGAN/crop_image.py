import os
import cv2
import numpy as np
from tqdm import tqdm

def center_crop(
    img: np.ndarray, 
    crop_size: tuple, 
    pad_mode: str = 'reflect'
) -> np.ndarray:
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



def batch_center_crop_folders(
    input_dirs: list,
    output_dirs: list,
    crop_size: tuple,
    pad_mode: str = 'reflect',
    image_exts: tuple = ('.jpg', '.png', '.jpeg', '.bmp', '.tiff')
):
    """
    对多个文件夹中的图像进行中心裁剪并保存结果。
    
    参数：
        input_dirs: 输入图像文件夹列表
        output_dirs: 裁剪后图像保存的文件夹列表
        crop_size: 裁剪尺寸 (H, W)
        pad_mode: 填充方式，支持'reflect', 'constant', 'edge'
        image_exts: 支持的图像扩展名
    """
    assert len(input_dirs) == len(output_dirs), "输入输出文件夹数量不一致"

    for in_dir, out_dir in zip(input_dirs, output_dirs):
        os.makedirs(out_dir, exist_ok=True)

        img_files = [f for f in os.listdir(in_dir) if f.lower().endswith(image_exts)]
        for img_name in tqdm(img_files, desc=f'Processing {in_dir}'):
            img_path = os.path.join(in_dir, img_name)
            out_path = os.path.join(out_dir, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue

            img = img if img.ndim == 3 else np.expand_dims(img, axis=-1)
            img_cropped = center_crop(img, crop_size, pad_mode)

            # 写出时根据通道数自动判断保存格式
            if img_cropped.shape[2] == 1:
                img_cropped = img_cropped[:, :, 0]
            cv2.imwrite(out_path, img_cropped)

    print("所有图像裁剪完成！")

# DIV2K_Flickr2K_LSDIR_TRAIN
# input_folders = [
#             '../data/DIV2K/DIV2K_train_HR',  # 800
#             '../data/Flickr2K/Flickr2K_HR',  # 2650
#             '../data/LSDIR/train', # 5000 
# ]
# output_folders = [
#     './cropped_512/DIV2K_HR',
#     './cropped_512/Flickr2K_HR',
#     './cropped_512/LSDIR_HR'
# ]

# DIV2K_VAL
# input_folders = [
#     './val_HR'
#     ]

# output_folders = [
#     './crop512_DIV2K_VAL',
# ]

# RealPhoto60
# input_folders = [
#     '../data/RealPhoto60/LQ'
#     ]

# output_folders = [
#     './crop512_RealPhoto60',
# ]

# RealDeg
input_folders = [
    '../data/RealDeg/classic_film',
    '../data/RealDeg/old_photo',
    '../data/RealDeg/social_media',
    ]

output_folders = [
    './crop512_RealDeg/classic_film',
    './crop512_RealDeg/old_photo',
    './crop512_RealDeg/social_media',
]


batch_center_crop_folders(input_folders, output_folders, crop_size=(512, 512))

# save_dir_dict = {
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_llava": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_llava",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_universal": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_universal",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_haze": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_haze",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_jpeg",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_lowlight": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_lowlight",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_noise": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_noise",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_noise_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_noise_jpeg",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_rain": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_rain",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_raindrop": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_raindrop",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_deblur": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_deblur",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_dehaze": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_dehaze",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_denoise": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_denoise",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_derain": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_derain",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_deraindrop": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_deraindrop",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_haze_jpeg",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_lowlight": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_haze_lowlight",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_noise": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_haze_noise",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_rain": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_haze_rain",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_raindrop": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_haze_raindrop",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_jpeg",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_blur_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_blur_jpeg",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_blur_noise": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_blur_noise",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_jpeg",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_noise": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_noise",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_rain": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_rain",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_raindrop": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_raindrop",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_noise_jpeg",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_rain": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_noise_rain",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_raindrop": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_noise_raindrop",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_rain_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_rain_jpeg",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_raindrop_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_raindrop_jpeg",
#     "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_sr": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_sr",
# }