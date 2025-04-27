# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# gt_img = Image.open('../../data/DIV2K/x4/gt/gt_001.png')

# res_deblur = Image.open('../../data/DIV2K/x4/sr_omini_dehaze/lq_001.png')

# # 转换为 NumPy 数组 (形状: [H, W, 3], 范围: 0-255)
# gt_array = np.array(gt_img).astype(np.float32)
# res_array = np.array(res_deblur).astype(np.float32)

# # 计算残差 (绝对值更直观)
# residual = np.abs(gt_array - res_array)  # shape: [H, W, 3]

# # 将残差缩放到 0-255 并转换为 uint8
# residual_vis = (residual / np.max(residual) * 255).astype(np.uint8)

# # 显示残差图
# plt.figure(figsize=(10, 5))
# plt.imshow(residual_vis)
# plt.title("Absolute Pixel-wise Difference (Bright = Large Error)")
# plt.axis('off')
# plt.show()
# plt.savefig('residual.png')


# plt.figure(figsize=(10, 5))
# # plt.imshow(gt_img)  # 原图作为底图
# # 用热力图叠加残差（均值化三通道）
# residual_mean = np.mean(residual, axis=2)  # 合并RGB通道
# plt.imshow(residual_mean, cmap='hot', alpha=1)  # 半透明热力图
# plt.colorbar(label="Difference Intensity")
# plt.title("Residual Overlay on Ground Truth (Red = Large Error)")
# plt.axis('off')
# plt.show()

# plt.savefig('residual_hot.png')

# # 计算全局差值（以R通道为例）
# diff_r = gt_array[:, :, 0] - res_array[:, :, 0]
# mean = diff_r.mean()
# print(mean) 
# # 0.44961464
# # 0.4466026
# #


# # 保存为文本文件（每行一个像素行）
# np.savetxt('pixel_diff_r_channel.txt', diff_r, fmt='%d')  # %d表示整数格式
import cv2
import numpy as np

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

# # 输入
# img = cv2.imread("../../RealESRGAN/val_HR/DIV2K_HR_001.png")  # 假设原图尺寸 (512, 768, 3)
# print(img.shape) # (H,W,3)
# cropped = center_crop(img, (1024, 1024))

# # 输出形状
# print(cropped.shape)  # (1024, 1024, 3)
# cv2.imwrite("../../RealESRGAN/crop_val_HR/crop_001.png",cropped)
save_dir_dict = {
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_haze": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_haze",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_jpeg",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_lowlight": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_lowlight",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_noise": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_noise",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_noise_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_noise_jpeg",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_rain": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_rain",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_raindrop": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_blur_raindrop",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_deblur": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_deblur",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_dehaze": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_dehaze",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_denoise": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_denoise",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_derain": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_derain",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_deraindrop": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_deraindrop",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_haze_jpeg",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_lowlight": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_haze_lowlight",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_noise": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_haze_noise",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_rain": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_haze_rain",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_raindrop": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_haze_raindrop",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_jpeg",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_blur_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_blur_jpeg",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_blur_noise": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_blur_noise",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_jpeg",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_noise": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_noise",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_rain": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_rain",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_raindrop": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_lowlight_raindrop",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_noise_jpeg",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_rain": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_noise_rain",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_raindrop": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_noise_raindrop",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_rain_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_rain_jpeg",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_raindrop_jpeg": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_raindrop_jpeg",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_sr": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_sr",
    "../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_universal": "../data/DIV2K_Flickr_LSDIR5000/caption/lq_caption_omini_universal",
}
print(len(save_dir_dict)) #32

# import os

# folder_path = '../../data/DIV2K/crop_1024/x2/gt_1024'  # 当前目录，或者你可以指定完整路径

# for filename in os.listdir(folder_path):
#     if filename.startswith('DIV2K_HR_') and filename.endswith('.png'):
#         # 提取数字部分
#         number = filename[9:-4]  # 去掉'DIV2K_HR_'和'.png'
#         new_name = f'lq_{number}.png'
        
#         # 重命名文件
#         old_path = os.path.join(folder_path, filename)
#         new_path = os.path.join(folder_path, new_name)
#         os.rename(old_path, new_path)

# print("重命名完成！")