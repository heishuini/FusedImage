import pyiqa
import os
import argparse
from pathlib import Path
import torch
from utils import util_image
import tqdm
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device) 

def evaluate(in_path, ref_path, folder_type, id, ntest):
    # No-ref metrics: MUSIQ, CLIPIQA
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)

    # For reference-based metrics: PSNR, SSIM, LPIPS
    metric_paired_dict = {}
    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])
        
        if ntest is not None: 
            ref_path_list = ref_path_list[:ntest]
        metric_paired_dict["psnr"] = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        metric_paired_dict["lpips"] = pyiqa.create_metric('lpips',pretrained_model_path='LPIPS_v0.1_alex-df73285e.pth').to(device)
        metric_paired_dict["ssim"] = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)

    # Process input images
    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None:
        lr_path_list = lr_path_list[:ntest]
    print(f'Find {len(lr_path_list)} images in {in_path}')

    # List to store per-image data as dictionaries
    results_data = []

    # Process images
    for i in tqdm.tqdm(range(len(lr_path_list))):
        current_in_path = lr_path_list[i]
        current_ref_path = ref_path_list[i] if ref_path_list is not None else None

        # Load input image and convert to tensor
        im_in = util_image.imread(current_in_path, chn='rgb', dtype='float32')
        im_in_tensor = util_image.img2tensor(im_in).to(device)

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
            im_ref_tensor = util_image.img2tensor(im_ref).to(device)

            im_in = util_image.imread(current_in_path, chn='rgb', dtype='float32')
            im_in_tensor = util_image.img2tensor(im_in).to(device)
            
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
    df.to_csv(f"../../data/DIV2K_Flickr_LSDIR5000/{folder_type}/{id}_evaluation_results.csv", index=False)
    
    # 返回avg结果
    return avg_row

if __name__ == "__main__":

    in_path_restore_dict = {
            "llava":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_llava",
            "universal":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_universal",
            "blur_haze":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_haze",
            "blur_jpeg":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_jpeg",
            "blur_lowlight":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_lowlight",
            "blur_noise":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_noise",
            "blur_noise_jpeg":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_noise_jpeg",
            "blur_rain":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_rain",
            "blur_raindrop":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_blur_raindrop",
            "deblur":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_deblur",
            "dehaze":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_dehaze",
            "denoise":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_denoise",
            "derain":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_derain",
            "deraindrop":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_deraindrop",
            "haze_jpeg":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_jpeg",
            "haze_lowlight":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_lowlight",
            "haze_noise":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_noise",
            "haze_rain":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_rain",
            "haze_raindrop":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_haze_raindrop",
            "jpeg":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_jpeg",
            "lowlight":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight",
            "lowlight_blur_jpeg":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_blur_jpeg",
            "lowlight_blur_noise":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_blur_noise",
            "lowlight_jpeg":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_jpeg",
            "lowlight_noise":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_noise",
            "lowlight_rain":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_rain",
            "lowlight_raindrop":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_lowlight_raindrop",
            "noise_jpeg":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_jpeg",
            "noise_rain":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_rain",
            "noise_raindrop":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_noise_raindrop",
            "rain_jpeg":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_rain_jpeg",
            "raindrop_jpeg":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_raindrop_jpeg",
            "sr":"../../data/DIV2K_Flickr_LSDIR5000/restore/sr_omini_sr",
    }
    
    in_path_restore_dict = {"pixel_result_mse_lpips":"../../data/DIV2K_Flickr_LSDIR5000/result_pixel_mse_lpips",}

    # ref_path_512 = "../../RealESRGAN/crop512_DIV2K_VAL"

    ref_path_512 = "../../data/DIV2K_Flickr_LSDIR5000/gt_1000name"
    # ref_path_512 = None
    # ref_path_512 = None

    all_avg_rows = []
    for key,value in in_path_restore_dict.items():
        avg_row = evaluate(value, ref_path_512, folder_type="restoreEvaluation",id=key,ntest=None)
        avg_row['id'] = key
        all_avg_rows.append(avg_row)

    # 创建 DataFrame 并保存所有版本的平均指标
    # df_all_avg = pd.DataFrame(all_avg_rows)
    # df_all_avg.to_csv("../../data/DIV2K_Flickr_LSDIR5000/restoreEvaluation/summary_avg_results.csv", index=False)

    # in_path_lq_512 = "../../data/DIV2K/crop_1024/x2/lq_512" 
    # in_path_lq_1024 = "../../data/DIV2K/crop_1024/lq_1024/lq"

    # evaluate(in_path_lq_1024, ref_path_1024, folder_type="lqEvaluation",id="lq_1024",ntest=None)

    # weighted
    # in_path_weighted = "../../data/DIV2K/crop_1024/x2/weighted/resultImage"
    # evaluate(in_path_weighted, ref_path_1024, folder_type="weighted",id="weighted_1024",ntest=None)