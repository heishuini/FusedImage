import torch.cuda
import argparse
from FaithDiff.create_FaithDiff_model import FaithDiff_pipeline
from PIL import Image
from CKPT_PTH import SDXL_PATH, FAITHDIFF_PATH, VAE_FP16_PATH
from utils.color_fix import wavelet_color_fix, adain_color_fix
from utils.image_process import check_image_size
import os
import json


    
# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--json_dir", type=str)
parser.add_argument("--upscale", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min_size", type=int, default=1024)
parser.add_argument("--latent_tiled_overlap", type=float, default=0.5)
parser.add_argument("--latent_tiled_size", type=int, default=1024)
parser.add_argument("--guidance_scale", type=float, default=5)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--vae_tiled_overlap", type=float, default=0.25)
parser.add_argument("--vae_tiled_size", type=int, default=1024)
parser.add_argument("--color_fix", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr')
parser.add_argument("--cpu_offload", action='store_true', default=False)
parser.add_argument("--use_fp8", action='store_true', default=False)
args = parser.parse_args()
print(args)
cpu_offload = args.cpu_offload
use_fp8 = args.use_fp8

# load FaithDiff FP16
pipe = FaithDiff_pipeline(sdxl_path=SDXL_PATH, VAE_FP16_path=VAE_FP16_PATH, FaithDiff_path=FAITHDIFF_PATH, use_fp8=use_fp8)
pipe = pipe.to('cuda:0')

if args.use_tile_vae:
    ### enable_vae_tiling
    pipe.set_encoder_tile_settings()
    pipe.enable_vae_tiling()

if cpu_offload:
    pipe.enable_model_cpu_offload()

# DIV2K_Flickr_LSDIR
# DIV2K_VAL
# DIV2K/crop_512/restoreTest
save_dir_dict = {
    "data/DIV2K/crop_512/restoreTest/sr_llava": "data/DIV2K/crop_512/caption/lq_caption_llava",
    "data/DIV2K/crop_512/restoreTest/sr_omini_universal": "data/DIV2K/crop_512/caption/lq_caption_omini_universal",
    "data/DIV2K/crop_512/restoreTest/sr_omini_blur_haze": "data/DIV2K/crop_512/caption/lq_caption_omini_blur_haze",
    "data/DIV2K/crop_512/restoreTest/sr_omini_blur_jpeg": "data/DIV2K/crop_512/caption/lq_caption_omini_blur_jpeg",
    "data/DIV2K/crop_512/restoreTest/sr_omini_blur_lowlight": "data/DIV2K/crop_512/caption/lq_caption_omini_blur_lowlight",
    "data/DIV2K/crop_512/restoreTest/sr_omini_blur_noise": "data/DIV2K/crop_512/caption/lq_caption_omini_blur_noise",
    "data/DIV2K/crop_512/restoreTest/sr_omini_blur_noise_jpeg": "data/DIV2K/crop_512/caption/lq_caption_omini_blur_noise_jpeg",
    "data/DIV2K/crop_512/restoreTest/sr_omini_blur_rain": "data/DIV2K/crop_512/caption/lq_caption_omini_blur_rain",
    "data/DIV2K/crop_512/restoreTest/sr_omini_blur_raindrop": "data/DIV2K/crop_512/caption/lq_caption_omini_blur_raindrop",
    "data/DIV2K/crop_512/restoreTest/sr_omini_deblur": "data/DIV2K/crop_512/caption/lq_caption_omini_deblur",
    "data/DIV2K/crop_512/restoreTest/sr_omini_dehaze": "data/DIV2K/crop_512/caption/lq_caption_omini_dehaze",
    "data/DIV2K/crop_512/restoreTest/sr_omini_denoise": "data/DIV2K/crop_512/caption/lq_caption_omini_denoise",
    "data/DIV2K/crop_512/restoreTest/sr_omini_derain": "data/DIV2K/crop_512/caption/lq_caption_omini_derain",
    "data/DIV2K/crop_512/restoreTest/sr_omini_deraindrop": "data/DIV2K/crop_512/caption/lq_caption_omini_deraindrop",
    "data/DIV2K/crop_512/restoreTest/sr_omini_haze_jpeg": "data/DIV2K/crop_512/caption/lq_caption_omini_haze_jpeg",
    "data/DIV2K/crop_512/restoreTest/sr_omini_haze_lowlight": "data/DIV2K/crop_512/caption/lq_caption_omini_haze_lowlight",
    "data/DIV2K/crop_512/restoreTest/sr_omini_haze_noise": "data/DIV2K/crop_512/caption/lq_caption_omini_haze_noise",
    "data/DIV2K/crop_512/restoreTest/sr_omini_haze_rain": "data/DIV2K/crop_512/caption/lq_caption_omini_haze_rain",
    "data/DIV2K/crop_512/restoreTest/sr_omini_haze_raindrop": "data/DIV2K/crop_512/caption/lq_caption_omini_haze_raindrop",
    "data/DIV2K/crop_512/restoreTest/sr_omini_jpeg": "data/DIV2K/crop_512/caption/lq_caption_omini_jpeg",
    "data/DIV2K/crop_512/restoreTest/sr_omini_lowlight": "data/DIV2K/crop_512/caption/lq_caption_omini_lowlight",
    "data/DIV2K/crop_512/restoreTest/sr_omini_lowlight_blur_jpeg": "data/DIV2K/crop_512/caption/lq_caption_omini_lowlight_blur_jpeg",
    "data/DIV2K/crop_512/restoreTest/sr_omini_lowlight_blur_noise": "data/DIV2K/crop_512/caption/lq_caption_omini_lowlight_blur_noise",
    "data/DIV2K/crop_512/restoreTest/sr_omini_lowlight_jpeg": "data/DIV2K/crop_512/caption/lq_caption_omini_lowlight_jpeg",
    "data/DIV2K/crop_512/restoreTest/sr_omini_lowlight_noise": "data/DIV2K/crop_512/caption/lq_caption_omini_lowlight_noise",
    "data/DIV2K/crop_512/restoreTest/sr_omini_lowlight_rain": "data/DIV2K/crop_512/caption/lq_caption_omini_lowlight_rain",
    "data/DIV2K/crop_512/restoreTest/sr_omini_lowlight_raindrop": "data/DIV2K/crop_512/caption/lq_caption_omini_lowlight_raindrop",
    "data/DIV2K/crop_512/restoreTest/sr_omini_noise_jpeg": "data/DIV2K/crop_512/caption/lq_caption_omini_noise_jpeg",
    "data/DIV2K/crop_512/restoreTest/sr_omini_noise_rain": "data/DIV2K/crop_512/caption/lq_caption_omini_noise_rain",
    "data/DIV2K/crop_512/restoreTest/sr_omini_noise_raindrop": "data/DIV2K/crop_512/caption/lq_caption_omini_noise_raindrop",
    "data/DIV2K/crop_512/restoreTest/sr_omini_rain_jpeg": "data/DIV2K/crop_512/caption/lq_caption_omini_rain_jpeg",
    "data/DIV2K/crop_512/restoreTest/sr_omini_raindrop_jpeg": "data/DIV2K/crop_512/caption/lq_caption_omini_raindrop_jpeg",
    "data/DIV2K/crop_512/restoreTest/sr_omini_sr": "data/DIV2K/crop_512/caption/lq_caption_omini_sr",
}

# 单文件夹
# os.makedirs(args.save_dir, exist_ok=True)
# exist_file = os.listdir(args.save_dir)

# with torch.no_grad():
#     for file_name in sorted(os.listdir(args.img_dir)):
#         img_name, ext = os.path.splitext(file_name)
#         if ext == ".json":
#             continue
        
#         if f"{img_name}.png" in exist_file:
#             print(f"{img_name}.png exist")
#             continue
#         else:
#             print(img_name)

#         image = Image.open(os.path.join(args.img_dir,file_name)).convert('RGB')

#         json_file = json.load(open(os.path.join(args.json_dir,'gt'+img_name[2:]+'.json'))) 
#         init_text = json_file["caption"]
#         words = init_text.split()
#         words = words[3:]
#         words[0] = words[0].capitalize()
#         text = ' '.join(words)
#         text = text.split('. ')
#         text = '. '.join(text[:2]) + '.'  
#         print(text)
#         # step 2: Restoration
#         w, h = image.size
#         w *= args.upscale
#         h *= args.upscale
#         image = image.resize((w, h), Image.LANCZOS)
#         input_image, width_init, height_init, width_now, height_now = check_image_size(image)
#         prompt_init = text 
#         negative_prompt_init = ""
#         generator = torch.Generator(device='cuda:0').manual_seed(args.seed)
#         gen_image = pipe(lr_img=input_image, prompt = prompt_init, negative_prompt = negative_prompt_init, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator, start_point=args.start_point, height = height_now, width=width_now, overlap=args.latent_tiled_overlap, target_size=(args.latent_tiled_size, args.latent_tiled_size)).images[0]
#         path = os.path.join(args.save_dir, img_name+'.png')
#         cropped_image = gen_image.crop((0, 0, width_init, height_init))

#         if args.color_fix == 'nofix':
#             out_image = cropped_image
#         else:
#             if args.color_fix == 'wavelet':
#                 out_image = wavelet_color_fix(cropped_image, image)
#             elif args.color_fix == 'adain':
#                 out_image = adain_color_fix(cropped_image, image)
#         out_image.save(path)

# 批量文件夹
for save_dir,json_dir in save_dir_dict.items():
    os.makedirs(save_dir, exist_ok=True)
    exist_file = os.listdir(save_dir)

    with torch.no_grad():
        for file_name in sorted(os.listdir(args.img_dir)):
            img_name, ext = os.path.splitext(file_name)
            if ext == ".json":
                continue
            
            if f"{img_name}.png" in exist_file:
                print(f"{img_name}.png exist")
                continue
            else:
                print(img_name)

            image = Image.open(os.path.join(args.img_dir,file_name)).convert('RGB')

            json_file = json.load(open(os.path.join(json_dir,img_name+'.json'))) 
            init_text = json_file["caption"]
            words = init_text.split()
            words = words[3:]
            print(words)
            if(words == []):
                json_file = json.load(open(os.path.join("data/DIV2K/crop_512/caption/lq_caption_omini_universal",img_name+'.json'))) 
                init_text = json_file["caption"]
                words = init_text.split()
                words = words[3:]
            words[0] = words[0].capitalize()
            text = ' '.join(words)
            text = text.split('. ')
            text = '. '.join(text[:2]) + '.'  
            print(text)
            # step 2: Restoration
            w, h = image.size
            w *= args.upscale
            h *= args.upscale
            image = image.resize((w, h), Image.LANCZOS)
            input_image, width_init, height_init, width_now, height_now = check_image_size(image)
            prompt_init = text 
            negative_prompt_init = ""
            generator = torch.Generator(device='cuda:0').manual_seed(args.seed)
            gen_image = pipe(lr_img=input_image, prompt = prompt_init, negative_prompt = negative_prompt_init, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator, start_point=args.start_point, height = height_now, width=width_now, overlap=args.latent_tiled_overlap, target_size=(args.latent_tiled_size, args.latent_tiled_size)).images[0]
            path = os.path.join(save_dir, img_name+'.png')
            cropped_image = gen_image.crop((0, 0, width_init, height_init))

            if args.color_fix == 'nofix':
                out_image = cropped_image
            else:
                if args.color_fix == 'wavelet':
                    out_image = wavelet_color_fix(cropped_image, image)
                elif args.color_fix == 'adain':
                    out_image = adain_color_fix(cropped_image, image)
            out_image.save(path)