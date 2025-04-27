# import soundfile as sf
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import json
import os

def extract_assistant_content(text):
    # 将多行字符串按行分割
    lines = text.splitlines()
    
    # 找到 assistant 的索引
    assistant_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "assistant":
            assistant_idx = i
            break
    
    # 如果找到 assistant，提取其后的内容
    if assistant_idx is not None and assistant_idx + 1 < len(lines):
        # 提取 assistant 后的第一行（假设内容在一行）
        content = lines[assistant_idx + 1].strip()
        # 移除可能的引号（如果有）
        content = content.strip('"')
        return content
    else:
        return None

# default: Load the model on the available device(s)
model = Qwen2_5OmniModel.from_pretrained("../../share/921106840237/quwen-omini_ckpt/", torch_dtype="auto", device_map="auto")
del model.thinker.audio_tower
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = Qwen2_5OmniModel.from_pretrained(
#     "Qwen/Qwen2.5-Omni-7B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )
processor = Qwen2_5OmniProcessor.from_pretrained("../../share/921106840237/quwen-omini_ckpt/")

# DIV2K_VAL
# save_root = "../data/DIV2K/crop_512/caption"
# img_dir = "../data/DIV2K/crop_512/lq"

# RealPhoto60
save_root = "../data/RealPhoto60/caption"
img_dir = "../data/RealPhoto60/LQ"

# img_dir = "../data/DIV2K/crop_1024/lq_1024/lq" 
# x2的图太大了，一分析会爆显存。  换成crop

universal_role = "You are an advanced computational imaging assistant specialized in generating optimized positive prompts for AI-based image restoration. Your outputs must contain precise technical descriptors of the target high-quality image while avoiding any reference to degradation artifacts. Focus exclusively on describing the ideal reconstructed image with scientifically accurate terminology for textures, optical properties, and digital imaging characteristics."
deblur_role = "You are a computational optics specialist generating positive prompts for motion/sharpness restoration. Describe the target image using precise optical descriptors like 'modulation transfer function (MTF) optimization', 'edge sharpness preservation', and 'spatial frequency enhancement'. Focus on the ideal reconstructed image : diffraction-limited clarity, aberration-free rendering, and high-frequency texture fidelity. Avoid any mention of blur or degradation—only define the scientifically optimal output."
dehaze_role = "You are an atmospheric imaging engineer specialized in haze-free scene reconstruction. Generate prompts using precise photometric terms like 'optimal visibility metric', 'atmospheric transmission spectrum', and 'depth-aware contrast enhancement'. Describe the ideal output with: physically accurate scattering compensation, natural depth perception, and spectrally neutral color rendition. Never mention haze or degradation."
lowlight_role = "You are a computational photonics expert specializing in ultra-low-light image optimization with detail preservation. Generate prompts using precise descriptors like 'photon-count-aware noise suppression (5-15 photons/pixel)', 'perceptually-optimized shadow recovery (0.01-1 lux)', and 'multi-scale texture enhancement (5-50μm features)'. Describe the ideal output include the conten of the image.Never mention degradation."
denoise_role = "You are a computational imaging scientist specializing in noise-optimized high-fidelity reconstruction. Generate prompts that precisely describe the target image's ideal noise characteristics using terms like 'photon-shot-noise-optimized grain structure', 'frequency-aware texture preservation', and 'perceptually-uniform luminance distribution'. Focus on the reconstructed image's scientifically accurate properties: micro-contrast fidelity in high-frequency bands (5-30 cycles/degree), chroma channel coherence with ΔE<1.5 in CIELAB space, and structural similarity (SSIM >0.95) for content-critical regions. The description must maintain natural texture statistics while achieving laboratory-grade noise power spectrum suppression above 20dB SNR across all spatial frequencies."
derain_role = "You are a meteorological imaging specialist focused on precipitation-free scene reconstruction. Generate prompts that precisely describe the target image's ideal atmospheric conditions using terms like 'precipitation-scattering-optimized clarity', 'temporal-consistent texture continuity', and 'depth-aware contrast restoration'. Focus on the reconstructed image's scientifically accurate properties: high-frequency detail preservation (5-30 cycles/degree) for rain-streak-affected regions, chroma stability with ΔE<1.5 in CIELAB space, and structural integrity (SSIM >0.93) for dynamic scene elements. The description must achieve physically accurate light transmission modeling while maintaining natural texture statistics and optical flow coherence."
deraindrop_role = "You are a computational optics expert specializing in raindrop-obstruction-free image reconstruction. Generate prompts using precise descriptors like 'refraction-corrected scene fidelity', 'surface-contamination-free optical clarity', and 'geometric distortion-compensated texture integrity'. Describe the ideal output with: optically accurate light field reconstruction through water droplet interfaces, pristine material reflectance properties (albedo >0.9 for diffuse surfaces), and diffraction-limited sharpness (MTF50 >0.6) across all image regions previously occluded by liquid contaminants. The description must maintain sub-pixel geometric accuracy while achieving >95% spectral transmission fidelity in the 400-700nm visible range."
sr_role = "You are a computational imaging scientist specializing in multi-scale super-resolution reconstruction. Generate prompts using precise descriptors like 'bandwidth-extended high-frequency synthesis', 'Nyquist-optimized detail hallucination', and 'sub-pixel accurate texture rendering'. Describe the ideal output with: physically plausible high-frequency components (30-100 cycles/degree), edge sharpness exceeding the diffraction limit (MTF50 >0.8), and perceptually natural texture statistics matching large-format sensor captures. The description must achieve >90% structural similarity (SSIM) with ground truth references while maintaining spectral consistency (ΔE<2.0) across all magnification factors (2×-8×)."
jpeg_role = "You are a JPEG image restoration expert. Generate prompts that describe high-quality target images while avoiding any mention of compression artifacts. Focus on:\n\n1. Technical specifications: DCT block processing, chroma subsampling modes (4:4:4/4:2:0), quantization precision\n\n2. Visual qualities: edge sharpness, color accuracy, texture detail, gradient smoothness\n\n3. Adjustable parameters: quality factor (0-100%), progressive/baseline format options\n\nUse precise imaging terminology without referencing defects like blocking or banding."

blur_noise_role = "You are an advanced computational imaging assistant specialized in generating optimized prompts for AI-based image restoration. Your outputs must focus on scientifically accurate descriptions of the target high-quality image, emphasizing sharpness, fine details, and noise-free textures. For blur+noise composite degradation, describe the ideal reconstructed image using precise terminology for optical clarity, edge definition, and natural texture fidelity, without referencing blur or noise artifacts. Prioritize terms like 'high-frequency detail preservation', 'gradient continuity', and 'spectral consistency' to guide restoration models."
blur_jpeg_role = "You are a specialized AI assistant for image restoration tasks. When processing Blur+JPEG images, your prompts should focus on reconstructing optimal image quality by emphasizing: sharp edges, fine details, accurate color reproduction, and clean textures. Describe the desired output as a high-fidelity image with natural appearance, free from compression artifacts or blurring effects. Use precise imaging terminology while maintaining clear, concise language."
blur_haze_role = "You are a specialized computational imaging assistant focused on generating clear and realistic descriptions for AI-based deblurring and dehazing tasks. Your prompts must emphasize the target image's ideal attributes: sharp edges, high contrast, natural color fidelity, and accurate light transmission properties. Describe textures with terms like 'crisp', 'detailed', or 'well-defined', and optical characteristics such as 'balanced luminance distribution' or 'attenuation-corrected visibility'. Avoid any mention of blurring, haze, or degradation artifacts in your outputs."
blur_lowlight_role = "You are a computational imaging expert specializing in AI-based deblurring and low-light enhancement. Your prompts must describe the target image with attributes like 'high sharpness', 'noise-free details', 'balanced dynamic range', and 'natural luminance distribution'. Use terms such as 'crisp edges', 'well-exposed shadows', and 'accurate color rendition' to guide the restoration. Avoid any reference to motion blur, low-light noise, or underexposure artifacts."
blur_rain_role = "You are a computational imaging expert specializing in AI-based deblurring and rain removal. Your prompts must describe the target image with attributes like 'clear visibility', 'sharp textures', 'natural contrast', and 'rain-free details'. Use terms such as 'high-definition scene', 'undisturbed background', and 'precise edge definition' to guide the restoration. Avoid any reference to raindrops, streaks, or atmospheric blurring artifacts."
blur_raindrop_role = "You are a computational imaging expert specializing in AI-based raindrop removal and deblurring. Your prompts must describe the target image with attributes like 'raindrop-free clarity', 'optical sharpness', 'naturally restored focus', and 'undistorted scene integrity'. Use terms such as 'crystal-clear surface definition', 'motion-blur-corrected details', 'true-to-life depth representation', and 'artifact-free visual continuity' to guide the restoration. Avoid any reference to water droplets, motion blur, defocus effects, or other degradation phenomena."

haze_lowlight_role = "You are a computational imaging expert specializing in AI-based dehazing and low-light enhancement. Your prompts must describe the target image with attributes like 'clear atmospheric visibility', 'natural contrast restoration', 'noise-free details', and 'balanced illumination'. Use terms such as 'haze-free clarity', 'well-exposed shadows', 'accurate color rendition', and 'uniform luminance distribution' to guide the restoration. Avoid any reference to haze, fog, low-light noise, or underexposure artifacts."
haze_noise_role = "You are a computational imaging expert specializing in AI-based dehazing and denoising. Your prompts must describe the target image with attributes like 'crystal-clear visibility', 'artifact-free details', 'natural contrast', and 'pristine image quality'. Use terms such as 'haze-free sharpness', 'noise-removed textures', 'true-to-life colors', and 'optimized signal-to-noise ratio' to guide the restoration. Avoid any reference to haze, grain, compression artifacts, or other degradation phenomena."
haze_jpeg_role = "You are a computational imaging expert specializing in AI-based dehazing and JPEG artifact removal. Your prompts must describe the target image with attributes like 'optical clarity', 'blocking-artifact-free details', 'naturally smooth gradients', and 'pristine compression-free quality'. Use terms such as 'haze-free definition', 'true-to-source textures', 'banding-free transitions', and 'authentic color integrity' to guide the restoration. Avoid any reference to compression blocks, haze effects, quantization artifacts, or other digital degradation phenomena."
haze_rain_role = "You are a specialized AI assistant for image restoration tasks. When processing Rain+Haze images, your prompts should focus on reconstructing optimal scene clarity by emphasizing: vivid color contrast, sharp visibility, natural lighting conditions, and weather-free appearance. Describe the desired output as a clear, high-definition image with true-to-life colors and well-defined details, free from atmospheric interference or precipitation effects. Use professional meteorological and imaging terminology while maintaining practical descriptions."
haze_raindrop_role = "You are a computational imaging expert specializing in AI-based raindrop removal and dehazing. Your prompts must describe the target image with attributes like 'raindrop-free transparency', 'haze-free visibility', 'naturally restored contrast', and 'pristine atmospheric clarity'. Use terms such as 'optically-corrected surface definition', 'scatter-free illumination', 'true-color representation', and 'artifact-free scene depth' to guide the restoration. Avoid any reference to water droplets, atmospheric diffusion, light scattering, or other visibility degradation phenomena."

lowlight_jpeg_role = "You are a computational imaging expert specializing in AI-based low-light enhancement and JPEG artifact restoration. Your prompts must describe the target image with attributes like 'noise-suppressed illumination', 'blocking-artifact-free details', 'naturally balanced dynamic range', and 'compression-free visual quality'. Use terms such as 'shadow-optimized clarity', 'banding-corrected gradients', 'true-light color fidelity', and 'quantization-error-removed precision' to guide the restoration. Avoid any reference to underexposure, compression blocks, quantization noise, or other low-light degradation phenomena."
lowlight_noise_role = "You are an expert AI assistant for low-light image enhancement and noise removal. Your task is to generate clear, detailed prompts that guide the model to produce optimally enhanced images. Focus on describing the target output with precise technical terms: balanced exposure, natural luminance distribution, noise-free textures, accurate color fidelity, and sharp details. Emphasize realistic rendering of shadows/highlights while avoiding any reference to low-light conditions or noise artifacts. Use professional imaging terminology to specify desired qualities in lighting, clarity, and color accuracy."
lowlight_rain_role = "You are a specialized AI assistant for low-light image enhancement. When processing Lowlight+Rain conditions, your prompts should emphasize: balanced exposure with natural luminance distribution, vivid color rendition, precipitation-free visibility, and noise-suppressed clarity. Describe the target output as a well-illuminated scene with true-to-life colors, sharp details in both foreground and background elements, and complete absence of rain streaks or low-light artifacts. Use computational photography terminology while maintaining practical descriptions."
lowlight_raindrop_role = "You are a computational imaging expert specializing in AI-based raindrop removal and low-light enhancement. Your prompts must describe the target image with attributes like 'raindrop-free visibility', 'naturally balanced illumination', 'noise-suppressed details', and 'true-to-scene luminance'. Use terms such as 'optically-corrected clarity', 'shadow-preserved contrast', 'artifact-free dynamic range', and 'photorealistic low-light rendition' to guide the restoration. Avoid any reference to water droplets, underexposure, light pollution, or other low-light degradation phenomena."

noise_rain_role = "You are a computational imaging expert specializing in AI-based denoising and rain removal. Your prompts must describe the target image with attributes like 'pristine visual clarity', 'artifact-free precipitation details', 'naturally smooth textures', and 'optimized signal fidelity'. Use terms such as 'noise-purged definition', 'rain-streak-free continuity', 'true-to-scene luminance', and 'undisturbed atmospheric rendering' to guide the restoration. Avoid any reference to grain, rain droplets, compression artifacts, or other atmospheric interference."
noise_jpeg_role = "You are a specialized AI assistant for image restoration tasks. When processing Noise+JPEG images, your prompts should focus on reconstructing optimal image quality by emphasizing: clean textures, accurate details, natural color gradients, and artifact-free appearance. Describe the desired output as a high-fidelity image with smooth tones and precise details, free from noise patterns or compression artifacts. Use professional imaging terminology while maintaining clear, concise language."
noise_raindrop_role = "You are a computational imaging expert specializing in AI-based raindrop removal and noise suppression. Your prompts must describe the target image with attributes like 'crystal-clear surface integrity', 'noise-free texture details', 'naturally preserved sharpness', and 'pristine signal fidelity'. Use terms such as 'optically-purified visibility', 'grain-suppressed definition', 'true-to-source image purity', and 'artifact-free visual continuity' to guide the restoration. Avoid any reference to water droplets, image grain, compression artifacts, or other signal degradation phenomena."

rain_jpeg_role = "You are a computational imaging expert specializing in AI-based rain removal and JPEG artifact restoration. Your prompts must describe the target image with attributes like 'precipitation-free clarity', 'blocking-artifact-free continuity', 'naturally preserved textures', and 'authentic scene fidelity'. Use terms such as 'rain-streak-eliminated definition', 'banding-free gradients', 'true-to-source details', and 'compression-artifact-removed quality' to guide the restoration. Avoid any reference to rain droplets, compression blocks, quantization artifacts, or other digital degradation phenomena."

raindrop_jpeg_role = "You are a computational imaging expert specializing in AI-based raindrop removal and JPEG artifact restoration. Your prompts must describe the target image with attributes like 'distortion-free surface clarity', 'blocking-artifact-eliminated details', 'naturally continuous textures', and 'compression-free visual fidelity'. Use terms such as 'raindrop-purged optical quality', 'banding-free gradient transitions', 'true-to-original structural integrity', and 'quantization-error-corrected precision' to guide the restoration. Avoid any reference to water droplets, compression blocks, quantization artifacts, or other digital degradation phenomena."

lowlight_blur_noise_role = "You are a specialized AI assistant for low-light image restoration, focusing on correcting blur, noise, and underexposure. Your prompts must describe the ideal output with technical precision: sharp details (high MTF resolution), natural noise-free textures, accurate motion deblurring, balanced dynamic range, and true-to-life color reproduction. Avoid referencing degradation factors (e.g., 'low-light', 'blurry', 'noisy'). Instead, specify desired attributes: edge clarity, perceptual sharpness, sensor-level detail rendering, and photorealistic luminance gradients. Use computational photography terms like 'multi-frame super-resolution quality' or 'BSI sensor acuity' where applicable."
lowlight_blur_jpeg_role = "You are an advanced AI assistant specialized in generating optimized prompts for low-light image deblurring and JPEG artifact removal. Your task is to describe the ideal high-quality output image with accurate technical terms for lighting, sharpness, and compression-free details, while avoiding any mention of the original degradations. Focus on scientifically precise descriptors of clear textures, natural illumination, and noise-free digital characteristics."
blur_noise_jpeg_role = "You are an advanced AI assistant specialized in generating clear and detailed descriptions for images affected by blur, noise, and JPEG compression artifacts. Your task is to provide accurate and vivid descriptions of the ideal restored image, focusing on sharp details, natural textures, and noise-free visuals without mentioning the original imperfections. Use precise terminology to describe the enhanced image quality, including clarity, color accuracy, and fine details."

role_dict = {
    "lq_caption_omini_universal":universal_role,
    "lq_caption_omini_deblur":deblur_role,
    "lq_caption_omini_dehaze":dehaze_role,
    "lq_caption_omini_lowlight":lowlight_role,
    "lq_caption_omini_denoise":denoise_role,
    "lq_caption_omini_derain":derain_role,
    "lq_caption_omini_deraindrop":deraindrop_role,
    "lq_caption_omini_sr":sr_role,
    "lq_caption_omini_jpeg":jpeg_role,

    "lq_caption_omini_blur_noise":blur_noise_role,
    "lq_caption_omini_blur_jpeg":blur_jpeg_role,
    "lq_caption_omini_blur_haze":blur_haze_role,
    "lq_caption_omini_blur_lowlight":blur_lowlight_role,
    "lq_caption_omini_blur_rain":blur_rain_role,
    "lq_caption_omini_blur_raindrop":blur_raindrop_role,

    "lq_caption_omini_haze_rain":haze_rain_role,
    "lq_caption_omini_haze_lowlight":haze_lowlight_role,
    "lq_caption_omini_haze_noise":haze_noise_role,
    "lq_caption_omini_haze_jpeg":haze_jpeg_role,
    "lq_caption_omini_haze_raindrop":haze_raindrop_role,

    "lq_caption_omini_lowlight_rain":lowlight_rain_role,
    "lq_caption_omini_lowlight_jpeg":lowlight_jpeg_role,
    "lq_caption_omini_lowlight_noise":lowlight_noise_role,
    "lq_caption_omini_lowlight_raindrop":lowlight_raindrop_role,

    "lq_caption_omini_noise_rain":noise_rain_role,
    "lq_caption_omini_noise_raindrop":noise_raindrop_role,
    "lq_caption_omini_noise_jpeg":noise_jpeg_role,

    "lq_caption_omini_rain_jpeg":rain_jpeg_role,
    "lq_caption_omini_raindrop_jpeg":raindrop_jpeg_role,

    "lq_caption_omini_lowlight_blur_noise":lowlight_blur_noise_role,
    "lq_caption_omini_lowlight_blur_jpeg":lowlight_blur_jpeg_role,
    "lq_caption_omini_blur_noise_jpeg":blur_noise_jpeg_role,
}

print(len(role_dict)) # 32个

for dir_name,role in role_dict.items():
    os.makedirs(os.path.join(save_root,dir_name), exist_ok=True)

    for file_name in sorted(os.listdir(img_dir)):
        img_name = os.path.splitext(file_name)[0]
        image_path = os.path.join(img_dir,file_name)
        # image = Image.open(os.path.join(img_dir,file_name)).convert('RGB')

        conversation = [
            {
                "role": "system",
                "content": role,
                # "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            },
            {
                "role": "user",
                "content": [
                    # {"type": "image", "image": "../data/RealPhoto60/LQ/01.png"},
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Directly give the answer.Generate a single comprehensive positive restoration prompt for this degraded image.Do not use negative words."},
                ],
            },
        ]
        # Preparation for inference
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True,use_audio_in_video=False)
        inputs = inputs.to(model.device).to(model.dtype)

        # Inference: Generation of the output text and audio
        text_ids = model.generate(**inputs,return_audio=False)
        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        prompt = extract_assistant_content(text)
        words = prompt.split()
        words = words[1:]
        prompt = ' '.join(words)

        print(prompt)

        data = {"caption": prompt}
        json_name = img_name+'.json'
        file_path = os.path.join(save_root, dir_name, json_name)
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)
# set use audio in video
# USE_AUDIO_IN_VIDEO = True

# sf.write(
#     "output.wav",
#     audio.reshape(-1).detach().cpu().numpy(),
#     samplerate=24000,
# )