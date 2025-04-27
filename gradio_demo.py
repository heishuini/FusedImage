import torch
import torch.utils.checkpoint
import torch.cuda
import random
import gradio as gr
import numpy as np
import argparse
from typing import List
from PIL import Image
from FaithDiff.create_FaithDiff_model import FaithDiff_pipeline
from PIL import Image
from CKPT_PTH import LLAVA_MODEL_PATH, SDXL_PATH, FAITHDIFF_PATH, VAE_FP16_PATH
from utils.color_fix import wavelet_color_fix, adain_color_fix
from utils.image_process import check_image_size, create_hdr_effect
from llava.llm_agent import LLavaAgent
from utils.system import torch_gc

MAX_SEED = np.iinfo(np.int32).max
parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, default='6688')
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--cpu_offload", action='store_true', default=False)
parser.add_argument("--use_fp8", action='store_true', default=False)
args = parser.parse_args()

server_ip = args.ip
server_port = args.port
use_llava = not args.no_llava
cpu_offload = args.cpu_offload
use_fp8 = args.use_fp8

if torch.cuda.device_count() >= 2:
    LLaVA_device = 'cuda:1'
    Diffusion_device = 'cuda:0'
elif torch.cuda.device_count() == 1:
    Diffusion_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

# load LLaVA
if use_llava:
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device, load_8bit=True, load_4bit=False)
else:
    llava_agent = None
    
pipe = FaithDiff_pipeline(sdxl_path=SDXL_PATH, VAE_FP16_path=VAE_FP16_PATH, FaithDiff_path=FAITHDIFF_PATH, use_fp8=use_fp8)
pipe = pipe.to(Diffusion_device)

### enable_vae_tiling
pipe.set_encoder_tile_settings()
pipe.enable_vae_tiling()

if cpu_offload:
    pipe.enable_model_cpu_offload()

@torch.no_grad()
def caption_process(
    image: Image.Image,
    ) -> List[np.ndarray]:

    if use_llava:                
        caption = llava_agent.gen_image_caption([image])        
    else:
        caption = ['Caption Generation is not available. Please add text manually.']
    return caption[0]

def clear_result():
    return gr.update(value=None)

def randomize_seed_fn(generation_seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        generation_seed = random.randint(0, MAX_SEED)
    return generation_seed

@torch.no_grad()
def process(
    image: Image.Image,
    user_prompt: str,
    num_inference_steps: int,
    scale_factor: int,
    guidance_scale: float,
    seed: int,
    latent_tiled_size: int,
    latent_tiled_overlap: int,
    color_fix: str,
    start_point: str, 
    hdr: float
    ) -> List[np.ndarray]:

    w, h = image.size
    w *= scale_factor
    h *= scale_factor
    image = image.resize((w, h), Image.LANCZOS)
    input_image, width_init, height_init, width_now, height_now = check_image_size(image)
    if use_llava:
        init_text = user_prompt
        words = init_text.split()
        words = words[3:]
        words[0] = words[0].capitalize()
        text = ' '.join(words)
        text = text.split('. ')
        text = '. '.join(text[:2]) + '.'
        user_prompt = text 

    negative_prompt_init = ""
    generator = torch.Generator(device=Diffusion_device).manual_seed(seed)    
    input_image = create_hdr_effect(input_image, hdr)
    print(user_prompt)
    gen_image = pipe(lr_img=input_image, prompt = user_prompt, negative_prompt = negative_prompt_init, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator, start_point=start_point, height = height_now, width=width_now,  overlap=latent_tiled_overlap, target_size=(latent_tiled_size, latent_tiled_size)).images[0]
    torch_gc()
    cropped_image = gen_image.crop((0, 0, width_init, height_init))
    if color_fix == 'nofix':
        out_image = cropped_image
    else:
        if color_fix == 'wavelet':
            out_image = wavelet_color_fix(cropped_image, image)
        elif color_fix == 'adain':
            out_image = adain_color_fix(cropped_image, image)

    image = np.array(out_image)    
    return image

#
css = """
body {    
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;    
    margin: 0;
    padding: 0;
}
.gradio-container {    
    border-radius: 15px;
    padding: 30px 40px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    margin: 40px 340px;    
}
.gradio-container h1 {    
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
}
.fillable {
    width: 100% !important;
    max-width: unset !important;
}
.gradio-slider-input {
    input[type="number"] {
        width:  8em;
    }
}
.slider-input-right > .wrap > .head {   
    display: flex;    
}
.slider-input-right > .wrap > .head > .tab-like-container {   
    margin-left: auto;    
}
#examples_container {
    margin: auto;
    width: 90%;
}
#examples_row {
    justify-content: center;
}
#tips_row{    
    padding-left: 20px;
}
.sidebar {    
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
.sidebar .toggle-button {    
    background: linear-gradient(90deg, #34d399, #10b981) !important;
    border: none;    
    padding: 12px 24px;
    text-transform: uppercase;
    font-weight: bold;
    letter-spacing: 1px;
    border-radius: 5px;
    cursor: pointer;
    transition: transform 0.2s ease-in-out;
}
.toggle-button:hover {
    transform: scale(1.05);
}
"""

title = """<h1 align="center">FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution</h1>
           <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; overflow:hidden;">
                <span>💻 <a href="https://github.com/JyChen9811/FaithDiff/">GitHub Code</a> | 📜 <a href="https://arxiv.org/abs/2411.18824"> Paper</a></span>
                <span>If FaithDiff is helpful for you, please help star the GitHub Repo. Thanks!</span>
           </div>
           """
block = gr.Blocks(css=css, theme=gr.themes.Ocean(), title="FaithDiff").queue()
with block:
    gr.Markdown(title)    
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Input Image", sources=["upload"], height=500)
                with gr.Column():
                    result = gr.Image(label="Generated Image", show_label=True, format="png", interactive=False, scale=1, height=500, min_width=670)            
            with gr.Row():
                with gr.Accordion("Input Prompt", open=True):
                    with gr.Column():                
                        user_prompt = gr.Textbox(lines=2, label="User Prompt", value="")
            with gr.Row():
                run_button = gr.Button(value="Restoration Run", variant="primary")
                llave_button = gr.Button(value="Caption Generation Run")       
    with gr.Sidebar(label="Parameters", open=True):
        gr.Markdown("### General parameters")
        with gr.Row():
            cfg_scale = gr.Slider(label="CFG Scale", elem_classes="gradio-slider-input slider-input-right", info="Set a value larger than 1 to enable it!", minimum=0.1, maximum=10.0, value=5, step=0.1)
            num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=50, value=20, step=1)
            generation_seed = gr.Slider(label="Seed", elem_classes="gradio-slider-input", minimum=0, maximum=MAX_SEED, step=1, value=42)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=False)                    
        with gr.Row():
            latent_tiled_size = gr.Slider(label="Tile Size", elem_classes="gradio-slider-input slider-input-right", minimum=1024, maximum=1280, value=1024, step=1)
            latent_tiled_overlap = gr.Slider(label="Tile Overlap", elem_classes="gradio-slider-input slider-input-right", minimum=0.1, maximum=0.9, value=0.5, step=0.1)
            scale_factor = gr.Number(label="SR Scale", value=2)
            color_fix = gr.Dropdown(label="Color Fix", choices=["wavelet", "adain", "nofix"], value="adain")
            hdr = gr.Slider(label="HDR Effect", elem_classes="gradio-slider-input", minimum=0, maximum=2, value=0, step=0.1)
            start_point = gr.Dropdown(label="Start Point", choices=["lr", "noise"], value="lr")
    with gr.Accordion(label="Example Images", open=True):
        with gr.Row(elem_id="examples_row"):
            with gr.Column(scale=12, elem_id="examples_container"):
                gr.Examples(
                    examples=[
                        [   "./examples/band.png",                           
                            "Three men posing for a picture with their guitars.",                          
                            20,
                            2.0,
                            5,
                            42,
                            1024,
                            0.5,
                            "adain",
                            "lr",
                            0
                        ],
                     
                    ],
                    inputs = [
                        input_image,
                        user_prompt,
                        num_inference_steps,
                        scale_factor,
                        cfg_scale,
                        generation_seed,
                        latent_tiled_size,
                        latent_tiled_overlap,
                        color_fix,
                        start_point,
                        hdr
                    ],                    
                    fn=process,
                    outputs=result,
                    cache_examples=False,
                )      
    inputs = [
        input_image,
        user_prompt,
        num_inference_steps,
        scale_factor,
        cfg_scale,
        generation_seed,
        latent_tiled_size,
        latent_tiled_overlap,
        color_fix,
        start_point,
        hdr
    ]
    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=randomize_seed_fn,
        inputs=[generation_seed, randomize_seed],
        outputs=generation_seed,
        queue=False,
        api_name=False,
    ).then(fn=process, inputs=inputs, outputs=[result])
    llave_button.click(fn=caption_process, inputs=[input_image], outputs=[user_prompt])
block.launch(server_name=server_ip, server_port=server_port)
