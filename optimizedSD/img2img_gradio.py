from cProfile import label
from turtle import update
import gradio as gr
import numpy as np
import torch
import os, re
from PIL import Image
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from einops import rearrange, repeat
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from transformers import logging
logging.set_verbosity_error()
from split_subprompts import split_weighted_subprompts
import mimetypes
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

def update_image_visibility(ddim_steps):
    visible_images = [gr.Image.update(visible=True) for _ in range(0, ddim_steps)]
    hidden_images = [gr.Image.update(visible=False) for _ in range(ddim_steps, 10)]
    return visible_images + hidden_images

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def set_image(image):
    return gr.Image.update(value=image)

def load_img(image, h0, w0):
   
    image = image.convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")   
    if(h0 is not None and w0 is not None):
        h, w = h0, w0
    
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample = Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


config = "optimizedSD/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
sd = load_model_from_config(f"{ckpt}")
li, lo = [], []
for key, v_ in sd.items():
    sp = key.split('.')
    if(sp[0]) == 'model':
        if('input_blocks' in sp):
            li.append(key)
        elif('middle_block' in sp):
            li.append(key)
        elif('time_embed' in sp):
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd['model1.' + key[6:]] = sd.pop(key)
for key in lo:
    sd['model2.' + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")

model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
    
modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
    
modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd


def generate(image, prompt, strength, ddim_steps, batch_size, seed, outdir):
    Height = 512
    Width = 512
    scale = 7.5
    n_iter = 1
    ddim_eta = 1
    seeds = ''
    unet_bs = 1
    outdir = os.path.join('%USERPROFILE%', 'img2img')
    strength = 0.99 if not image else strength
    device = "cuda"
    full_precision = False
    fallback_image = Image.fromarray(255 * np.ones((Height,Width,3), np.uint8))
    init_image = load_img(fallback_image if not image else image, Height, Width).to(device)
    model.unet_bs = unet_bs
    model.turbo = batch_size <= 4
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if device != 'cpu' and full_precision == False:
        model.half()
        modelCS.half()
        modelFS.half()
        init_image = init_image.half()

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    sample_path = os.path.join(outpath, '_'.join(re.split(':| ',prompt)))[:150]
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    
    if seed == '':
        seed = randint(0, 1000000)
    seed = int(seed)
    init_seed = seed
    seed_everything(seed)

    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    assert prompt is not None
    data = [batch_size * [prompt]]

    modelFS.to(device)

    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space

    if(device != 'cpu'):
        mem = torch.cuda.memory_allocated()/1e6
        modelFS.to("cpu")
        while(torch.cuda.memory_allocated()/1e6 >= mem):
            time.sleep(1)


    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength *ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    if full_precision== False and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples = []
    with torch.no_grad():
        for _ in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts,weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c,modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)
                    
                    c = modelCS.get_learned_conditioning(prompts)
                    if(device != 'cpu'):
                        mem = torch.cuda.memory_allocated()/1e6
                        modelCS.to("cpu")
                        while(torch.cuda.memory_allocated()/1e6 >= mem):
                            time.sleep(1)

                    # encode (scaled latent)
                    z_enc = model.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device), seed, ddim_eta, ddim_steps)
                    # decode it
                    samples_ddim = model.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,)

                    modelFS.to(device)
                    print("saving images")
                    for i in range(batch_size):
                        
                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                        x_sample_loc = os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.png")
                        Image.fromarray(x_sample.astype(np.uint8)).save(x_sample_loc)
                        all_samples.append(x_sample_loc)
                        seeds += str(seed) + ','
                        seed +=1
                        base_count += 1

                    if(device != 'cpu'):
                        mem = torch.cuda.memory_allocated()/1e6
                        modelFS.to("cpu")
                        while(torch.cuda.memory_allocated()/1e6 >= mem):
                            time.sleep(1)

                    del samples_ddim
                    del x_sample
                    del x_samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated()/1e6)

    toc = time.time()

    time_taken = (toc-tic)/60.0
    txt = "Took " + str(round(time_taken, 3)) + ". Used seed " + str(init_seed)

    out = list()
    for sample in all_samples:
        out.append(gr.Image.update(value=sample, visible=True))
        out.append(gr.Button.update(visible=True))
    for _ in range(0, 10 - len(all_samples)):
        out.append(gr.Image.update(value=None, visible=False))
        out.append(gr.Button.update(visible=False))
    out.append(txt)
    return tuple(out)

demo = gr.Blocks(title="Image to image")

with demo:
    with gr.Row():
        with gr.Column():
            with gr.Box():
                image = gr.Image(shape=[512, 512], type="pil", tool="editor")
                prompt = gr.Text(label="Prompt")
                strength = gr.Slider(0, 0.99, value=0.9, label="Strength")
                ddim_steps = gr.Slider(1, 50, value=10, step=1, label="Steps")
                batch_size = gr.Slider(1, 10, step=1, label="Batch size")
                seed = gr.Text(label="Seed")
                submit = gr.Button("Generate", variant="primary")
                submit.style(full_width=True)
        with gr.Column():
            with gr.Group():
                oImage1 = gr.Image(type="filepath", show_label=False, visible=False)
                oImage1Btn = gr.Button("Extend", visible=False)
                oImage1Btn.style(full_width=True)
            with gr.Group():
                oImage2 = gr.Image(type="filepath", show_label=False, visible=False)
                oImage2Btn = gr.Button("Extend", visible=False)
                oImage2Btn.style(full_width=True)
            with gr.Group():
                oImage3 = gr.Image(type="filepath", show_label=False, visible=False)
                oImage3Btn = gr.Button("Extend", visible=False)
                oImage3Btn.style(full_width=True)
            with gr.Group():
                oImage4 = gr.Image(type="filepath", show_label=False, visible=False)
                oImage4Btn = gr.Button("Extend", visible=False)
                oImage4Btn.style(full_width=True)
            with gr.Group():
                oImage5 = gr.Image(type="filepath", show_label=False, visible=False)
                oImage5Btn = gr.Button("Extend", visible=False)
                oImage5Btn.style(full_width=True)
            with gr.Group():
                oImage6 = gr.Image(type="filepath", show_label=False, visible=False)
                oImage6Btn = gr.Button("Extend", visible=False)
                oImage6Btn.style(full_width=True)
            with gr.Group():
                oImage7 = gr.Image(type="filepath", show_label=False, visible=False)
                oImage7Btn = gr.Button("Extend", visible=False)
                oImage7Btn.style(full_width=True)
            with gr.Group():
                oImage8 = gr.Image(type="filepath", show_label=False, visible=False)
                oImage8Btn = gr.Button("Extend", visible=False)
                oImage8Btn.style(full_width=True)
            with gr.Group():
                oImage9 = gr.Image(type="filepath", show_label=False, visible=False)
                oImage9Btn = gr.Button("Extend", visible=False)
                oImage9Btn.style(full_width=True)
            with gr.Group():
                oImage10= gr.Image(type="filepath", show_label=False, visible=False)
                oImage10Btn = gr.Button("Extend", visible=False)
                oImage10Btn.style(full_width=True)
            oText = gr.Text()
    oImagePairs = [
        [oImage1, oImage1Btn],
        [oImage2, oImage2Btn],
        [oImage3, oImage3Btn],
        [oImage4, oImage4Btn],
        [oImage5, oImage5Btn],
        [oImage6, oImage6Btn],
        [oImage7, oImage7Btn],
        [oImage8, oImage8Btn],
        [oImage9, oImage9Btn],
        [oImage10, oImage10Btn]
    ]
   
    for img, btn in oImagePairs:
        btn.click(fn=set_image, inputs=[img], outputs=[image])

    submit.click(fn=generate,
        inputs=[
            image, 
            prompt, 
            strength,
            ddim_steps, 
            batch_size, 
            seed],
        outputs=[
            oImage1,
            oImage1Btn,
            oImage2,
            oImage2Btn,
            oImage3,
            oImage3Btn,
            oImage4,
            oImage4Btn,
            oImage5,
            oImage5Btn,
            oImage6,
            oImage6Btn,
            oImage7,
            oImage7Btn,
            oImage8,
            oImage8Btn,
            oImage9,
            oImage9Btn,
            oImage10,
            oImage10Btn,
            oText])
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=8081)