import mimetypes
from split_subprompts import split_weighted_subprompts
import gradio as gr
import numpy as np
import torch
import os
import re
from PIL import Image
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from einops import rearrange, repeat
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from transformers import logging
logging.set_verbosity_error()
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

def toggle_strength(image):
    return gr.Button.update(visible=image is not None)

def update_image_visibility(ddim_steps):
    visible_images = [gr.Image.update(visible=True)
                      for _ in range(0, ddim_steps)]
    hidden_images = [gr.Image.update(visible=False)
                     for _ in range(ddim_steps, 10)]
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

    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 64, (w, h))

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
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


def generate(image, prompt, strength, ddim_steps, batch_size, seed, image_history):
    C = 4
    f = 8
    Height = 512
    Width = 512
    n_iter = 1
    ddim_eta = 0
    seeds = ''
    scale = 7.5
    unet_bs = 1
    outdir = os.path.join('output')
    strength = 0.99 if not image else strength
    ddim_steps = int(ddim_steps / strength)
    device = "cuda"
    full_precision = False
    fallback_image = Image.fromarray(
        255 * np.ones((Height, Width, 3), np.uint8))
    init_image = load_img(
        fallback_image if not image else image, Height, Width).to(device)
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
    sample_path = os.path.join(
        outpath, '_'.join(re.split(':| ', prompt)))[:150]
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

    if image is not None:
        modelFS.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = modelFS.get_first_stage_encoding(
            modelFS.encode_first_stage(init_image))  # move to latent space

        if(device != 'cpu'):
            mem = torch.cuda.memory_allocated()/1e6
            modelFS.to("cpu")
            while(torch.cuda.memory_allocated()/1e6 >= mem):
                time.sleep(1)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    if full_precision == False and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples = []
    with torch.no_grad():
        for _ in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = modelCS.get_learned_conditioning(
                        batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(
                                subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    if image is not None:
                        # encode (scaled latent)
                        z_enc = model.stochastic_encode(init_latent, torch.tensor(
                            [t_enc]*batch_size).to(device), seed, ddim_eta, ddim_steps)
                        # decode it
                        samples_ddim = model.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,)
                    else:
                        samples_ddim = model.sample(
                            S=ddim_steps,
                            conditioning=c,
                            batch_size=batch_size,
                            seed=seed,
                            shape=[C, Height // f, Width // f],
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=ddim_eta,
                            x_T=None,)

                    modelFS.to(device)
                    print("saving images")
                    for i in range(batch_size):
                        x_samples_ddim = modelFS.decode_first_stage(
                            samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255. * \
                            rearrange(
                                x_sample[0].cpu().numpy(), 'c h w -> h w c')
                        x_sample_loc = os.path.join(
                            sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.png")
                        Image.fromarray(x_sample.astype(
                            np.uint8)).save(x_sample_loc)
                        all_samples.append(x_sample_loc)
                        seeds += str(seed) + ','
                        seed += 1
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

    if len(image_history) >= 5:
        image_history.pop(0)
    if image is not None and (len(image_history) == 0 or not np.array_equal(image.getdata(), image_history[-1].getdata())):
        image_history.append(image)
    toc = time.time()

    time_taken = (toc-tic)/60.0
    out = list()
    out.append(gr.Button.update(visible=True))
    out.append(gr.Image.update(value=all_samples[0]))
    for sample in all_samples[1:]:
        out.append(gr.Row.update(visible=True))
        out.append(gr.Image.update(value=sample, visible=True))
    for _ in range(0, 9 - len(all_samples[1:])):
        out.append(gr.Row.update(visible=False))
        out.append(gr.Image.update(value=None, visible=False, ))
    out.append("Took " + str(round(time_taken, 3)) +
               ". Used seed " + str(init_seed))
    for hImage in image_history:
        out.append(gr.Image.update(value=hImage))
        out.append(gr.Button.update(visible=True))
    for _ in range(0, 5 - len(image_history)):
        out.append(gr.Image.update(value=None))
        out.append(gr.Button.update(visible=False))
    out.append(image_history)
    return tuple(out)


with gr.Blocks(title="Image to image", css="""
    .svelte-10ogue4 {
        flex: 1;
    } 
""") as blocks:
    with gr.Row() as main_row:
        main_row.style(equal_height=False)
        with gr.Column(variant="panel"):
            with gr.Group():
                image = gr.Image(shape=[512, 512], type="pil", tool="editor")
                prompt = gr.Text(label="Prompt")
                strength = gr.Slider(0.1, 0.9, value=0.5, step=0.1, label="Strength", visible=False)
                ddim_steps = gr.Slider(1, 200, value=10, step=1, label="Steps")
                batch_size = gr.Slider(1, 10, step=1, label="Batch size")
                seed = gr.Text(label="Seed")
                submit = gr.Button("Generate", variant="primary")
                submit.style(full_width=True)
        with gr.Column(variant="panel"):
            with gr.Row(visible=True) as oImage1Row:
                with gr.Group():
                    oImage1 = gr.Image(
                        type="filepath", show_label=False, interactive=False)
                    oImage1Btn = gr.Button("Extend", visible=False)
                    oImage1Btn.style(full_width=True)
            with gr.Row(visible=False) as oImage2Row:
                with gr.Group():
                    oImage2 = gr.Image(
                        type="filepath", show_label=False, interactive=False)
                    oImage2Btn = gr.Button("Extend")
                    oImage2Btn.style(full_width=True)
            with gr.Row(visible=False) as oImage3Row:
                with gr.Group():
                    oImage3 = gr.Image(
                        type="filepath", show_label=False, interactive=False)
                    oImage3Btn = gr.Button("Extend")
                    oImage3Btn.style(full_width=True)
            with gr.Row(visible=False) as oImage4Row:
                with gr.Group():
                    oImage4 = gr.Image(
                        type="filepath", show_label=False, interactive=False)
                    oImage4Btn = gr.Button("Extend")
                    oImage4Btn.style(full_width=True)
            with gr.Row(visible=False) as oImage5Row:
                with gr.Group():
                    oImage5 = gr.Image(
                        type="filepath", show_label=False, interactive=False)
                    oImage5Btn = gr.Button("Extend")
                    oImage5Btn.style(full_width=True)
            with gr.Row(visible=False) as oImage6Row:
                with gr.Group():
                    oImage6 = gr.Image(
                        type="filepath", show_label=False, interactive=False)
                    oImage6Btn = gr.Button("Extend")
                    oImage6Btn.style(full_width=True)
            with gr.Row(visible=False) as oImage7Row:
                with gr.Group():
                    oImage7 = gr.Image(
                        type="filepath", show_label=False, interactive=False)
                    oImage7Btn = gr.Button("Extend")
                    oImage7Btn.style(full_width=True)
            with gr.Row(visible=False) as oImage8Row:
                with gr.Group():
                    oImage8 = gr.Image(
                        type="filepath", show_label=False, interactive=False)
                    oImage8Btn = gr.Button("Extend")
                    oImage8Btn.style(full_width=True)
            with gr.Row(visible=False) as oImage9Row:
                with gr.Group():
                    oImage9 = gr.Image(
                        type="filepath", show_label=False, interactive=False)
                    oImage9Btn = gr.Button("Extend")
                    oImage9Btn.style(full_width=True)
            with gr.Row(visible=False) as oImage10Row:
                with gr.Group():
                    oImage10 = gr.Image(
                        type="filepath", show_label=False, interactive=False)
                    oImage10Btn = gr.Button("Extend")
                    oImage10Btn.style(full_width=True)
            oText = gr.Text(label="Time and seed", visible=True)
    gr.Markdown("### History (last 5)")
    with gr.Row():
        with gr.Column():
            with gr.Group():
                hImage1 = gr.Image(show_label=False, interactive=False)
                hImage1Btn = gr.Button("Rollback", visible=False)
                hImage1Btn.style(full_width=True)
        with gr.Column():
            with gr.Group():
                hImage2 = gr.Image(show_label=False, interactive=False)
                hImage2Btn = gr.Button("Rollback", visible=False)
                hImage2Btn.style(full_width=True)
        with gr.Column():
            with gr.Group():
                hImage3 = gr.Image(show_label=False, interactive=False)
                hImage3Btn = gr.Button("Rollback", visible=False)
                hImage3Btn.style(full_width=True)
        with gr.Column():
            with gr.Group():
                hImage4 = gr.Image(show_label=False, interactive=False)
                hImage4Btn = gr.Button("Rollback", visible=False)
                hImage4Btn.style(full_width=True)
        with gr.Column():
            with gr.Group():
                hImage5 = gr.Image(show_label=False, interactive=False)
                hImage5Btn = gr.Button("Rollback", visible=False)
                hImage5Btn.style(full_width=True)

    image_history = gr.Variable(list())
    image.change(fn=toggle_strength, inputs=[image], outputs=[strength], show_progress=False)

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

    hImagePairs = [
        [hImage1, hImage1Btn],
        [hImage2, hImage2Btn],
        [hImage3, hImage3Btn],
        [hImage4, hImage4Btn],
        [hImage5, hImage5Btn]
    ]
    for img, btn in hImagePairs:
        btn.click(fn=set_image, inputs=[img], outputs=[image])

    submit.click(fn=generate,
                 inputs=[
                     image,
                     prompt,
                     strength,
                     ddim_steps,
                     batch_size,
                     seed,
                     image_history],
                 outputs=[
                     oImage1Btn,
                     oImage1,
                     oImage2Row,
                     oImage2,
                     oImage3Row,
                     oImage3,
                     oImage4Row,
                     oImage4,
                     oImage5Row,
                     oImage5,
                     oImage6Row,
                     oImage6,
                     oImage7Row,
                     oImage7,
                     oImage8Row,
                     oImage8,
                     oImage9Row,
                     oImage9,
                     oImage10Row,
                     oImage10,
                     oText,
                     hImage1,
                     hImage1Btn,
                     hImage2,
                     hImage2Btn,
                     hImage3,
                     hImage3Btn,
                     hImage4,
                     hImage4Btn,
                     hImage5,
                     hImage5Btn,
                     image_history])

blocks.queue()
blocks.launch(share=True)
