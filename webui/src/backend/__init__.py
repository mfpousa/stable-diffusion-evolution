import mimetypes
from typing import Any, List, Tuple, Union

from PIL.Image import Image as ImageType
from src.backend.split_subprompts import split_weighted_subprompts
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
from torch import autocast  # type: ignore
from einops import rearrange, repeat
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from transformers import logging

logging.set_verbosity_error()
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

# constants
C = 4
f = 8
Height = 512
Width = 512
n_iter = 1
ddim_eta = 0
scale = 7.5
unet_bs = 1
device = "cuda"
full_precision = False
outdir = os.path.join("output")
fallback_image = Image.fromarray(
    255 * np.ones((Height, Width, 3), np.uint8))


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


def load_img(image, h0, w0):
    image = image.convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 64, (w, h))

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


config = "models/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
sd = load_model_from_config(f"{ckpt}")
li, lo = [], []
for key, v_ in sd.items():
    sp = key.split(".")
    if (sp[0]) == "model":
        if "input_blocks" in sp:
            li.append(key)
        elif "middle_block" in sp:
            li.append(key)
        elif "time_embed" in sp:
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd["model1." + key[6:]] = sd.pop(key)
for key in lo:
    sd["model2." + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")

model: Any = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
model.unet_bs = unet_bs
model.cdevice = device

modelCS: Any = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = device

modelFS: Any = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd


def generate(
    image,
    prompt: str,
    strength: float,
    ddim_steps: int,
    batch_size: int,
    seed: Union[str, int],
    image_history: List[ImageType],
):
    # set up variables and models
    strength = 0.99 if not image else strength
    ddim_steps = int(ddim_steps / strength)
    init_image = load_img(fallback_image if not image else image, Height, Width).to(
        device
    )
    model.turbo = batch_size <= 4
    seed = int(seed) if seed != "" else randint(0, 1000000)
    seed_everything(seed)

    if device != "cpu" and full_precision == False:
        model.half()
        modelCS.half()
        modelFS.half()
        init_image = init_image.half()

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    sample_path = os.path.join(
        outdir, "_".join(re.split(":| \n", prompt)))[:150]
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    assert prompt is not None
    data = [batch_size * [prompt]]
    init_latent = None

    if image is not None:
        modelFS.to(device)
        init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
        init_latent = modelFS.get_first_stage_encoding(
            modelFS.encode_first_stage(init_image)
        )

        if device != "cpu":
            mem = torch.cuda.memory_allocated() / 1e6
            modelFS.to("cpu")
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(1)

    assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    if full_precision == False and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples: List[Tuple[str, int]] = []
    with torch.no_grad():
        for _ in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = modelCS.get_learned_conditioning(batch_size * [""])
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
                            c = torch.add(
                                c,
                                modelCS.get_learned_conditioning(
                                    subprompts[i]),
                                alpha=weight,
                            )
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    if image is not None:
                        # encode (scaled latent)
                        z_enc = model.stochastic_encode(
                            init_latent,
                            torch.tensor([t_enc] * batch_size).to(device),
                            seed,
                            ddim_eta,
                            ddim_steps,
                        )
                        # decode it
                        samples_ddim = model.decode(
                            z_enc,
                            c,
                            t_enc,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                        )
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
                            x_T=None,
                        )

                    modelFS.to(device)
                    print("saving images")
                    x_sample = x_samples_ddim = None
                    for i in range(batch_size):
                        img_seed = seed + i
                        x_samples_ddim = modelFS.decode_first_stage(
                            samples_ddim[i].unsqueeze(0)
                        )
                        x_sample = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        x_sample = 255.0 * rearrange(
                            x_sample[0].cpu().numpy(), "c h w -> h w c"
                        )
                        x_sample_loc = os.path.join(
                            sample_path,
                            "seed_" + str(img_seed) + "_" +
                            f"{base_count:05}.png",
                        )
                        Image.fromarray(x_sample.astype(
                            np.uint8)).save(x_sample_loc)
                        all_samples.append((x_sample_loc, img_seed))
                        base_count += 1

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim
                    del x_sample
                    del x_samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    if image is not None and (
        len(image_history) == 0
        or not np.array_equal(image.getdata(), image_history[-1].getdata())
    ):
        image_history.append(image)
    toc = time.time()

    time_taken = (toc - tic) / 60.0
    return (all_samples, time_taken)
