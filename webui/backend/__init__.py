import mimetypes
from typing import Any, List, Tuple, Union
from PIL.Image import Image as ImageType
from webui.backend.split_subprompts import split_weighted_subprompts
import numpy as np
import torch
import os
from PIL import Image
from PIL.Image import Image as ImageType
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast  # type: ignore
from einops import rearrange, repeat
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from transformers import logging
import webui.backend.functions as functions
import webui.backend.constants as constants

logging.set_verbosity_error()
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


config = "models/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
sd = functions.load_model_from_config(f"{ckpt}")
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
model.unet_bs = constants.unet_bs
model.cdevice = constants.device

modelCS: Any = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = constants.device

modelFS: Any = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd


def generate(
    image: ImageType,
    mask: ImageType,
    prompt: str,
    strength: float,
    ddim_steps: int,
    batch_size: int,
    seed: Union[str, int],
    masking_render_full: bool,
    masking_render_padding: int = 15,
    mask_fading: int = 10,
):
    # set up variables and models
    strength = 0.99 if not image else strength
    ddim_steps = int(ddim_steps / strength)
    (
        cropped_mask,
        blurred_mask,
        cropped_mask_width,
        cropped_mask_height,
        mask_offset_top,
        mask_offset_left,
        mask_offset_bottom,
        mask_offset_right,
    ) = functions.load_masks(
        mask, masking_render_full, masking_render_padding, mask_fading
    )
    encode_image = functions.load_encode_image(
        mask,
        image,
        mask_offset_left,
        mask_offset_top,
        mask_offset_right,
        mask_offset_bottom,
    )

    model.turbo = batch_size <= 4
    seed = int(seed) if seed != "" else randint(0, 1000000)
    seed_everything(seed)

    if constants.device != "cpu" and constants.full_precision == False:
        model.half()
        modelCS.half()
        modelFS.half()
        encode_image = encode_image.half()

    tic = time.time()
    os.makedirs(constants.outdir, exist_ok=True)
    sample_path = constants.outdir
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    assert prompt is not None
    data = [batch_size * [prompt]]
    init_latent = None

    if image is not None:
        modelFS.to(constants.device)
        encode_image = repeat(encode_image, "1 ... -> b ...", b=batch_size)
        init_latent = modelFS.get_first_stage_encoding(
            modelFS.encode_first_stage(encode_image)
        )
        functions.release_memory(modelFS)

    assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    if constants.full_precision == False and constants.device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples: List[Tuple[ImageType, int]] = []
    with torch.no_grad():
        for _ in trange(constants.n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(constants.device)
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
                                modelCS.get_learned_conditioning(subprompts[i]),
                                alpha=weight,
                            )
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    functions.release_memory(modelCS)
                    if image is not None:
                        # encode (scaled latent)
                        z_enc = model.stochastic_encode(
                            init_latent,
                            torch.tensor([t_enc] * batch_size).to(constants.device),
                            seed,
                            constants.ddim_eta,
                            ddim_steps,
                        )
                        # decode it
                        samples_ddim = model.decode(
                            z_enc,
                            c,
                            t_enc,
                            unconditional_guidance_scale=constants.scale,
                            unconditional_conditioning=uc,
                        )
                    else:
                        samples_ddim = model.sample(
                            S=ddim_steps,
                            conditioning=c,
                            batch_size=batch_size,
                            seed=seed,
                            shape=[
                                constants.C,
                                constants.Height // constants.f,
                                constants.Width // constants.f,
                            ],
                            verbose=False,
                            unconditional_guidance_scale=constants.scale,
                            unconditional_conditioning=uc,
                            eta=constants.ddim_eta,
                            x_T=None,
                        )

                    modelFS.to(constants.device)
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
                            "seed_" + str(img_seed) + "_" + f"{base_count:05}.png",
                        )
                        sample_image = Image.fromarray(x_sample.astype(np.uint8))
                        final_sample = functions.generate_final_sample(
                            image,
                            mask,
                            blurred_mask,
                            sample_image,
                            cropped_mask,
                            cropped_mask_width,
                            cropped_mask_height,
                            mask_offset_left,
                            mask_offset_top,
                        )
                        final_sample.save(x_sample_loc)
                        all_samples.append((final_sample, img_seed))
                        base_count += 1

                    functions.release_memory(modelFS)
                    del samples_ddim
                    del x_sample
                    del x_samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    time_taken = (toc - tic) / 60.0
    return (all_samples, time_taken)
