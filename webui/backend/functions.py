from PIL.Image import Image as ImageType
import numpy as np
import torch
from PIL import Image, ImageFilter
from PIL.Image import Image as ImageType
import torch
import numpy as np
from PIL import Image
from itertools import islice
import time
from torch import Tensor
import webui.backend.constants as constants


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


def load_img(image: ImageType, h0: int, w0: int) -> Tensor:
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


def blur_mask(tensor: Tensor, radius: int):
    image = constants.ToPIL(tensor)
    return constants.ToTensor(image.filter(ImageFilter.GaussianBlur(radius))).to(
        constants.device
    )


def expand_mask(mask: Tensor, radius: int):
    blurred_mask = blur_mask(mask, radius)
    expanded_mask = (blurred_mask * 1000).clamp(0, 1)
    return expanded_mask.to(constants.device)


def crop(image: ImageType, target_width: int, target_height: int):
    width, height = image.size
    if width < target_width:
        growth_factor = target_width / width
        return crop(
            image.resize((target_width, int(height * growth_factor))),
            target_width,
            target_height,
        )
    if height < target_height:
        growth_factor = target_height / height
        return crop(
            image.resize((int(width * growth_factor), target_height)),
            target_width,
            target_height,
        )
    if width > target_width and height != target_height:
        shrink_factor = target_width / width
        return crop(
            image.resize((target_width, int(height * shrink_factor))),
            target_width,
            target_height,
        )
    if height > target_height and width != target_width:
        shrink_factor = target_height / height
        return crop(
            image.resize((int(width * shrink_factor), target_height)),
            target_width,
            target_height,
        )
    delta_width = width - target_width
    delta_height = height - target_height
    horizontal_margin = int(delta_width / 2)
    vertical_margin = int(delta_height / 2)
    return image.crop(
        (
            horizontal_margin,
            vertical_margin,
            target_width + horizontal_margin,
            target_height + vertical_margin,
        )
    ).convert("RGB")


def load_mask(mask: ImageType, target_width: int, target_height: int):
    cropped_mask = crop(mask, target_width, target_height)
    return constants.ToTensor(cropped_mask.convert("L")).to(constants.device)


def fill(image: ImageType, target_width: int, target_height: int):
    width, height = image.size
    if width < target_width and height < target_height:
        if width > height:
            growth_factor = target_width / width
            resized_image = image.resize((target_width, int(height * growth_factor)))
        else:
            growth_factor = target_height / height
            resized_image = image.resize((int(width * growth_factor), target_height))
    padded_image = Image.fromarray(
        0 * np.ones((constants.Height, constants.Width, 3), np.uint8)
    )
    padded_image.paste(resized_image)
    return padded_image


def cut_mask(mask: ImageType, target_width: int, target_height: int):
    top = left = 512
    bottom = right = 0
    for x in range(mask.width):
        for y in range(mask.height):
            if mask.getpixel((x, y)) > 1:
                top = y if y < top else top
                left = x if x < left else left
                bottom = y if y > bottom else bottom
                right = x if x > right else right
    cropped_mask = mask.crop((left, top, right, bottom))
    cropped_mask_width, cropped_mask_height = cropped_mask.size
    filled_mask = fill(cropped_mask, target_width, target_height)
    return (
        top,
        left,
        bottom,
        right,
        cropped_mask_width,
        cropped_mask_height,
        filled_mask,
    )


def load_masks(
    mask: ImageType,
    masking_render_full: bool,
    masking_render_padding: int,
    mask_fading: int,
):
    init_mask = load_mask(
        mask if mask is not None else constants.fallback_image,
        constants.Width,
        constants.Height,
    )
    expanded_mask = (
        constants.ToTensor(constants.fallback_image).to(constants.device)
        if masking_render_full
        else expand_mask(init_mask, masking_render_padding)
    )
    blurred_mask = blur_mask(init_mask, mask_fading)
    (
        cropped_mask_top,
        cropped_mask_left,
        cropped_mask_bottom,
        cropped_mask_right,
        cropped_mask_width,
        cropped_mask_height,
        cropped_mask,
    ) = cut_mask(constants.ToPIL(expanded_mask[0]), constants.Width, constants.Height)
    return (
        cropped_mask,
        blurred_mask,
        cropped_mask_width,
        cropped_mask_height,
        cropped_mask_top,
        cropped_mask_left,
        cropped_mask_bottom,
        cropped_mask_right,
    )


def load_encode_image(
    mask: ImageType,
    image: ImageType,
    mask_offset_left: int,
    mask_offset_top: int,
    mask_offset_right: int,
    mask_offset_bottom: int,
):
    if mask is not None:
        cropped_image = image.crop(
            (mask_offset_left, mask_offset_top, mask_offset_right, mask_offset_bottom)
        )
        filled_image = fill(cropped_image, constants.Width, constants.Height)
        encode_image = load_img(filled_image, constants.Height, constants.Width).to(
            constants.device
        )
    else:
        encode_image = load_img(
            constants.fallback_image if not image else image,
            constants.Height,
            constants.Width,
        ).to(constants.device)
    return encode_image


def release_memory(model):
    if constants.device != "cpu":
        mem = torch.cuda.memory_allocated() / 1e6
        model.to("cpu")
        while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(1)


def generate_image_patch(
    sample_image: ImageType,
    cropped_mask: ImageType,
    cropped_mask_width: int,
    cropped_mask_height: int,
):
    sample_image_tensor = constants.ToTensor(sample_image).to(constants.device)
    cropped_mask_tensor = constants.ToTensor(cropped_mask).to(constants.device)
    masked_sample_image = constants.ToPIL(sample_image_tensor * cropped_mask_tensor)
    max_original_mask_dimension = max(cropped_mask_width, cropped_mask_height)
    return masked_sample_image.resize(
        (
            max_original_mask_dimension,
            max_original_mask_dimension,
        )
    )


def patch_image(
    image: ImageType,
    patch: ImageType,
    blurred_mask: Tensor,
    mask_offset_left: int,
    mask_offset_top: int,
):
    patched_image = image.copy()
    patched_image.paste(patch, (mask_offset_left, mask_offset_top))
    negative_mask = 1 - blurred_mask
    image_tensor = constants.ToTensor(image).to(constants.device)
    patched_image_tensor = constants.ToTensor(patched_image).to(
        constants.device
    )
    return constants.ToPIL(
        image_tensor * negative_mask + patched_image_tensor * blurred_mask
    )


def generate_final_sample(
    image: ImageType,
    mask: ImageType,
    blurred_mask: Tensor,
    sample_image: ImageType,
    cropped_mask: ImageType,
    cropped_mask_width: int,
    cropped_mask_height: int,
    mask_offset_left: int,
    mask_offset_top: int,
):
    if mask is not None:
        image_patch = generate_image_patch(
            sample_image,
            cropped_mask,
            cropped_mask_width,
            cropped_mask_height,
        )
        final_sample = patch_image(
            image,
            image_patch,
            blurred_mask,
            mask_offset_left,
            mask_offset_top,
        )
    else:
        final_sample = sample_image
    return final_sample
