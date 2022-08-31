import gradio as gr
import src.backend as backend


def toggle_strength(image):
    return gr.Button.update(visible=image is not None)


def update_image_visibility(ddim_steps):
    visible_images = [gr.Image.update(visible=True)
                      for _ in range(0, ddim_steps)]
    hidden_images = [gr.Image.update(visible=False)
                     for _ in range(ddim_steps, 10)]
    return visible_images + hidden_images


def update_image_history(image_history: list, image_history_offset: int):
    out = list()
    image_history_chunk = image_history[
        int(-5 + image_history_offset): int(0 + image_history_offset) or None
    ]
    for history_img in image_history_chunk:
        out.append(gr.Image.update(value=history_img))
        out.append(gr.Button.update(visible=True))
    for _ in range(0, 5 - len(image_history_chunk)):
        out.append(gr.Image.update(value=None))
        out.append(gr.Button.update(visible=False))
    start_offset = max(len(image_history) - 5 + image_history_offset, 0)
    end_offset = len(image_history) + image_history_offset
    out.append(
        gr.Markdown.update(
            "### History (showing {})".format(
                "last 5"
                if end_offset == len(image_history)
                else "{} to {} of {}".format(
                    start_offset, end_offset, len(image_history)
                )
            )
        )
    )
    out.append(gr.Button.update(visible=start_offset > 0))
    out.append(gr.Button.update(visible=end_offset < len(image_history)))
    return out


def image_history_prev(image_history: list, image_history_offset: int):
    out = list()
    new_offset = min(max(image_history_offset - 5,
                     int(5 - len(image_history))), 0)
    out.append(new_offset)
    return tuple(out + update_image_history(image_history, new_offset))


def image_history_next(image_history: list, image_history_offset: int):
    out = list()
    new_offset = min(max(image_history_offset + 5,
                     int(5 - len(image_history))), 0)
    out.append(new_offset)
    return tuple(out + update_image_history(image_history, new_offset))


def set_image(image: str):
    return gr.Image.update(value=image)


def set_seed(seed: str):
    return seed


def generate(image,
             prompt,
             strength,
             ddim_steps,
             batch_size,
             seed,
             image_history):
    samples, _ = backend.generate(
        image,
        prompt,
        strength,
        ddim_steps,
        batch_size,
        seed,
        image_history)
    out = list()
    out.append(gr.Button.update(visible=True))
    out.append(gr.Button.update(visible=True))
    out.append(gr.Image.update(value=samples[0][0], label=str(samples[0][1])))
    out.append(samples[0][1])
    for sample_image, sample_seed in samples[1:]:
        out.append(gr.Row.update(visible=True))
        out.append(gr.Image.update(value=sample_image,
                   show_label=True, label=str(sample_seed), visible=True))
        out.append(sample_seed)
    for _ in range(0, 9 - len(samples[1:])):
        out.append(gr.Row.update(visible=False))
        out.append(
            gr.Image.update(
                value=None,
                show_label=False,
                label=None,
                visible=False,
            )
        )
        out.append("")
    out.append(image_history)
    return out