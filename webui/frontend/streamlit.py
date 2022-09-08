import streamlit as st
from PIL import Image
from PIL.Image import Image as ImageType
import numpy as np
from streamlit_drawable_canvas import st_canvas
import webui.backend as backend
from torchvision.transforms import transforms

ToTensor = transforms.ToTensor()
ToPIL = transforms.ToPILImage()

masking_render_full = False
masking_render_padding = 100
mask_fading = 10
prompt = ""
seed = None
mask = None
strength = None

def merge(above: ImageType, below: ImageType):
    above_tensor = ToTensor(above)
    above_tensor_mask = 1 - (above_tensor * 1000).clamp(0, 1)
    below_tensor = ToTensor(below)
    masked_below_tensor = below_tensor * above_tensor_mask
    return ToPIL(above_tensor + masked_below_tensor)


def generate(
    image,
    mask,
    prompt,
    strength,
    ddim_steps,
    batch_size,
    seed,
    masking_render_full,
    masking_render_padding,
    mask_fading,
):
    samples, _ = backend.generate(
        image,
        mask,
        prompt,
        strength,
        ddim_steps,
        batch_size,
        seed,
        masking_render_full,
        masking_render_padding,
        mask_fading,
    )
    st.session_state.output_samples = samples
    input_images_history = st.session_state.get("input_images_history") or []
    if image is not None and (
        len(input_images_history) == 0
        or not np.array_equal(image.getdata(), input_images_history[-1].getdata())
    ):
        st.session_state.input_images_history = (
            [image]
            if len(input_images_history) == 0
            else input_images_history + [image]
        )


def override_image(new_image):
    st.session_state.image = new_image


def clear_image_override():
    st.session_state.image = None

main_tab, editors_tab = st.tabs(["Main", "Editors"])
with main_tab:
    inputs_column_left, inputs_column_right = st.columns(2, gap="medium")
with editors_tab:
    editors_column, tools_column = st.columns([3, 1])

with main_tab:
    with inputs_column_left:
        input_image = st.file_uploader(
            "Input image", type=["png", "jpg"], accept_multiple_files=False
        )
        image = (
            st.session_state.get("image")
            if st.session_state.get("image") is not None
            else input_image
        )
        if image is not None:
            if st.session_state.get("image") is None:
                image = (
                    backend.crop(Image.open(image), 512, 512)
                    if image is not None
                    else None
                )

with editors_tab:
    with editors_column:
        enable_drawing = st.checkbox("Enable drawing")
    if enable_drawing:
        with tools_column:
            drawing_mode = st.selectbox(
                "Drawing tool:",
                ("freedraw", "line", "rect", "circle", "transform"),
                key="draw_drawing_mode",
            )
            stroke_width = st.slider(
                "Stroke width: ",
                min_value=1,
                max_value=100,
                value=25,
                key="draw_stroke_width",
            )
            stroke_color = st.color_picker(
                "Stroke color", "#FFF", key="draw_stroke_color"
            )
            background_color = st.color_picker(
                "Background color", "#000", key="draw_background_color"
            )
    with editors_column:
        if enable_drawing:
            drawing_canvas = st_canvas(
                width=512,
                height=512,
                fill_color="rgba(255, 255, 255, 1)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=background_color,
                background_image=image,
                update_streamlit=True,
                drawing_mode=drawing_mode,
                point_display_radius=0,
                key="draw",
            )
            drawn_image = (
                Image.fromarray(
                    drawing_canvas.image_data.astype("uint8"), mode="RGBA"
                ).convert("RGB")
                if drawing_canvas.image_data is not None
                else None
            )
            image = (
                merge(
                    drawn_image,
                    image,
                )
                if image is not None and drawn_image is not None
                else drawn_image
            )

    if image is not None:
        editors_column, tools_column = st.columns([3, 1])
        with editors_column:
            enable_masking = st.checkbox("Enable masking")
        if enable_masking:
            with tools_column:
                drawing_mode = st.selectbox(
                    "Drawing tool:",
                    ("freedraw", "line", "rect", "circle", "transform"),
                    key="mask_drawing_mode",
                )
                stroke_width = st.slider(
                    "Stroke width: ",
                    min_value=1,
                    max_value=100,
                    value=25,
                    key="mask_stroke_width",
                )
            with st.sidebar:
                masking_render_padding = st.slider(
                    "Padding around mask",
                    min_value=0,
                    max_value=100,
                    value=masking_render_padding,
                )
                mask_fading = st.slider(
                    "Mask blur radius",
                    min_value=0,
                    max_value=100,
                    value=mask_fading,
                )
        with editors_column:
            if enable_masking:
                with st.container():
                    drawing_canvas = st_canvas(
                        width=512,
                        height=512,
                        fill_color="rgba(255, 255, 255, 1)",  # Fixed fill color with some opacity
                        stroke_width=stroke_width,
                        stroke_color="rgba(255, 255, 255, 1)",
                        background_image=image.point(lambda point: point * 0.5),
                        update_streamlit=True,
                        drawing_mode=drawing_mode,
                        point_display_radius=0,
                        key="mask",
                    )
                    mask = (
                        Image.fromarray(
                            drawing_canvas.image_data.astype("uint8"), mode="RGBA"
                        )
                        if drawing_canvas.image_data is not None
                        else None
                    )


with main_tab:

    with inputs_column_left:
        if image is not None:
            st.image(image)
            if input_image is not None and st.session_state.get("image") is not None:
                st.button("Use input image instead", on_click=clear_image_override)
            if input_image is None and st.session_state.get("image") is not None:
                st.button("Clear image", on_click=clear_image_override)

    with inputs_column_right:
        prompt = st.text_area("Prompt")
        seed = st.text_input("Seed")
        ddim_steps = st.slider("Quality", min_value=1, max_value=50, step=1, value=10)
        batch_size = st.slider("Batch size", min_value=1, max_value=10, step=1, value=1)
        if image is not None:
            strength = st.slider(
                "Prompt influence",
                min_value=0.1,
                max_value=0.9,
                step=0.1,
                value=0.5,
            )

    st.button(
        "Generate {} image{}".format(batch_size, "s" if batch_size > 1 else ""),
        on_click=generate,
        args=[
            image,
            mask,
            prompt,
            strength,
            ddim_steps,
            batch_size,
            seed,
            masking_render_padding == 100,
            masking_render_padding,
            mask_fading,
        ],
    )

    with st.expander("Outputs", expanded=True):
        output_samples = st.session_state.get("output_samples")
        if output_samples is not None:
            image_columns = st.columns(2)
            for i in range(len(output_samples)):
                with image_columns[i % 2]:
                    sample_image, sample_seed = output_samples[i]
                    st.image(sample_image, use_column_width="always")
                    st.button(
                        "Use this image",
                        key=sample_seed,
                        on_click=override_image,
                        args=[sample_image],
                    )

    with st.expander("Input images history"):
        input_images_history = st.session_state.get("input_images_history")
        if input_images_history is not None:
            image_columns = st.columns(4)
            for i in range(len(input_images_history)):
                with image_columns[i % 4]:
                    st.image(input_images_history[i], use_column_width="always")
                    st.button(
                        "Use this image",
                        key=i,
                        on_click=override_image,
                        args=[input_images_history[i]],
                    )
st.markdown(
    """
    <style>
        button[kind=primary] {
            width: 100% !important
        }
    </style>
""",
    unsafe_allow_html=True,
)
