import mimetypes
import gradio as gr
from transformers import logging
from src.frontend.components import HistoryImage, OutputImage
import src.frontend.events as events

logging.set_verbosity_error()
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


def render():
    with gr.Blocks(
        title="Image to image",
        css="""
            .svelte-10ogue4 {
                flex: 1;
            } 
        """,
    ) as blocks:
        image_history = gr.Variable(list())
        image_history_offset = gr.Variable(0)

        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel"):
                with gr.Group():
                    image = gr.Image(shape=(512, 512), type="pil", tool="editor")
                    prompt = gr.Text(label="Prompt")
                    strength = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.1,
                        label="Strength",
                        visible=False,
                    )
                    ddim_steps = gr.Slider(
                        minimum=1, maximum=200, value=10, step=1, label="Quality"
                    )
                    batch_size = gr.Slider(
                        minimum=1, maximum=10, step=1, label="Batch size"
                    )
                    seed = gr.Text(label="Seed")
                    submit = gr.Button("Generate", variant="primary")
                    submit.style(full_width="True")
            with gr.Column(variant="panel"):
                out_images = [
                    OutputImage(visible=True, show_label=True, actions_visible=False),
                    OutputImage(visible=False),
                    OutputImage(visible=False),
                    OutputImage(visible=False),
                    OutputImage(visible=False),
                    OutputImage(visible=False),
                    OutputImage(visible=False),
                    OutputImage(visible=False),
                    OutputImage(visible=False),
                    OutputImage(visible=False),
                ]
        image_history_header = gr.Markdown("### History (nothing so far)")
        with gr.Row():
            history_images = [
                HistoryImage(),
                HistoryImage(),
                HistoryImage(),
                HistoryImage(),
                HistoryImage(),
            ]
        image_history_prev_btn = gr.Button("Older", visible=False)
        image_history_next_btn = gr.Button("Newer", visible=False)

        # scroll image history backwards
        image_history_prev_btn.click(
            fn=events.image_history_prev,
            inputs=[image_history, image_history_offset],
            outputs=[
                image_history_offset,
                history_images[0].image,
                history_images[0].rollback_btn,
                history_images[1].image,
                history_images[1].rollback_btn,
                history_images[2].image,
                history_images[2].rollback_btn,
                history_images[3].image,
                history_images[3].rollback_btn,
                history_images[4].image,
                history_images[4].rollback_btn,
                image_history_header,
                image_history_prev_btn,
                image_history_next_btn,
            ],
        )
        # scroll image history forwards
        image_history_next_btn.click(
            fn=events.image_history_next,
            inputs=[image_history, image_history_offset],
            outputs=[
                image_history_offset,
                history_images[0].image,
                history_images[0].rollback_btn,
                history_images[1].image,
                history_images[1].rollback_btn,
                history_images[2].image,
                history_images[2].rollback_btn,
                history_images[3].image,
                history_images[3].rollback_btn,
                history_images[4].image,
                history_images[4].rollback_btn,
                image_history_header,
                image_history_prev_btn,
                image_history_next_btn,
            ],
        )
        # show / hide strength slider based on whether the input image is present
        image.change(
            fn=events.toggle_strength,
            inputs=[image],
            outputs=[strength],
            show_progress=False,
        )
        # update the image history after a new image is done generating
        out_images[0].image.change(
            fn=events.update_image_history,
            inputs=[image_history, image_history_offset],
            outputs=[
                history_images[0].image,
                history_images[0].rollback_btn,
                history_images[1].image,
                history_images[1].rollback_btn,
                history_images[2].image,
                history_images[2].rollback_btn,
                history_images[3].image,
                history_images[3].rollback_btn,
                history_images[4].image,
                history_images[4].rollback_btn,
                image_history_header,
                image_history_prev_btn,
                image_history_next_btn,
            ],
        )
        # handle "extend" and "copy seed" on each output image
        for out_image in out_images:
            out_image.extend_btn.click(
                fn=events.set_image, inputs=[out_image.image], outputs=[image]
            )
            out_image.copy_seed_btn.click(
                fn=events.set_seed, inputs=[out_image.seed], outputs=[seed]
            )

        for history_image in history_images:
            history_image.rollback_btn.click(
                fn=events.set_image, inputs=[history_image.image], outputs=[image]
            )
        # generate a new image
        submit.click(
            fn=events.generate,
            inputs=[
                image,
                prompt,
                strength,
                ddim_steps,
                batch_size,
                seed,
                image_history,
            ],
            outputs=[
                out_images[0].extend_btn,
                out_images[0].copy_seed_btn,
                out_images[0].image,
                out_images[0].seed,
                out_images[1].row,
                out_images[1].image,
                out_images[1].seed,
                out_images[2].row,
                out_images[2].image,
                out_images[2].seed,
                out_images[3].row,
                out_images[3].image,
                out_images[3].seed,
                out_images[4].row,
                out_images[4].image,
                out_images[4].seed,
                out_images[5].row,
                out_images[5].image,
                out_images[5].seed,
                out_images[6].row,
                out_images[6].image,
                out_images[6].seed,
                out_images[7].row,
                out_images[7].image,
                out_images[7].seed,
                out_images[8].row,
                out_images[8].image,
                out_images[8].seed,
                out_images[9].row,
                out_images[9].image,
                out_images[9].seed,
                image_history,
            ],  # type: ignore
        )

    blocks.launch(server_name="0.0.0.0", share=True)
