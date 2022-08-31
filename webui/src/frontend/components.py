import gradio as gr


class OutputImage:
    row: gr.Row
    image: gr.Image
    extend_btn: gr.Button
    copy_seed_btn: gr.Button

    def __init__(self, visible: bool, show_label: bool = False, actions_visible: bool = True):
        with gr.Row(visible=visible) as row:
            with gr.Group():
                self.image = gr.Image(
                    type="filepath", show_label=show_label, interactive=False)
                with gr.Row():
                    with gr.Column():
                        self.extend_btn = gr.Button(
                            "Extend", visible=actions_visible)
                        self.extend_btn.style(full_width="True")
                    with gr.Column():
                        self.copy_seed_btn = gr.Button(
                            "Copy seed", visible=actions_visible)
                        self.copy_seed_btn.style(full_width="True")
        self.seed = gr.Text(visible=False)
        self.row = row


class HistoryImage:
    image: gr.Image
    rollback_btn: gr.Button

    def __init__(self):
        with gr.Column():
            with gr.Group():
                self.image = gr.Image(show_label=False, interactive=False)
                self.rollback_btn = gr.Button("Rollback", visible=False)
                self.rollback_btn.style(full_width="True")
