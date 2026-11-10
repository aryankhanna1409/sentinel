"""A local browser UI for neural style transfer."""

from __future__ import annotations

from typing import Any

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from .cli import resize_to_max_edge, resolve_device
from .core import StyleTransfer, StyleTransferConfig


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a batched RGB tensor with values in ``[0, 1]``."""

    return pil_to_tensor(image.convert("RGB")).float().div(255).unsqueeze(0)


def transfer_images(
    content_image: Image.Image | None,
    style_image: Image.Image | None,
    steps: int,
    image_size: int,
    style_weight: float,
    device_name: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[Image.Image, str]:
    """Run the model for the Gradio UI and return its image plus a status label."""

    if content_image is None or style_image is None:
        raise gr.Error("Upload both a content image and a style image first.")

    content = resize_to_max_edge(content_image.convert("RGB"), image_size)
    style = style_image.convert("RGB").resize(content.size, Image.Resampling.LANCZOS)
    content_tensor = image_to_tensor(content)
    style_tensor = image_to_tensor(style)
    device = resolve_device(device_name)
    config = StyleTransferConfig(steps=steps, style_weight=style_weight)
    progress(0, desc="Loading VGG-19 features…")
    transfer = StyleTransfer(device=device, config=config)

    def update_progress(step: int, _: float, __: float, total: float) -> None:
        progress(
            step / steps,
            desc=f"Optimizing image — step {step}/{steps} (loss {total:.4f})",
        )

    result = transfer.run(content_tensor, style_tensor, progress=update_progress)
    output = to_pil_image(result.squeeze(0))
    width, height = content.size
    return output, f"Done — {width}×{height}px, {steps} optimization steps on {device}."


def build_demo() -> gr.Blocks:
    """Create the local, two-image style-transfer experience."""

    with gr.Blocks(title="Neural Style Transfer") as demo:
        gr.Markdown(
            "# Neural Style Transfer\n"
            "Upload a **content** image for the composition and a **style** image "
            "for the texture, colour, and visual character."
        )
        with gr.Row():
            content = gr.Image(
                label="Content image", type="pil", sources=["upload", "clipboard"], height=360
            )
            style = gr.Image(
                label="Style image", type="pil", sources=["upload", "clipboard"], height=360
            )
            output = gr.Image(label="Styled result", format="png", height=360)

        with gr.Row():
            steps = gr.Slider(50, 1_000, value=400, step=50, label="Quality (optimization steps)")
            image_size = gr.Slider(128, 768, value=512, step=64, label="Output resolution")
        with gr.Accordion("Advanced controls", open=False):
            style_weight = gr.Slider(
                100_000,
                2_000_000,
                value=1_000_000,
                step=100_000,
                label="Style strength",
            )
            device = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Compute device")

        run_button = gr.Button("Create styled image", variant="primary")
        status = gr.Markdown("Ready when you are.")
        run_button.click(
            transfer_images,
            inputs=[content, style, steps, image_size, style_weight, device],
            outputs=[output, status],
            show_progress="full",
        )

    return demo


def main() -> None:
    build_demo().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1")


if __name__ == "__main__":
    main()
