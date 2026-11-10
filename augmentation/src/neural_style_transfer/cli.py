"""Command-line entry point and image conversion helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from .core import StyleTransfer, StyleTransferConfig


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def resize_to_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_edge / max(width, height))
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(resized, Image.Resampling.LANCZOS)


def load_image(path: Path, size: tuple[int, int] | None = None, max_edge: int | None = None) -> torch.Tensor:
    """Load an image as a batched RGB float tensor in ``[0, 1]``."""

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
    if size is not None:
        image = image.resize(size, Image.Resampling.LANCZOS)
    elif max_edge is not None:
        image = resize_to_max_edge(image, max_edge)
    return pil_to_tensor(image).float().div(255).unsqueeze(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transfer the style of one image onto the content of another using VGG-19."
    )
    parser.add_argument("--content", required=True, type=Path, help="Path to the content image.")
    parser.add_argument("--style", required=True, type=Path, help="Path to the style image.")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/styled.png"), help="Where to save the PNG result."
    )
    parser.add_argument("--steps", type=_positive_int, default=500, help="Number of optimization steps.")
    parser.add_argument(
        "--image-size", type=_positive_int, default=512, help="Maximum size of the content image's long edge."
    )
    parser.add_argument("--learning-rate", type=_positive_float, default=0.02)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=1_000_000.0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto", help="Processing device (default: auto)."
    )
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = build_parser().parse_args()
    if args.content_weight < 0 or args.style_weight < 0:
        raise SystemExit("--content-weight and --style-weight cannot be negative")

    content = load_image(args.content, max_edge=args.image_size)
    _, _, height, width = content.shape
    style = load_image(args.style, size=(width, height))
    device = resolve_device(args.device)
    config = StyleTransferConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
    )

    print(f"Using {device}; optimizing {width}x{height} image for {config.steps} steps.")
    transfer = StyleTransfer(device=device, config=config)

    def report(step: int, content_loss: float, style_loss: float, total_loss: float) -> None:
        print(
            f"step {step:>4}/{config.steps} | content {content_loss:.5f} | "
            f"style {style_loss:.7f} | total {total_loss:.5f}"
        )

    result = transfer.run(content, style, progress=report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(result.squeeze(0)).save(args.output, format="PNG")
    print(f"Saved result to {args.output.resolve()}")
