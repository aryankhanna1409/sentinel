"""The perceptual-loss model and image optimization loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights, vgg19


DEFAULT_STYLE_LAYERS: dict[str, float] = {
    "conv1_1": 1.0,
    "conv2_1": 0.8,
    "conv3_1": 0.5,
    "conv4_1": 0.3,
    "conv5_1": 0.1,
}


@dataclass(frozen=True)
class StyleTransferConfig:
    """Tunable values for one style-transfer run."""

    steps: int = 500
    learning_rate: float = 0.02
    content_weight: float = 1.0
    style_weight: float = 1_000_000.0
    content_layer: str = "conv4_2"
    style_layers: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_STYLE_LAYERS.copy()
    )

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.content_weight < 0 or self.style_weight < 0:
            raise ValueError("content_weight and style_weight cannot be negative")
        if not self.style_layers:
            raise ValueError("at least one style layer is required")
        if any(weight < 0 for weight in self.style_layers.values()):
            raise ValueError("style-layer weights cannot be negative")


def gram_matrix(features: Tensor) -> Tensor:
    """Return a size-normalized feature-correlation matrix for each image.

    ``features`` has shape ``(batch, channels, height, width)``. Flattening the
    spatial dimensions and multiplying by the transpose removes spatial location
    while retaining how strongly feature channels co-occur: the style signature.
    """

    if features.ndim != 4:
        raise ValueError("features must have shape (batch, channels, height, width)")

    batch, channels, height, width = features.shape
    flattened = features.reshape(batch, channels, height * width)
    return torch.bmm(flattened, flattened.transpose(1, 2)) / (channels * height * width)


class VGGFeatureExtractor(nn.Module):
    """Expose selected VGG-19 activations with paper-style layer names."""

    def __init__(self, layers: set[str]) -> None:
        super().__init__()
        self.layers = layers
        self.network = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        activations: dict[str, Tensor] = {}
        block, convolution = 1, 0
        current = image

        for module in self.network:
            current = module(current)
            if isinstance(module, nn.Conv2d):
                convolution += 1
            elif isinstance(module, nn.ReLU):
                layer_name = f"conv{block}_{convolution}"
                if layer_name in self.layers:
                    activations[layer_name] = current
                    if len(activations) == len(self.layers):
                        break
            elif isinstance(module, nn.MaxPool2d):
                block, convolution = block + 1, 0

        missing = self.layers.difference(activations)
        if missing:
            missing_layers = ", ".join(sorted(missing))
            raise ValueError(f"Unknown or unavailable VGG layer(s): {missing_layers}")
        return activations


ProgressCallback = Callable[[int, float, float, float], None]


class StyleTransfer:
    """Optimize an image to preserve content while matching another image's style."""

    def __init__(self, device: torch.device, config: StyleTransferConfig | None = None) -> None:
        self.device = device
        self.config = config or StyleTransferConfig()
        requested_layers = set(self.config.style_layers) | {self.config.content_layer}
        self.extractor = VGGFeatureExtractor(requested_layers).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def normalize(self, image: Tensor) -> Tensor:
        return (image - self.mean) / self.std

    def denormalize(self, image: Tensor) -> Tensor:
        return image * self.std + self.mean

    def _clamp_to_valid_image_range(self, image: Tensor) -> None:
        lower = -self.mean / self.std
        upper = (1 - self.mean) / self.std
        with torch.no_grad():
            image.copy_(torch.maximum(torch.minimum(image, upper), lower))

    def run(
        self,
        content: Tensor,
        style: Tensor,
        progress: ProgressCallback | None = None,
    ) -> Tensor:
        """Run style transfer and return an RGB tensor in the range ``[0, 1]``.

        Inputs must be unnormalized tensors with shape ``(1, 3, height, width)``.
        The style image should use the same dimensions as the content image.
        """

        if content.shape != style.shape:
            raise ValueError("content and style tensors must have identical shapes")
        if content.ndim != 4 or content.shape[:2] != (1, 3):
            raise ValueError("inputs must have shape (1, 3, height, width)")

        content = self.normalize(content.to(self.device))
        style = self.normalize(style.to(self.device))
        with torch.no_grad():
            content_targets = self.extractor(content)
            style_targets = self.extractor(style)
            style_grams = {
                layer: gram_matrix(style_targets[layer]).detach()
                for layer in self.config.style_layers
            }

        generated = content.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([generated], lr=self.config.learning_rate)
        report_every = max(1, self.config.steps // 20)

        for step in range(1, self.config.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            generated_features = self.extractor(generated)
            content_loss = F.mse_loss(
                generated_features[self.config.content_layer],
                content_targets[self.config.content_layer],
            )
            style_loss = torch.zeros((), device=self.device)
            for layer, layer_weight in self.config.style_layers.items():
                style_loss = style_loss + layer_weight * F.mse_loss(
                    gram_matrix(generated_features[layer]), style_grams[layer]
                )

            total_loss = (
                self.config.content_weight * content_loss
                + self.config.style_weight * style_loss
            )
            total_loss.backward()
            optimizer.step()
            self._clamp_to_valid_image_range(generated)

            if progress and (step == 1 or step == self.config.steps or step % report_every == 0):
                progress(
                    step,
                    content_loss.detach().item(),
                    style_loss.detach().item(),
                    total_loss.detach().item(),
                )

        return self.denormalize(generated.detach()).clamp(0, 1).cpu()
