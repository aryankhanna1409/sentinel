from PIL import Image

from neural_style_transfer.frontend import image_to_tensor


def test_image_to_tensor_converts_to_batched_rgb() -> None:
    image = Image.new("RGBA", (3, 2), color=(255, 128, 0, 64))

    result = image_to_tensor(image)

    assert result.shape == (1, 3, 2, 3)
    assert result.dtype.is_floating_point
    assert result.min().item() >= 0
    assert result.max().item() <= 1
