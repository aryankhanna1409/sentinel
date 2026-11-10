import pytest
import torch

from neural_style_transfer.core import StyleTransferConfig, gram_matrix


def test_gram_matrix_matches_normalized_channel_correlations() -> None:
    features = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])

    result = gram_matrix(features)

    expected = torch.tensor([[[1.25, 2.75], [2.75, 6.25]]])
    torch.testing.assert_close(result, expected)


def test_gram_matrix_requires_four_dimensions() -> None:
    with pytest.raises(ValueError, match="batch, channels"):
        gram_matrix(torch.ones(3, 3, 3))


@pytest.mark.parametrize(
    ("keyword", "message"),
    [
        ({"steps": 0}, "steps"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"style_layers": {}}, "style layer"),
    ],
)
def test_config_rejects_invalid_values(keyword: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StyleTransferConfig(**keyword)
