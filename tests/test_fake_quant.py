from __future__ import annotations

import pytest
import torch

from quantization.fake_quant import (
    calculate_symmetric_scale,
    fake_quantize_symmetric_int8,
    quantize_symmetric_int8,
)


def test_zero_tensor() -> None:
    tensor = torch.zeros(8, dtype=torch.float32)

    result = fake_quantize_symmetric_int8(tensor)

    assert result.scale == 1.0
    assert result.saturation_rate == 0.0
    assert result.mse == 0.0
    assert result.max_abs_error == 0.0

    assert torch.equal(
        result.quantized,
        torch.zeros(8, dtype=torch.int8),
    )

    assert torch.equal(
        result.dequantized,
        tensor,
    )


def test_exact_quantization_boundaries() -> None:
    scale = 0.5

    tensor = torch.tensor(
        [-63.5, -1.0, 0.0, 1.0, 63.5],
        dtype=torch.float32,
    )

    quantized, returned_scale, saturation_rate = (
        quantize_symmetric_int8(
            tensor=tensor,
            scale=scale,
        )
    )

    expected = torch.tensor(
        [-127, -2, 0, 2, 127],
        dtype=torch.int8,
    )

    assert returned_scale == scale
    assert saturation_rate == 0.0
    assert torch.equal(quantized, expected)


def test_extreme_values_are_clamped() -> None:
    tensor = torch.tensor(
        [-100.0, 0.0, 100.0],
        dtype=torch.float32,
    )

    result = fake_quantize_symmetric_int8(
        tensor=tensor,
        scale=0.5,
    )

    expected = torch.tensor(
        [-127, 0, 127],
        dtype=torch.int8,
    )

    assert torch.equal(result.quantized, expected)
    assert result.saturation_rate == pytest.approx(2.0 / 3.0)
    assert result.max_abs_error > 0.0


def test_automatically_calculated_scale() -> None:
    tensor = torch.tensor(
        [-2.0, 0.0, 1.0],
        dtype=torch.float32,
    )

    scale = calculate_symmetric_scale(tensor)

    assert scale == pytest.approx(2.0 / 127.0)


def test_quantization_error_is_bounded_without_saturation() -> None:
    torch.manual_seed(1234)

    tensor = torch.randn(
        1024,
        dtype=torch.float32,
    )

    result = fake_quantize_symmetric_int8(tensor)

    assert result.quantized.dtype == torch.int8
    assert result.dequantized.dtype == tensor.dtype
    assert result.dequantized.shape == tensor.shape
    assert result.saturation_rate == 0.0

    # Rounding to the nearest integer should produce an error no larger
    # than approximately half one quantization step.
    assert result.max_abs_error <= result.scale / 2.0 + 1e-6


@pytest.mark.parametrize(
    "invalid_scale",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_invalid_scale_raises_error(
    invalid_scale: float,
) -> None:
    tensor = torch.ones(4, dtype=torch.float32)

    with pytest.raises(ValueError):
        fake_quantize_symmetric_int8(
            tensor=tensor,
            scale=invalid_scale,
        )


def test_integer_input_is_rejected() -> None:
    tensor = torch.ones(4, dtype=torch.int32)

    with pytest.raises(TypeError):
        fake_quantize_symmetric_int8(tensor)