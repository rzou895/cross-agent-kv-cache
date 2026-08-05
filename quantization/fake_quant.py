from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# We use the symmetric range [-127, 127].
# The value -128 is deliberately not used.
INT8_MIN = -127
INT8_MAX = 127


@dataclass(frozen=True)
class FakeQuantResult:
    """Results produced by one fake-quantization operation."""

    quantized: torch.Tensor
    dequantized: torch.Tensor
    scale: float
    saturation_rate: float
    mse: float
    max_abs_error: float


def validate_input_tensor(tensor: torch.Tensor) -> None:
    """Check that the tensor can be quantized."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    if tensor.numel() == 0:
        raise ValueError("Input tensor must not be empty.")

    if not tensor.is_floating_point():
        raise TypeError("Input tensor must have a floating-point dtype.")

    if not torch.isfinite(tensor).all():
        raise ValueError("Input tensor must contain only finite values.")


def calculate_symmetric_scale(tensor: torch.Tensor) -> float:
    """Calculate one symmetric INT8 scale for the complete tensor."""

    validate_input_tensor(tensor)

    max_abs = float(tensor.detach().abs().max().item())

    # A zero tensor has no natural non-zero scale.
    # Using 1.0 avoids division by zero while preserving all zeros.
    if max_abs == 0.0:
        return 1.0

    return max_abs / INT8_MAX


def validate_scale(scale: float) -> float:
    """Validate and return a floating-point scale."""

    scale = float(scale)

    if not math.isfinite(scale):
        raise ValueError("Scale must be finite.")

    if scale <= 0.0:
        raise ValueError("Scale must be greater than zero.")

    return scale


def quantize_symmetric_int8(
    tensor: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, float, float]:
    """
    Quantize a floating-point tensor to symmetric INT8.

    If scale is None, the scale is calculated from the input tensor.
    If scale is provided, the supplied fixed scale is reused.
    """

    validate_input_tensor(tensor)

    actual_scale = (
        calculate_symmetric_scale(tensor)
        if scale is None
        else validate_scale(scale)
    )

    scaled = tensor / actual_scale
    rounded = torch.round(scaled)

    saturation_mask = (
        (rounded < INT8_MIN)
        | (rounded > INT8_MAX)
    )

    quantized = torch.clamp(
        rounded,
        min=INT8_MIN,
        max=INT8_MAX,
    ).to(torch.int8)

    saturation_rate = float(
        saturation_mask.to(torch.float32).mean().item()
    )

    return quantized, actual_scale, saturation_rate


def dequantize_symmetric_int8(
    quantized: torch.Tensor,
    scale: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert an integer tensor back to floating point."""

    if not isinstance(quantized, torch.Tensor):
        raise TypeError("Quantized input must be a torch.Tensor.")

    if quantized.numel() == 0:
        raise ValueError("Quantized tensor must not be empty.")

    actual_scale = validate_scale(scale)

    return quantized.to(dtype) * actual_scale


def fake_quantize_symmetric_int8(
    tensor: torch.Tensor,
    scale: float | None = None,
) -> FakeQuantResult:
    """
    Quantize and immediately dequantize a tensor.

    This simulates the error introduced by INT8 storage while returning
    a floating-point tensor that can still be used by PyTorch operations.
    """

    quantized, actual_scale, saturation_rate = (
        quantize_symmetric_int8(
            tensor=tensor,
            scale=scale,
        )
    )

    dequantized = dequantize_symmetric_int8(
        quantized=quantized,
        scale=actual_scale,
        dtype=tensor.dtype,
    )

    error = (
        dequantized.to(torch.float32)
        - tensor.to(torch.float32)
    )

    mse = float(torch.mean(error.square()).item())
    max_abs_error = float(error.abs().max().item())

    return FakeQuantResult(
        quantized=quantized,
        dequantized=dequantized,
        scale=actual_scale,
        saturation_rate=saturation_rate,
        mse=mse,
        max_abs_error=max_abs_error,
    )