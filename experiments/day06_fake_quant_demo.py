from __future__ import annotations

import torch

from quantization.fake_quant import (
    calculate_symmetric_scale,
    fake_quantize_symmetric_int8,
)


def print_result(
    name: str,
    tensor: torch.Tensor,
    scale: float,
) -> None:
    result = fake_quantize_symmetric_int8(
        tensor=tensor,
        scale=scale,
    )

    print(f"\n===== {name} =====")
    print(f"Shape: {tuple(tensor.shape)}")
    print(f"Input dtype: {tensor.dtype}")
    print(f"Quantized dtype: {result.quantized.dtype}")
    print(f"Scale: {result.scale:.8f}")
    print(
        "Saturation rate: "
        f"{result.saturation_rate * 100.0:.4f}%"
    )
    print(f"MSE: {result.mse:.8f}")
    print(
        "Maximum absolute error: "
        f"{result.max_abs_error:.8f}"
    )
    print(
        "Quantized range: "
        f"[{result.quantized.min().item()}, "
        f"{result.quantized.max().item()}]"
    )


def main() -> None:
    torch.manual_seed(1234)

    # This tensor acts as the calibration data.
    calibration_tensor = torch.randn(
        4096,
        dtype=torch.float32,
    )

    fixed_scale = calculate_symmetric_scale(
        calibration_tensor
    )

    # Similar to the calibration distribution.
    in_distribution_tensor = torch.randn(
        4096,
        dtype=torch.float32,
    )

    # A deliberately wider distribution. Reusing the calibration scale
    # should cause some values to saturate at -127 or 127.
    wider_distribution_tensor = (
        torch.randn(
            4096,
            dtype=torch.float32,
        )
        * 3.0
    )

    print("Symmetric INT8 fake-quantization demonstration")
    print(f"Fixed calibration scale: {fixed_scale:.8f}")

    print_result(
        name="Calibration tensor",
        tensor=calibration_tensor,
        scale=fixed_scale,
    )

    print_result(
        name="Similar distribution",
        tensor=in_distribution_tensor,
        scale=fixed_scale,
    )

    print_result(
        name="Wider distribution",
        tensor=wider_distribution_tensor,
        scale=fixed_scale,
    )


if __name__ == "__main__":
    main()