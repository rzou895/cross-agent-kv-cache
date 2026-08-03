from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from harness.inspect_kv_cache import tensor_bytes


def test_sequence_dimension_growth() -> None:
    """The sequence dimension should grow from 3 to 4."""

    key = torch.zeros(
        (1, 2, 3, 64),
        dtype=torch.bfloat16,
    )

    value = torch.zeros_like(key)

    new_key = torch.zeros(
        (1, 2, 1, 64),
        dtype=torch.bfloat16,
    )

    new_value = torch.zeros_like(new_key)

    grown_key = torch.cat(
        [key, new_key],
        dim=-2,
    )

    grown_value = torch.cat(
        [value, new_value],
        dim=-2,
    )

    assert key.shape[-2] == 3
    assert value.shape[-2] == 3
    assert grown_key.shape[-2] == 4
    assert grown_value.shape[-2] == 4


def test_byte_calculation() -> None:
    """Program and manual byte calculations should match."""

    key = torch.zeros(
        (1, 2, 3, 64),
        dtype=torch.bfloat16,
    )

    value = torch.zeros_like(key)

    program_bytes = tensor_bytes(key) + tensor_bytes(value)

    manual_bytes = (
        2          # K and V
        * 1        # batch size
        * 2        # KV heads
        * 3        # sequence length
        * 64       # head dimension
        * 2        # BF16 bytes per element
    )

    assert program_bytes == manual_bytes


def main() -> None:
    test_sequence_dimension_growth()
    test_byte_calculation()

    print("test_sequence_dimension_growth: PASSED")
    print("test_byte_calculation: PASSED")
    print("ALL LIGHTWEIGHT KV CACHE TESTS PASSED")


if __name__ == "__main__":
    main()