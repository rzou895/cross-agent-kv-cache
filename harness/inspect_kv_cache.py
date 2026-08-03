from __future__ import annotations

import os
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ID = os.environ.get(
    "MODEL_ID",
    "Qwen/Qwen2.5-0.5B-Instruct",
) 

MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "Briefly explain what a transformer KV cache is.",
    },
]


def choose_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """Choose the device and model data type."""

    if torch.cuda.is_available():
        device = torch.device("cuda")

        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    return device, dtype


def iter_cache_layers(
    cache: object,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    """
    Yield layer index, key tensor and value tensor.

    Transformers 5 uses cache.layers[i].keys and .values.
    The fallback also supports older Transformers cache formats.
    """

    # Transformers 5.x cache structure
    if hasattr(cache, "layers"):
        layers = getattr(cache, "layers")

        for layer_index, layer in enumerate(layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)

            if key is None or value is None:
                raise RuntimeError(
                    f"Cache layer {layer_index} is not initialized."
                )

            yield layer_index, key, value

        return

    # Older Transformers cache structure
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        key_cache = getattr(cache, "key_cache")
        value_cache = getattr(cache, "value_cache")

        for layer_index, (key, value) in enumerate(
            zip(key_cache, value_cache)
        ):
            yield layer_index, key, value

        return

    # Legacy tuple format
    try:
        for layer_index, layer_cache in enumerate(cache):
            key, value = layer_cache[:2]
            yield layer_index, key, value
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Unsupported cache type: {type(cache).__name__}"
        ) from error


def tensor_bytes(tensor: torch.Tensor) -> int:
    """Calculate the storage occupied by one tensor."""

    return tensor.numel() * tensor.element_size()


def inspect_cache(
    cache: object,
    label: str,
) -> dict[str, object]:
    """Print and summarize all KV cache layers."""

    layers = list(iter_cache_layers(cache))

    if not layers:
        raise RuntimeError("No KV cache layers were returned.")

    total_bytes = 0
    sequence_lengths: set[int] = set()

    print(f"\n===== {label} =====")
    print(f"Cache class: {type(cache).__name__}")
    print(f"Number of cache layers: {len(layers)}")

    for layer_index, key, value in layers:
        if key.shape != value.shape:
            raise AssertionError(
                f"Layer {layer_index}: K and V shapes do not match: "
                f"{tuple(key.shape)} versus {tuple(value.shape)}"
            )

        sequence_lengths.add(int(key.shape[-2]))

        key_bytes = tensor_bytes(key)
        value_bytes = tensor_bytes(value)
        layer_bytes = key_bytes + value_bytes
        total_bytes += layer_bytes

        print(
            f"Layer {layer_index:02d}: "
            f"K shape={tuple(key.shape)}, "
            f"V shape={tuple(value.shape)}, "
            f"dtype={key.dtype}, "
            f"device={key.device}, "
            f"bytes={layer_bytes}"
        )

    if len(sequence_lengths) != 1:
        raise AssertionError(
            "Not all cache layers have the same sequence length: "
            f"{sorted(sequence_lengths)}"
        )

    sequence_length = sequence_lengths.pop()

    print(f"Cache sequence length: {sequence_length}")
    print(f"Total cache bytes: {total_bytes}")
    print(f"Total cache KiB: {total_bytes / 1024:.3f}")
    print(f"Total cache MiB: {total_bytes / 1024**2:.6f}")

    return {
        "layer_count": len(layers),
        "sequence_length": sequence_length,
        "total_bytes": total_bytes,
        "element_size": layers[0][1].element_size(),
        "dtype": layers[0][1].dtype,
        "device": layers[0][1].device,
    }


def expected_cache_bytes(
    model_config: object,
    batch_size: int,
    sequence_length: int,
    element_size: int,
) -> int:
    """
    Calculate KV cache bytes manually.

    Formula:
        layers
        × 2 for K and V
        × batch size
        × KV heads
        × sequence length
        × head dimension
        × bytes per element
    """

    number_of_layers = int(model_config.num_hidden_layers)

    number_of_kv_heads = int(
        getattr(
            model_config,
            "num_key_value_heads",
            model_config.num_attention_heads,
        )
    )

    head_dimension = int(
        getattr(
            model_config,
            "head_dim",
            model_config.hidden_size
            // model_config.num_attention_heads,
        )
    )

    return (
        number_of_layers
        * 2
        * batch_size
        * number_of_kv_heads
        * sequence_length
        * head_dimension
        * element_size
    )


def print_manual_calculation(
    model_config: object,
    batch_size: int,
    sequence_length: int,
    element_size: int,
    program_bytes: int,
) -> int:
    """Print the manual KV cache byte calculation."""

    number_of_layers = int(model_config.num_hidden_layers)

    number_of_kv_heads = int(
        getattr(
            model_config,
            "num_key_value_heads",
            model_config.num_attention_heads,
        )
    )

    head_dimension = int(
        getattr(
            model_config,
            "head_dim",
            model_config.hidden_size
            // model_config.num_attention_heads,
        )
    )

    manual_bytes = expected_cache_bytes(
        model_config=model_config,
        batch_size=batch_size,
        sequence_length=sequence_length,
        element_size=element_size,
    )

    print("\n===== Manual byte calculation =====")
    print(
        f"{number_of_layers} layers "
        f"× 2 (K and V) "
        f"× {batch_size} batch "
        f"× {number_of_kv_heads} KV heads "
        f"× {sequence_length} tokens "
        f"× {head_dimension} head dimension "
        f"× {element_size} bytes"
    )
    print(f"Manual result:  {manual_bytes} bytes")
    print(f"Program result: {program_bytes} bytes")
    print(f"Difference:     {program_bytes - manual_bytes} bytes")

    return manual_bytes


def run_inspection() -> dict[str, int]:
    """Run prefill and one-token decode cache inspection."""

    device, dtype = choose_device_and_dtype()

    print("===== Model setup =====")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Model dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    inputs = tokenizer.apply_chat_template(
        MESSAGES,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        name: tensor.to(device)
        for name, tensor in inputs.items()
    }

    batch_size = int(inputs["input_ids"].shape[0])
    prompt_length = int(inputs["input_ids"].shape[1])

    print(f"Batch size: {batch_size}")
    print(f"Prompt token count: {prompt_length}")
    print(f"Configured layers: {model.config.num_hidden_layers}")
    print(
        "Configured attention heads: "
        f"{model.config.num_attention_heads}"
    )
    print(
        "Configured KV heads: "
        f"{model.config.num_key_value_heads}"
    )

    # Prefill: process the complete prompt and create its KV cache.
    with torch.inference_mode():
        prefill_outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )

    prefill_cache = prefill_outputs.past_key_values

    if prefill_cache is None:
        raise RuntimeError(
            "The model did not return past_key_values."
        )

    prefill_summary = inspect_cache(
        prefill_cache,
        label="Cache after prompt prefill",
    )

    prefill_manual_bytes = print_manual_calculation(
        model_config=model.config,
        batch_size=batch_size,
        sequence_length=prompt_length,
        element_size=int(prefill_summary["element_size"]),
        program_bytes=int(prefill_summary["total_bytes"]),
    )

    # Greedily select one new token.
    next_token = prefill_outputs.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    old_attention_mask = inputs.get("attention_mask")

    if old_attention_mask is None:
        old_attention_mask = torch.ones(
            (batch_size, prompt_length),
            dtype=torch.long,
            device=device,
        )

    new_attention_mask = torch.cat(
        [
            old_attention_mask,
            torch.ones(
                (batch_size, 1),
                dtype=old_attention_mask.dtype,
                device=device,
            ),
        ],
        dim=-1,
    )

    cache_position = torch.arange(
        prompt_length,
        prompt_length + 1,
        dtype=torch.long,
        device=device,
    )

    # Decode: only pass the new, previously unprocessed token.
    with torch.inference_mode():
        decode_outputs = model(
            input_ids=next_token,
            attention_mask=new_attention_mask,
            past_key_values=prefill_cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )

    decode_cache = decode_outputs.past_key_values

    if decode_cache is None:
        raise RuntimeError(
            "The second forward pass did not return a cache."
        )

    decode_summary = inspect_cache(
        decode_cache,
        label="Cache after one additional token",
    )

    decode_manual_bytes = print_manual_calculation(
        model_config=model.config,
        batch_size=batch_size,
        sequence_length=prompt_length + 1,
        element_size=int(decode_summary["element_size"]),
        program_bytes=int(decode_summary["total_bytes"]),
    )

    # Required Day 3 checks.
    assert int(prefill_summary["layer_count"]) == int(
        model.config.num_hidden_layers
    )

    assert int(prefill_summary["sequence_length"]) == prompt_length

    assert int(decode_summary["sequence_length"]) == (
        prompt_length + 1
    )

    assert int(prefill_summary["total_bytes"]) == (
        prefill_manual_bytes
    )

    assert int(decode_summary["total_bytes"]) == (
        decode_manual_bytes
    )

    byte_increase = (
        int(decode_summary["total_bytes"])
        - int(prefill_summary["total_bytes"])
    )

    print("\n===== Growth check =====")
    print(
        "Sequence length: "
        f"{prefill_summary['sequence_length']} "
        f"→ {decode_summary['sequence_length']}"
    )
    print(
        "Cache bytes: "
        f"{prefill_summary['total_bytes']} "
        f"→ {decode_summary['total_bytes']}"
    )
    print(f"Bytes added by one token: {byte_increase}")

    print("\nALL DAY 3 KV CACHE CHECKS PASSED")

    return {
        "prompt_length": prompt_length,
        "decode_length": int(
            decode_summary["sequence_length"]
        ),
        "prefill_bytes": int(
            prefill_summary["total_bytes"]
        ),
        "decode_bytes": int(
            decode_summary["total_bytes"]
        ),
        "byte_increase": byte_increase,
    }


if __name__ == "__main__":
    run_inspection()