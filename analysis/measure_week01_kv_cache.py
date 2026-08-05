from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.benchmark_single_model import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    PROMPTS,
    prepare_inputs,
)


OUTPUT_PATH = Path("results/week01/kv_cache_fp16.jsonl")


def iter_cache_layers(
    cache: object,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    """Yield the key and value tensors stored in each cache layer."""

    # Current Transformers Cache / DynamicCache representation.
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

    # Fallback for older tuple-based past_key_values.
    if isinstance(cache, (tuple, list)):
        for layer_index, layer in enumerate(cache):
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise RuntimeError(
                    f"Unexpected cache layer format at layer "
                    f"{layer_index}."
                )

            key = layer[0]
            value = layer[1]

            if not isinstance(key, torch.Tensor):
                raise TypeError(
                    f"Layer {layer_index} key is not a tensor."
                )

            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Layer {layer_index} value is not a tensor."
                )

            yield layer_index, key, value

        return

    raise TypeError(
        f"Unsupported cache type: {type(cache).__name__}"
    )


def describe_cache(cache: object) -> dict[str, object]:
    layers = list(iter_cache_layers(cache))

    if not layers:
        raise RuntimeError("The KV cache contains no layers.")

    total_bytes = sum(
        key.numel() * key.element_size()
        + value.numel() * value.element_size()
        for _, key, value in layers
    )

    _, first_key, first_value = layers[0]

    return {
        "bytes": int(total_bytes),
        "mb": float(total_bytes / (1024**2)),
        "num_layers": len(layers),
        "key_dtype": str(first_key.dtype),
        "value_dtype": str(first_value.dtype),
        "first_key_shape": list(first_key.shape),
        "first_value_shape": list(first_value.shape),
    }


@torch.inference_mode()
def measure_prompt(
    model: torch.nn.Module,
    tokenizer: object,
    prompt_id: int,
    prompt: str,
    device: torch.device,
) -> dict[str, object]:
    inputs = prepare_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Prefill processes the full input prompt.
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    cache = outputs.past_key_values
    prefill_cache = describe_cache(cache)

    next_token = outputs.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    # Prefill predicts the first output token. The decode loop processes
    # that token and then the following output tokens one at a time.
    decode_tokens = MAX_NEW_TOKENS - 1

    for _ in range(decode_tokens):
        attention_mask = torch.cat(
            [
                attention_mask,
                attention_mask.new_ones(
                    (attention_mask.shape[0], 1)
                ),
            ],
            dim=1,
        )

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        cache = outputs.past_key_values

        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

    final_cache = describe_cache(cache)

    input_tokens = int(input_ids.shape[1])

    # The final predicted token has not yet been passed into the model,
    # so it is not stored in the cache.
    final_cached_tokens = input_tokens + decode_tokens

    bytes_per_cached_token = (
        int(final_cache["bytes"]) / final_cached_tokens
    )

    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "model": MODEL_ID,
        "dtype": "float16",
        "gpu": torch.cuda.get_device_name(device),
        "input_tokens": input_tokens,
        "requested_output_tokens": MAX_NEW_TOKENS,
        "prefill_cached_tokens": input_tokens,
        "final_cached_tokens": final_cached_tokens,
        "prefill_kv_cache_bytes": prefill_cache["bytes"],
        "prefill_kv_cache_mb": prefill_cache["mb"],
        "final_kv_cache_bytes": final_cache["bytes"],
        "final_kv_cache_mb": final_cache["mb"],
        "kv_cache_bytes_per_cached_token": (
            bytes_per_cached_token
        ),
        "num_cache_layers": final_cache["num_layers"],
        "key_dtype": final_cache["key_dtype"],
        "value_dtype": final_cache["value_dtype"],
        "first_key_shape": final_cache["first_key_shape"],
        "first_value_shape": final_cache["first_value_shape"],
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this inside a Slurm GPU job."
        )

    device = torch.device("cuda:0")

    print(f"Loading {MODEL_ID} ...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        local_files_only=True,
    ).to(device)

    model.eval()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for prompt_id, prompt in enumerate(PROMPTS):
            result = measure_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt_id=prompt_id,
                prompt=prompt,
                device=device,
            )

            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            print(
                f"Prompt {prompt_id}: "
                f"input={result['input_tokens']} tokens, "
                f"prefill cache="
                f"{result['prefill_kv_cache_mb']:.4f} MiB, "
                f"final cache="
                f"{result['final_kv_cache_mb']:.4f} MiB",
                flush=True,
            )

    print(f"Saved KV cache results to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
