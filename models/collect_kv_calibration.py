from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Iterator

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_ID = "tatsu-lab/alpaca"
DATASET_REVISION = (
    "dce01c9b08f87459cf36a430d809084718273017"
)

DEFAULT_OUTPUT_PATH = Path(
    "models/qwen05b/calib_general_scales.npz"
)

QMAX = 127


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect per-layer, per-KV-head calibration statistics "
            "from Qwen prefill KV caches."
        )
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=512,
        help="Number of Alpaca examples to sample. Default: 512.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used to select examples. Default: 1234.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9,
        help=(
            "Percentile of per-sample absolute maxima used to "
            "construct the scale. Default: 99.9."
        ),
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=512,
        help="Maximum tokenized prompt length. Default: 512.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output compressed NPZ path.",
    )

    args = parser.parse_args()

    if args.sample_count <= 0:
        parser.error("--sample-count must be greater than zero.")

    if args.max_input_tokens <= 0:
        parser.error("--max-input-tokens must be greater than zero.")

    if not 0.0 < args.percentile <= 100.0:
        parser.error("--percentile must be in the interval (0, 100].")

    return args


def iter_cache_layers(
    cache: object,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    """
    Yield layer index, key tensor, and value tensor.

    This supports the current Transformers Cache representation and
    the older tuple-based past_key_values representation.
    """

    if hasattr(cache, "layers"):
        layers = getattr(cache, "layers")

        for layer_index, layer in enumerate(layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)

            if key is None or value is None:
                raise RuntimeError(
                    f"Cache layer {layer_index} is not initialized."
                )

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

    if isinstance(cache, (tuple, list)):
        for layer_index, layer in enumerate(cache):
            if not isinstance(layer, (tuple, list)):
                raise TypeError(
                    f"Unexpected cache layer type at layer "
                    f"{layer_index}: {type(layer).__name__}"
                )

            if len(layer) < 2:
                raise RuntimeError(
                    f"Cache layer {layer_index} does not contain "
                    "both Key and Value."
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


def build_user_prompt(example: dict[str, Any]) -> str:
    """
    Construct a user prompt from Alpaca instruction and input fields.

    The reference output is deliberately excluded from calibration.
    """

    instruction = str(example["instruction"]).strip()
    input_text = str(example["input"]).strip()

    if not instruction:
        raise ValueError("Alpaca example has an empty instruction.")

    if input_text:
        return (
            f"{instruction}\n\n"
            f"Additional input:\n{input_text}"
        )

    return instruction


def prepare_inputs(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_input_tokens: int,
) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(
        formatted_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_input_tokens,
    )

    return {
        name: tensor.to(device)
        for name, tensor in encoded.items()
    }


def per_head_abs_max(
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    Calculate one absolute maximum for every KV head.

    Expected tensor shape:
        [batch, KV heads, cached tokens, head dimension]

    Reduction dimensions:
        batch, cached tokens, and head dimension
    """

    if tensor.ndim != 4:
        raise RuntimeError(
            "Expected a four-dimensional KV tensor, "
            f"but received shape {tuple(tensor.shape)}."
        )

    maxima = tensor.detach().abs().amax(
        dim=(0, 2, 3)
    )

    return (
        maxima.to(torch.float32)
        .cpu()
        .numpy()
    )


def extract_cache_abs_maxima(
    cache: object,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return K and V maxima with shape:

        [number of layers, number of KV heads]
    """

    key_layer_maxima: list[np.ndarray] = []
    value_layer_maxima: list[np.ndarray] = []

    expected_heads: int | None = None

    for layer_index, key, value in iter_cache_layers(cache):
        key_maxima = per_head_abs_max(key)
        value_maxima = per_head_abs_max(value)

        if key_maxima.shape != value_maxima.shape:
            raise RuntimeError(
                f"K/V head shapes differ at layer {layer_index}: "
                f"{key_maxima.shape} and {value_maxima.shape}."
            )

        if expected_heads is None:
            expected_heads = int(key_maxima.shape[0])
        elif key_maxima.shape[0] != expected_heads:
            raise RuntimeError(
                f"Layer {layer_index} has "
                f"{key_maxima.shape[0]} KV heads; expected "
                f"{expected_heads}."
            )

        key_layer_maxima.append(key_maxima)
        value_layer_maxima.append(value_maxima)

    if not key_layer_maxima:
        raise RuntimeError("No initialized KV-cache layers were found.")

    return (
        np.stack(key_layer_maxima).astype(
            np.float32,
            copy=False,
        ),
        np.stack(value_layer_maxima).astype(
            np.float32,
            copy=False,
        ),
    )


@torch.inference_mode()
def collect_one_example(
    model: torch.nn.Module,
    tokenizer: Any,
    example: dict[str, Any],
    device: torch.device,
    max_input_tokens: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    prompt = build_user_prompt(example)

    inputs = prepare_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_input_tokens=max_input_tokens,
    )

    outputs = model(
        **inputs,
        use_cache=True,
        return_dict=True,
    )

    key_maxima, value_maxima = (
        extract_cache_abs_maxima(
            outputs.past_key_values
        )
    )

    input_token_count = int(
        inputs["input_ids"].shape[1]
    )

    return (
        key_maxima,
        value_maxima,
        input_token_count,
    )


def calculate_scales(
    abs_maxima: np.ndarray,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-sample maxima and calculate symmetric INT8 scales.

    Input shape:
        [samples, layers, KV heads]

    Output shape:
        [layers, KV heads]
    """

    percentile_values = np.percentile(
        abs_maxima,
        q=percentile,
        axis=0,
        method="linear",
    ).astype(np.float32)

    scales = np.where(
        percentile_values > 0.0,
        percentile_values / QMAX,
        1.0,
    ).astype(np.float32)

    if not np.isfinite(scales).all():
        raise RuntimeError(
            "Calculated scales contain non-finite values."
        )

    if not (scales > 0.0).all():
        raise RuntimeError(
            "Calculated scales must all be greater than zero."
        )

    return percentile_values, scales


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this program inside "
            "a Slurm GPU job."
        )

    start_time = time.perf_counter()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0")

    print("===== Calibration configuration =====", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Dataset: {DATASET_ID}", flush=True)
    print(
        f"Dataset revision: {DATASET_REVISION}",
        flush=True,
    )
    print(f"Sample count: {args.sample_count}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Percentile: {args.percentile}", flush=True)
    print(
        f"Maximum input tokens: {args.max_input_tokens}",
        flush=True,
    )
    print(f"Output: {args.output}", flush=True)

    print("\nLoading Alpaca from cache ...", flush=True)

    dataset = load_dataset(
        DATASET_ID,
        split="train",
        revision=DATASET_REVISION,
    )

    if args.sample_count > len(dataset):
        raise ValueError(
            f"Requested {args.sample_count} examples, but the "
            f"dataset contains only {len(dataset)}."
        )

    random_generator = np.random.default_rng(
        args.seed
    )

    sample_indices = random_generator.choice(
        len(dataset),
        size=args.sample_count,
        replace=False,
    ).astype(np.int64)

    print(f"Dataset rows: {len(dataset)}", flush=True)

    print("\nLoading model and tokenizer ...", flush=True)

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

    model_revision = str(
        getattr(model.config, "_commit_hash", "") or ""
    )

    key_sample_maxima: list[np.ndarray] = []
    value_sample_maxima: list[np.ndarray] = []
    input_token_counts: list[int] = []

    print("\nCollecting prefill KV statistics ...", flush=True)

    for sample_position, dataset_index in enumerate(
        sample_indices
    ):
        example = dataset[int(dataset_index)]

        key_maxima, value_maxima, token_count = (
            collect_one_example(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                max_input_tokens=args.max_input_tokens,
            )
        )

        key_sample_maxima.append(key_maxima)
        value_sample_maxima.append(value_maxima)
        input_token_counts.append(token_count)

        completed = sample_position + 1

        if completed == 1 or completed % 32 == 0:
            print(
                f"  completed {completed}/"
                f"{args.sample_count}; "
                f"dataset_index={int(dataset_index)}; "
                f"input_tokens={token_count}; "
                f"shape={key_maxima.shape}",
                flush=True,
            )

    key_abs_maxima = np.stack(
        key_sample_maxima
    ).astype(np.float32, copy=False)

    value_abs_maxima = np.stack(
        value_sample_maxima
    ).astype(np.float32, copy=False)

    input_token_counts_array = np.asarray(
        input_token_counts,
        dtype=np.int32,
    )

    key_percentile, key_scales = calculate_scales(
        key_abs_maxima,
        percentile=args.percentile,
    )

    value_percentile, value_scales = calculate_scales(
        value_abs_maxima,
        percentile=args.percentile,
    )

    number_of_layers = int(key_scales.shape[0])
    number_of_kv_heads = int(key_scales.shape[1])

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        args.output,
        key_abs_maxima=key_abs_maxima,
        value_abs_maxima=value_abs_maxima,
        key_percentile=key_percentile,
        value_percentile=value_percentile,
        key_scales=key_scales,
        value_scales=value_scales,
        sample_indices=sample_indices,
        input_token_counts=input_token_counts_array,
        model_id=np.asarray(MODEL_ID),
        model_revision=np.asarray(model_revision),
        dataset_id=np.asarray(DATASET_ID),
        dataset_revision=np.asarray(
            DATASET_REVISION
        ),
        dataset_fingerprint=np.asarray(
            str(dataset._fingerprint)
        ),
        sample_count=np.asarray(
            args.sample_count,
            dtype=np.int64,
        ),
        seed=np.asarray(
            args.seed,
            dtype=np.int64,
        ),
        percentile=np.asarray(
            args.percentile,
            dtype=np.float64,
        ),
        max_input_tokens=np.asarray(
            args.max_input_tokens,
            dtype=np.int64,
        ),
        qmax=np.asarray(
            QMAX,
            dtype=np.int16,
        ),
        number_of_layers=np.asarray(
            number_of_layers,
            dtype=np.int64,
        ),
        number_of_kv_heads=np.asarray(
            number_of_kv_heads,
            dtype=np.int64,
        ),
        cache_axis_order=np.asarray(
            "sample,layer,kv_head"
        ),
        aggregation=np.asarray(
            "per-sample absmax over batch/token/head_dim; "
            "percentile over samples"
        ),
        torch_version=np.asarray(torch.__version__),
        transformers_version=np.asarray(
            transformers.__version__
        ),
        datasets_version=np.asarray(
            datasets.__version__
        ),
        numpy_version=np.asarray(np.__version__),
    )

    elapsed_seconds = time.perf_counter() - start_time

    print("\n===== Calibration summary =====", flush=True)
    print(
        f"Raw K maxima shape: {key_abs_maxima.shape}",
        flush=True,
    )
    print(
        f"Raw V maxima shape: {value_abs_maxima.shape}",
        flush=True,
    )
    print(
        f"K scale shape: {key_scales.shape}",
        flush=True,
    )
    print(
        f"V scale shape: {value_scales.shape}",
        flush=True,
    )
    print(
        f"Input-token range: "
        f"{input_token_counts_array.min()}–"
        f"{input_token_counts_array.max()}",
        flush=True,
    )
    print(
        f"K scale range: "
        f"{key_scales.min():.8f}–"
        f"{key_scales.max():.8f}",
        flush=True,
    )
    print(
        f"V scale range: "
        f"{value_scales.min():.8f}–"
        f"{value_scales.max():.8f}",
        flush=True,
    )
    print(
        f"Elapsed: {elapsed_seconds:.2f} seconds",
        flush=True,
    )
    print(
        f"Saved calibration data to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()