from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = Path("results/week01/single_model_fp16.jsonl")
RUNS = 22
WARMUP_RUNS = 2
MAX_NEW_TOKENS = 64

# Do not change these after the first official run. Later experiments should use
# exactly the same prompts so the results are comparable.
PROMPTS = [
    "Explain what a KV cache is in a transformer language model.",
    "Why can reusing a KV cache reduce autoregressive decoding latency?",
    "Describe the difference between prefill and decode in language-model inference.",
    "Give a simple explanation of symmetric INT8 quantization.",
    "What is the purpose of a calibration dataset in static quantization?",
    "Write a Python function that returns the larger of two integers.",
    "List three possible sources of measurement error in a GPU benchmark.",
    "Why can two agents safely share a KV cache only for an identical token prefix?",
    "What is the difference between mean latency and p95 latency?",
    "Describe an experiment comparing FP16 and INT8 KV-cache accuracy.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 prompt, 3 repetitions, 1 warm-up and 8 output tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed. Default: 1234.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path.",
    )

    return parser.parse_args()


def percentile(values: list[float], percent: float) -> float:
    """Calculate a linearly interpolated percentile without NumPy."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    weight = position - lower
    return float(
        ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    )


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 95.0),
        # Sample standard deviation. Return 0 only for the small smoke test.
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def prepare_inputs(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


@torch.inference_mode()
def run_once(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    baseline_memory_bytes: int,
) -> dict[str, Any]:
    inputs = prepare_inputs(tokenizer, prompt, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated_tokens: list[torch.Tensor] = []

    # Start a new peak-memory measurement for this repetition.
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # ---------- Prefill ----------
    # The model processes the complete prompt and predicts the first new token.
    torch.cuda.synchronize(device)
    start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    next_token = outputs.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - start) * 1000.0

    cache = outputs.past_key_values
    generated_tokens.append(next_token)

    # ---------- Decode ----------
    # The first new token came from prefill, so decode produces the remaining
    # max_new_tokens - 1 tokens, one token at a time, while reusing the cache.
    decode_tokens = max_new_tokens - 1

    torch.cuda.synchronize(device)
    start = time.perf_counter()

    for _ in range(decode_tokens):
        attention_mask = torch.cat(
            [
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1)),
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
        generated_tokens.append(next_token)

    torch.cuda.synchronize(device)
    decode_total_ms = (time.perf_counter() - start) * 1000.0

    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    output_ids = torch.cat(generated_tokens, dim=1)[0].tolist()

    return {
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": len(output_ids),
        "prefill_ms": prefill_ms,
        "decode_total_ms": decode_total_ms,
        "decode_ms_per_token": decode_total_ms / decode_tokens,
        "peak_gpu_memory_mb": peak_memory_bytes / (1024**2),
        "peak_gpu_memory_increment_mb": (
            peak_memory_bytes - baseline_memory_bytes
        )
        / (1024**2),
        "generated_text": tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ),
    }


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this inside a Slurm GPU job, not on the login node."
        )

    if args.smoke:
        prompts = PROMPTS[:1]
        runs = 3
        warmup_runs = 1
        max_new_tokens = 8
        default_output_path = Path(
            "results/week01/single_model_fp16_smoke.jsonl"
        )
    else:
        prompts = PROMPTS
        runs = RUNS
        warmup_runs = WARMUP_RUNS
        max_new_tokens = MAX_NEW_TOKENS
        default_output_path = OUTPUT_PATH

    output_path = (
        args.output
        if args.output is not None
        else default_output_path
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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

    torch.cuda.synchronize(device)
    baseline_memory_bytes = int(torch.cuda.memory_allocated(device))
    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for prompt_id, prompt in enumerate(prompts):
            print(
                f"\nPrompt {prompt_id + 1}/{len(prompts)}: {prompt}",
                flush=True,
            )

            measured: list[dict[str, Any]] = []

            for run_id in range(runs):
                result = run_once(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    baseline_memory_bytes=baseline_memory_bytes,
                )

                if run_id < warmup_runs:
                    print(
                        f"  run {run_id + 1}/{runs}: warm-up discarded",
                        flush=True,
                    )
                    continue

                measured.append(result)
                print(
                    f"  run {run_id + 1}/{runs}: "
                    f"prefill={result['prefill_ms']:.3f} ms, "
                    f"decode={result['decode_ms_per_token']:.3f} ms/token, "
                    f"peak={result['peak_gpu_memory_mb']:.1f} MiB",
                    flush=True,
                )

            metrics = [
                "prefill_ms",
                "decode_total_ms",
                "decode_ms_per_token",
                "peak_gpu_memory_mb",
                "peak_gpu_memory_increment_mb",
            ]

            record = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "model": MODEL_ID,
                "dtype": "float16",
                "seed": args.seed,
                "gpu": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "runs_total": runs,
                "warmup_runs": warmup_runs,
                "measured_runs": len(measured),
                "input_tokens": measured[0]["input_tokens"],
                "output_tokens": measured[0]["output_tokens"],
                "model_parameter_memory_mb": model_parameter_bytes / (1024**2),
                "baseline_gpu_memory_mb": baseline_memory_bytes / (1024**2),
                "statistics": {
                    metric: summarize(
                        [float(run[metric]) for run in measured]
                    )
                    for metric in metrics
                },
                "raw_measurements": {
                    metric: [float(run[metric]) for run in measured]
                    for metric in metrics
                },
                "generated_text": measured[0]["generated_text"],
            }

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            output_file.flush()

    print(f"\nSaved benchmark results to {output_path}", flush=True)


if __name__ == "__main__":
    main()