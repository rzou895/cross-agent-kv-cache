"""Minimal Qwen2.5 GPU smoke test on NeSI."""

from __future__ import annotations

import os
import platform

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def bytes_to_gib(num_bytes: int) -> float:
    """Convert bytes to GiB."""
    return num_bytes / (1024**3)


def main() -> None:
    print("===== Runtime information =====")
    print("Hostname:", platform.node())
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. "
            "Run this script through a Slurm GPU job."
        )

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    free_memory, total_memory = torch.cuda.mem_get_info(device)

    print("\n===== GPU information =====")
    print("GPU model:", properties.name)
    print(
        f"Total VRAM: "
        f"{bytes_to_gib(properties.total_memory):.2f} GiB"
    )
    print(
        f"Free VRAM before loading: "
        f"{bytes_to_gib(free_memory):.2f} GiB"
    )
    print(
        f"CUDA capability: "
        f"{properties.major}.{properties.minor}"
    )
    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported(),
    )

    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print("\n===== Model loading =====")
    print("Model:", MODEL_ID)
    print("Model dtype:", dtype)
    print(
        "Hugging Face cache:",
        os.environ.get("HF_HOME"),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()

    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    free_memory, total_memory = torch.cuda.mem_get_info(device)

    print(
        f"Allocated after loading: "
        f"{bytes_to_gib(allocated_memory):.2f} GiB"
    )
    print(
        f"Reserved after loading: "
        f"{bytes_to_gib(reserved_memory):.2f} GiB"
    )
    print(
        f"Free VRAM after loading: "
        f"{bytes_to_gib(free_memory):.2f} GiB"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a concise and helpful assistant.",
        },
        {
            "role": "user",
            "content": (
                "Explain the purpose of a KV cache "
                "in a large language model in one sentence."
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    print("\n===== Greedy generation =====")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_token_ids = output_ids[:, prompt_length:]

    response = tokenizer.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
    )[0]

    print("Prompt:")
    print(messages[-1]["content"])

    print("\nGenerated response:")
    print(response)

    print(
        "\nGenerated token count:",
        new_token_ids.shape[1],
    )

    peak_memory = torch.cuda.max_memory_allocated(device)

    print(
        "Peak allocated VRAM:",
        f"{bytes_to_gib(peak_memory):.2f} GiB",
    )

    print("\nQwen GPU smoke test passed.")


if __name__ == "__main__":
    main()