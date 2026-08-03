import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.environ.get(
    "MODEL_ID",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": (
            "Briefly explain what a KV cache is in a transformer "
            "and why it is useful."
        ),
    },
]


def main() -> None:
    # 自动判断使用 GPU 还是 CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # GPU 优先使用 BF16；CPU 使用 FP32
    if device.type == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    else:
        dtype = torch.float32

    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Data type: {dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            local_files_only=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            local_files_only=True,
        )
    except OSError as error:
        raise RuntimeError(
            "Cannot find the downloaded model locally. "
            "Check MODEL_ID or change it to the local model folder."
        ) from error

    model.to(device)
    model.eval()

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    generated_token_count = generated_ids.shape[0]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    tokens_per_second = (
        generated_token_count / elapsed_time
        if elapsed_time > 0
        else 0.0
    )

    print("\n=== Generated text ===")
    print(generated_text)

    print("\n=== Generation statistics ===")
    print(f"Input token count: {input_length}")
    print(f"Generated token count: {generated_token_count}")
    print(f"Elapsed time: {elapsed_time:.3f} seconds")
    print(f"Generation speed: {tokens_per_second:.2f} tokens/second")

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated()
        print(
            f"Peak allocated GPU memory: "
            f"{peak_memory / 1024**3:.3f} GiB"
        )
    else:
        print("Peak allocated GPU memory: N/A (CPU execution)")


if __name__ == "__main__":
    main()