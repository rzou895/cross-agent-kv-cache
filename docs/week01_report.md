# Week 1 Sanity Report: Single-Model FP16 Baseline

## Experimental setup

The baseline experiment used `Qwen/Qwen2.5-0.5B-Instruct` in FP16 on an
NVIDIA A100-PCIE-40GB GPU. The software environment used PyTorch
2.13.0+cu126 and Transformers 5.14.1.

The benchmark contained 10 fixed prompts. For each prompt, the model
generated 64 output tokens using greedy decoding. Each experiment used
22 runs per prompt, with the first 2 runs discarded as warm-up runs.
Therefore, each experiment contained 200 measured runs in total.

Three experiments were compared:

| Experiment | Seed | NeSI node |
|---|---:|---|
| Day 4 original | 1234 | g04 |
| Day 5 reproduction | 1234 | g03 |
| Day 5 different-seed run | 5678 | g02 |

## Reproducibility

All three experiments used identical prompts, input-token counts and
output-token counts. The generated text was also identical for all 10
prompts.

Changing the seed did not affect the generated output because the model
used greedy decoding with `argmax`, rather than random sampling.

The measured latency results were:

| Experiment | Prefill latency | Decode latency |
|---|---:|---:|
| Day 4 original | 12.7607 ± 1.6617 ms | 11.1460 ± 0.5619 ms/token |
| Reproduction, seed 1234 | 11.4822 ± 0.0641 ms | 10.0846 ± 0.0184 ms/token |
| Repeated run, seed 5678 | 11.3945 ± 0.0772 ms | 10.0227 ± 0.0225 ms/token |

The two Day 5 runs differed by less than 1% in mean latency, showing that
the current benchmark procedure is stable. However, the original Day 4
run was approximately 10% slower and had substantially higher variance.

Therefore, functional reproducibility was achieved because the inputs,
token counts and generated outputs were identical. Exact latency
reproducibility across separate Slurm jobs was not achieved. The
difference may be related to physical-node variation, GPU clock or power
state, temperature, or other system-level effects.

The latency distributions are shown in
[`figures/week01_latency.pdf`](../figures/week01_latency.pdf).

## GPU memory

The FP16 model parameters occupied **942.293 MiB**.

The allocated GPU memory immediately after loading the model was
**950.167 MiB**. During generation, the mean peak allocated GPU memory
was **971.422 MiB**, corresponding to an increase of **21.255 MiB** above
the post-loading baseline.

The peak-memory increase is larger than the KV-cache size because it also
includes logits, temporary tensors, attention computation, and other
PyTorch or CUDA allocations.

## KV-cache size

The FP16 KV cache used **12,288 bytes**, or **12 KiB**, per cached token.

This matches the theoretical calculation:

```text
24 layers
× 2 for Key and Value
× 2 KV heads
× 64 dimensions per head
× 2 bytes for FP16
= 12,288 bytes per cached token
```

The input prompts contained between 40 and 45 tokens, with a mean of
42.1 tokens. After prefill, the KV-cache size was:

- mean: **0.493359 MiB**
- range: **0.468750–0.527344 MiB**

At the end of generation, the cache contained between 103 and 108
tokens, with a mean of 105.1 cached tokens. The final KV-cache size was:

- mean: **1.231641 MiB**
- range: **1.207031–1.265625 MiB**

The final predicted token was not included in the cache because it had
not yet been passed back into the model as an input token.

## Sources of measurement error

Possible sources of measurement error include different physical compute
nodes, GPU clock and power states, GPU temperature, CUDA kernel warm-up,
memory allocation and caching, Python overhead, operating-system noise,
and other cluster-level variation.

CUDA synchronization was performed before and after the timed regions,
and warm-up runs were discarded. These controls reduce timing error, but
they cannot remove all variation between independently scheduled Slurm
jobs.

## Conclusion

The single-model FP16 baseline is functionally reproducible. Identical
prompts produced identical token counts and generated text under both
random seeds.

The two Day 5 runs established a stable latency baseline of approximately
11.4–11.5 ms for prefill and 10.0–10.1 ms per decode token. The original
Day 4 run was approximately 10% slower and more variable, showing that
absolute latency can vary across independently scheduled GPU jobs.

This baseline will be used for later comparisons with INT8 KV-cache
quantization and cross-agent KV-cache sharing.
