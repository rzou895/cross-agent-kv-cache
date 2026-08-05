from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DATASETS = {
    "day4": {
        "label": "Day 4 original\nseed 1234",
        "path": Path("results/week01/single_model_fp16.jsonl"),
    },
    "reproduce": {
        "label": "Reproduction\nseed 1234",
        "path": Path(
            "results/week01/single_model_fp16_reproduce.jsonl"
        ),
    },
    "seed5678": {
        "label": "Repeated run\nseed 5678",
        "path": Path(
            "results/week01/single_model_fp16_seed_5678.jsonl"
        ),
    },
}

OUTPUT_PATH = Path("figures/week01_latency.pdf")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {path}")

    records = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from error

    return sorted(records, key=lambda record: record["prompt_id"])


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    weight = position - lower
    return float(
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 95.0),
        "std": (
            float(statistics.stdev(values))
            if len(values) > 1
            else 0.0
        ),
    }


def collect_measurements(
    records: list[dict[str, Any]],
    metric: str,
) -> list[float]:
    values = []

    for record in records:
        values.extend(
            float(value)
            for value in record["raw_measurements"][metric]
        )

    return values


def validate_records(
    loaded: dict[str, list[dict[str, Any]]],
) -> None:
    reference_prompt_ids = [
        record["prompt_id"] for record in loaded["day4"]
    ]

    for name, records in loaded.items():
        prompt_ids = [record["prompt_id"] for record in records]

        if prompt_ids != reference_prompt_ids:
            raise RuntimeError(
                f"Prompt IDs in {name} do not match Day 4."
            )

        if len(records) != 10:
            raise RuntimeError(
                f"Expected 10 prompts in {name}, found {len(records)}."
            )

        for record in records:
            if record["measured_runs"] != 20:
                raise RuntimeError(
                    f"{name}, prompt {record['prompt_id']} has "
                    f"{record['measured_runs']} measured runs, not 20."
                )


def compare_generated_text(
    loaded: dict[str, list[dict[str, Any]]],
) -> None:
    all_identical = True

    for prompt_index in range(len(loaded["day4"])):
        texts = {
            name: records[prompt_index]["generated_text"]
            for name, records in loaded.items()
        }

        identical = len(set(texts.values())) == 1
        all_identical = all_identical and identical

        print(
            f"Prompt {prompt_index}: "
            f"generated text identical = {identical}"
        )

    print(f"\nAll generated texts identical: {all_identical}")


def print_metric_summary(
    loaded: dict[str, list[dict[str, Any]]],
    metric: str,
    unit: str,
) -> None:
    summaries = {}

    print(f"\n===== {metric} =====")

    for name, dataset in DATASETS.items():
        values = collect_measurements(loaded[name], metric)
        summary = summarize(values)
        summaries[name] = summary

        print(
            f"{dataset['label'].replace(chr(10), ' '):30s} "
            f"n={int(summary['count']):3d}, "
            f"mean={summary['mean']:.4f} {unit}, "
            f"median={summary['median']:.4f}, "
            f"p95={summary['p95']:.4f}, "
            f"std={summary['std']:.4f}"
        )

    original_mean = summaries["day4"]["mean"]

    reproduce_difference = (
        summaries["reproduce"]["mean"] - original_mean
    ) / original_mean * 100.0

    new_seed_difference = (
        summaries["seed5678"]["mean"] - original_mean
    ) / original_mean * 100.0

    print(
        "Reproduction mean difference from Day 4: "
        f"{reproduce_difference:+.2f}%"
    )
    print(
        "Seed 5678 mean difference from Day 4: "
        f"{new_seed_difference:+.2f}%"
    )


def create_plot(
    pdf: PdfPages,
    loaded: dict[str, list[dict[str, Any]]],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    labels = [
        dataset["label"]
        for dataset in DATASETS.values()
    ]

    values = [
        collect_measurements(loaded[name], metric)
        for name in DATASETS
    ]

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.boxplot(
        values,
        labels=labels,
        showfliers=True,
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def main() -> None:
    loaded = {
        name: load_jsonl(dataset["path"])
        for name, dataset in DATASETS.items()
    }

    validate_records(loaded)

    print("All three files contain 10 prompts and 20 measurements per prompt.")
    compare_generated_text(loaded)

    print_metric_summary(
        loaded,
        metric="prefill_ms",
        unit="ms",
    )
    print_metric_summary(
        loaded,
        metric="decode_ms_per_token",
        unit="ms/token",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUTPUT_PATH) as pdf:
        create_plot(
            pdf,
            loaded,
            metric="prefill_ms",
            title="Prefill latency distribution",
            ylabel="Prefill latency (ms)",
        )
        create_plot(
            pdf,
            loaded,
            metric="decode_ms_per_token",
            title="Decode latency distribution",
            ylabel="Decode latency (ms/token)",
        )

    print(f"\nSaved latency figures to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
