from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


INPUT_PATH = Path(
    "models/qwen05b/calib_general_scales.npz"
)

OUTPUT_PATH = Path(
    "figures/week02/calib_general_scales_heatmap.pdf"
)


def add_heatmap(
    pdf: PdfPages,
    values: np.ndarray,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 8))

    image = axis.imshow(
        values,
        aspect="auto",
    )

    axis.set_title(title)
    axis.set_xlabel("KV head")
    axis.set_ylabel("Transformer layer")

    axis.set_xticks(
        range(values.shape[1])
    )
    axis.set_yticks(
        range(values.shape[0])
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )
    colorbar.set_label("Symmetric INT8 scale")

    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    with np.load(
        INPUT_PATH,
        allow_pickle=False,
    ) as data:
        key_scales = data["key_scales"]
        value_scales = data["value_scales"]

    if key_scales.shape != (24, 2):
        raise RuntimeError(
            f"Unexpected K scale shape: {key_scales.shape}"
        )

    if value_scales.shape != (24, 2):
        raise RuntimeError(
            f"Unexpected V scale shape: {value_scales.shape}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PdfPages(OUTPUT_PATH) as pdf:
        add_heatmap(
            pdf,
            key_scales,
            "Qwen2.5-0.5B K calibration scales",
        )
        add_heatmap(
            pdf,
            value_scales,
            "Qwen2.5-0.5B V calibration scales",
        )

    print(
        f"Saved heatmaps to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()