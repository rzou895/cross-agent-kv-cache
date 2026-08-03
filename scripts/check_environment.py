import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import torch


PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
]


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "Not installed"


def bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def main() -> None:
    print("=== System ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")

    print("\n=== Python packages ===")
    for package in PACKAGES:
        print(f"{package}: {package_version(package)}")

    print("\n=== CUDA and device ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        free_memory, total_memory = torch.cuda.mem_get_info(device_index)

        print(f"GPU index: {device_index}")
        print(f"GPU name: {torch.cuda.get_device_name(device_index)}")
        print(
            f"GPU compute capability: "
            f"{properties.major}.{properties.minor}"
        )
        print(
            f"Total GPU memory: "
            f"{bytes_to_gib(total_memory):.2f} GiB"
        )
        print(
            f"Currently free GPU memory: "
            f"{bytes_to_gib(free_memory):.2f} GiB"
        )
    else:
        print("GPU name: N/A")
        print("Total GPU memory: N/A")
        print("Execution device: CPU")


if __name__ == "__main__":
    main()