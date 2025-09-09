import json
from pathlib import Path
from typing import Any
from collections.abc import Iterator
from matplotlib import pyplot as plt

MEASUREMENTS_JSON = Path(__file__).parent.absolute() / "measurements.json"
ROOT_DIR = Path(__file__).parent.parent.absolute()
IMAGES_DIR = ROOT_DIR / "images"

with MEASUREMENTS_JSON.open(encoding="utf-8") as file:
    MEASUREMENT_DATA: list[dict[str, Any]] = json.load(file)


def find_measurements(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for measurement in MEASUREMENT_DATA:
        if all(
            key in measurement and measurement[key] == value
            for key, value in config.items()
        ):
            yield measurement


def plot_fps_vs_num_boids() -> None:
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of boids")
    ax.set_ylabel("Frames per second")

    for label, base_config in [
        ("Naive (w/o vis.)", {"UNIFORM_GRID": 0, "COHERENT_GRID": 0, "VISUALIZE": 0}),
        ("Naive (w/ vis.)", {"UNIFORM_GRID": 0, "COHERENT_GRID": 0, "VISUALIZE": 1}),
        (
            "Scattered Uniform Grid (w/o vis.)",
            {"UNIFORM_GRID": 1, "COHERENT_GRID": 0, "VISUALIZE": 0},
        ),
        (
            "Scattered Uniform Grid (w/ vis.)",
            {"UNIFORM_GRID": 1, "COHERENT_GRID": 0, "VISUALIZE": 1},
        ),
        (
            "Coherent Uniform Grid (w/o vis.)",
            {"UNIFORM_GRID": 1, "COHERENT_GRID": 1, "VISUALIZE": 0},
        ),
        (
            "Coherent Uniform Grid (w/ vis.)",
            {"UNIFORM_GRID": 1, "COHERENT_GRID": 1, "VISUALIZE": 1},
        ),
    ]:
        measurements = list(
            find_measurements(
                {**base_config, "FINE_GRAINED_CELLS": 1, "CUDA_BLOCK_SIZE": 128}
            )
        )
        measurements.sort(key=lambda m: m["N_FOR_VIS"])
        nums_boids = [m["N_FOR_VIS"] for m in measurements]
        fps = [m["fps"] for m in measurements]
        ax.plot(nums_boids, fps, marker="o", label=label)

    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "Frame rate vs number of boids.png")


def plot_fps_vs_block_size() -> None:
    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[-1].set_xlabel("Block size")

    for ax, (title, base_config) in zip(
        axes,
        [
            ("Naive", {"UNIFORM_GRID": 0, "COHERENT_GRID": 0}),
            ("Scattered Uniform Grid", {"UNIFORM_GRID": 1, "COHERENT_GRID": 0}),
            ("Coherent Uniform Grid", {"UNIFORM_GRID": 1, "COHERENT_GRID": 1}),
        ],
    ):
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_ylabel("FPS")

        measurements = list(
            find_measurements(
                {
                    **base_config,
                    "N_FOR_VIS": 320000,
                    "VISUALIZE": 0,
                    "FINE_GRAINED_CELLS": 1,
                }
            )
        )
        measurements.sort(key=lambda m: m["CUDA_BLOCK_SIZE"])
        block_sizes = [m["CUDA_BLOCK_SIZE"] for m in measurements]
        fps = [m["fps"] for m in measurements]
        ax.plot(block_sizes, fps, marker="o")

    fig.align_labels()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "Frame rate vs block size.png")


def plot_fps_vs_fine_grained_cells() -> None:
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of boids")
    ax.set_ylabel("Frames per second")

    for fine_grained, label in [(0, "2x-sized cells"), (1, "1x-sized cells")]:
        measurements = list(
            find_measurements(
                {
                    "VISUALIZE": 0,
                    "FINE_GRAINED_CELLS": fine_grained,
                    "UNIFORM_GRID": 1,
                    "COHERENT_GRID": 1,
                    "CUDA_BLOCK_SIZE": 128,
                }
            )
        )
        measurements.sort(key=lambda m: m["N_FOR_VIS"])
        nums_boids = [m["N_FOR_VIS"] for m in measurements]
        fps = [m["fps"] for m in measurements]
        ax.plot(nums_boids, fps, marker="o", label=label)

    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "Frame rate vs fine-grained cells.png")


if __name__ == "__main__":
    plot_fps_vs_num_boids()
    plot_fps_vs_block_size()
    plot_fps_vs_fine_grained_cells()
