#!/usr/bin/env python3
"""
Analyze boids data vs blockSize from data.txt
- Ignores the first reading in each BlockSize section
- Aggregates mean/std of fps, frame_ms, step_ms
- Saves plots in plots/
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def load_blocksize_data(path: str) -> pd.DataFrame:
    records = []
    current_block = None
    current_lines = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"BlockSize:\s*(\d+)", line)
            if m:
                # flush previous block's lines if any
                if current_block is not None and current_lines:
                    # drop first row of block
                    for row in current_lines[1:]:
                        records.append(row)
                # start new block
                current_block = int(m.group(1))
                current_lines = []
                continue

            # data line
            parts = line.split(",")
            if len(parts) >= 6 and current_block is not None:
                N = int(parts[0])
                method = parts[1]
                fps = float(parts[2])
                frame_ms = float(parts[3])
                step_ms = float(parts[4])
                copy_ms = float(parts[5])
                current_lines.append(
                    dict(
                        blockSize=current_block,
                        N=N,
                        method=method,
                        fps=fps,
                        frame_ms=frame_ms,
                        step_ms=step_ms,
                        copy_ms=copy_ms,
                    )
                )

    # flush last block
    if current_block is not None and current_lines:
        for row in current_lines[1:]:
            records.append(row)

    return pd.DataFrame(records)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("blockSize")
        .agg(
            fps_mean=("fps", "mean"),
            fps_std=("fps", "std"),
            frame_ms_mean=("frame_ms", "mean"),
            frame_ms_std=("frame_ms", "std"),
            step_ms_mean=("step_ms", "mean"),
            step_ms_std=("step_ms", "std"),
            samples=("fps", "size"),
        )
        .reset_index()
        .sort_values("blockSize")
    )
    return agg


def plot_metric(df: pd.DataFrame, mean_col, std_col, ylabel, title, out_file: Path):
    plt.figure(figsize=(6, 4))
    x = df["blockSize"]
    y = df[mean_col]
    yerr = df[std_col]

    plt.errorbar(
        x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.5,
    )
    plt.title(title)
    plt.xlabel("Block Size")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close()


def main():
    in_path = Path("data_numBlocks.txt")
    df = load_blocksize_data(str(in_path))
    agg = aggregate(df)

    print("Aggregated block size results (first values dropped):")
    print(agg.to_string(index=False))

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    plot_metric(agg, "fps_mean", "fps_std",
                "FPS", "FPS vs BlockSize", plots_dir / "blocksize_fps.png")
    plot_metric(agg, "frame_ms_mean", "frame_ms_std",
                "Frame time (ms)", "Frame Time vs BlockSize", plots_dir / "blocksize_frame.png")
    plot_metric(agg, "step_ms_mean", "step_ms_std",
                "Step time (ms)", "Step Kernel Time vs BlockSize", plots_dir / "blocksize_step.png")

    print(f"Plots saved in {plots_dir}/")


if __name__ == "__main__":
    main()