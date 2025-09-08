#!/usr/bin/env python3
"""
Compare multiple distance cell configs.

Input format:
  <Section Header>
  N,method,fps,avg_frame_ms,avg_step_ms,avg_copy_ms
  ...
  <Another Section Header>
  ...

- Ignores the first row from each (label, method) group
- Aggregates mean/std
- Plots FPS, frame time, step time comparisons
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def load_sectioned_data(path: str) -> pd.DataFrame:
    records = []
    current_label = None
    for line in open(path, "r"):
        line = line.strip()
        if not line:
            continue
        # detect section like "2-Distance Cell"
        if re.match(r"\d+-Distance", line):
            current_label = line
            continue
        # otherwise row
        parts = line.split(",")
        if len(parts) >= 6 and current_label is not None:
            N = int(parts[0])
            method = parts[1]
            fps = float(parts[2])
            frame_ms = float(parts[3])
            step_ms = float(parts[4])
            records.append(
                dict(
                    label=current_label,
                    N=N,
                    method=method,
                    fps=fps,
                    frame_ms=frame_ms,
                    step_ms=step_ms,
                )
            )
    return pd.DataFrame(records)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # drop first row in each (label, method)
    df = (
        df.sort_values(by=["label", "method"])
        .groupby(["label", "method"], group_keys=False)
        .apply(lambda g: g.iloc[1:])
        .reset_index(drop=True)
    )

    agg = (
        df.groupby(["label", "method"])
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
    )
    return agg


def plot_comparison(agg: pd.DataFrame, metric: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(7, 5))
    methods = agg["method"].unique()
    labels = agg["label"].unique()

    # grouped bar chart
    x = range(len(labels))
    w = 0.25
    for i, method in enumerate(methods):
        sub = agg[agg["method"] == method]
        means = []
        stds = []
        for lbl in labels:
            row = sub[sub["label"] == lbl]
            if not row.empty:
                means.append(row[metric + "_mean"].values[0])
                stds.append(row[metric + "_std"].values[0])
            else:
                means.append(0)
                stds.append(0)
        xpos = [xx + i * w for xx in x]
        plt.bar(
            xpos,
            means,
            width=w,
            yerr=stds,
            capsize=3,
            label=method,
        )

    plt.xticks([xx + (w * (len(methods) - 1) / 2) for xx in x], labels, rotation=20)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} comparison across methods")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    in_path = Path("../data/data3celldist.txt")
    df = load_sectioned_data(str(in_path))
    agg = aggregate(df)

    print("Aggregated results:")
    print(agg.to_string(index=False))

    plots_dir = Path("plots")
    plot_comparison(agg, "fps", "FPS", plots_dir / "compare_fps.png")
    plot_comparison(agg, "frame_ms", "Frame time (ms)", plots_dir / "compare_frame.png")
    plot_comparison(agg, "step_ms", "Step time (ms)", plots_dir / "compare_step.png")
    print(f"Plots saved in {plots_dir}/")


if __name__ == "__main__":
    main()