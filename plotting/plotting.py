#!/usr/bin/env python3
"""
Aggregate and plot Boids perf logs (no visualization mode).
- Ignores the first sample for each (N, method) group
- Produces aggregated.csv with mean/std values
- Saves separate plots (fps.png, frame_time.png, step_time.png) in plots/
- Ignores copy_ms since VISUALIZE=0
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", engine="python")

    # drop stray header rows in data
    df["N_num"] = pd.to_numeric(df["N"], errors="coerce")
    df = df[df["N_num"].notna()].copy()
    df["N"] = df["N_num"].astype(int)
    df.drop(columns=["N_num"], inplace=True)

    # ensure numeric
    for col in ["fps", "avg_frame_ms", "avg_step_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["fps", "avg_frame_ms", "avg_step_ms"]).reset_index(
        drop=True
    )


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # drop first reading per group (warm-up)
    df = (
        df.sort_values(by=["N", "method"])
        .groupby(["N", "method"], group_keys=False)
        .apply(lambda g: g.iloc[1:])
        .reset_index(drop=True)
    )

    agg = (
        df.groupby(["N", "method"])
        .agg(
            fps_mean=("fps", "mean"),
            fps_std=("fps", "std"),
            frame_ms_mean=("avg_frame_ms", "mean"),
            frame_ms_std=("avg_frame_ms", "std"),
            step_ms_mean=("avg_step_ms", "mean"),
            step_ms_std=("avg_step_ms", "std"),
            samples=("fps", "size"),
        )
        .reset_index()
        .sort_values(by=["method", "N"])
    )
    return agg



def plot_metric(
    agg: pd.DataFrame, mean_col: str, std_col: str,
    y_label: str, title: str, out_path: Path
):
    plt.figure(figsize=(6, 4))
    methods = agg["method"].unique()
    colors = plt.cm.tab10.colors

    for i, m in enumerate(methods):
        sub = agg[agg["method"] == m].sort_values("N")
        if sub.empty:
            continue
        x = sub["N"].to_numpy()
        y = sub[mean_col].to_numpy()
        yerr = sub[std_col].to_numpy()
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            label=m,
            color=colors[i % len(colors)],
            marker="o",
            capsize=3,
            linewidth=1.5,
        )

    plt.title(title)
    plt.xlabel("N (particles, log scale)")
    plt.ylabel(y_label)
    plt.xscale("log")                      # <-- log scale for particle counts
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data.csv")
    if not in_path.exists():
        print(f"File not found: {in_path}")
        sys.exit(1)

    df = load_data(str(in_path))
    agg = aggregate(df)

    # Save aggregated CSV
    out_csv = in_path.with_name("aggregated.csv")
    agg.to_csv(out_csv, index=False)
    print(f"Aggregated results saved to {out_csv}")
    print(agg.to_string(index=False))

    # Plots directory
    plots_dir = in_path.parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Only plot fps, frame time, and step time
    metrics = [
        ("fps_mean", "fps_std", "FPS", "Frames per second", "fps.png"),
        ("frame_ms_mean", "frame_ms_std", "Frame time (ms)", "Whole-frame time", "frame_time.png"),
        ("step_ms_mean", "step_ms_std", "Step time (ms)", "Boids step kernel time", "step_time.png"),
    ]

    for mean_col, std_col, y_label, title, fname in metrics:
        out_path = plots_dir / fname
        plot_metric(agg, mean_col, std_col, y_label, title, out_path)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()