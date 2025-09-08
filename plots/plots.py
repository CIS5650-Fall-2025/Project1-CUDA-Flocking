from pathlib import Path
from matplotlib import pyplot as plt


def plot_frame_rate_vs_number_of_boids() -> None:
    numbers_of_boids = [
        5000,
        10000,
        20000,
        40000,
        80000,
        160000,
        320000,
        640000,
        1280000,
        2560000,
        5120000,
        10240000,
        20480000,
    ]
    frame_rates_naive_no_vis = [
        1057.18,
        578.73,
        303.208,
        156.814,
        80.9807,
        33.699,
        18.9407,
        12.8056,
        10.4033,
        None,
        None,
        None,
        None,
    ]
    frame_rates_naive_with_vis = [
        669.401,
        430.536,
        246.626,
        133.908,
        66.5314,
        21.914,
        7.64789,
        2.02074,
        0.521029,
        None,
        None,
        None,
        None,
    ]
    frame_rates_scattered_no_vis = [
        954.948,
        606.269,
        450.065,
        547.381,
        754.17,
        298.219,
        105.776,
        28.1882,
        7.32915,
        1.88598,
        0.463071,
        None,
        None,
    ]
    frame_rates_scattered_with_vis = [
        637.855,
        448.857,
        378.916,
        420.64,
        500.578,
        250.635,
        96.8395,
        27.1921,
        7.16215,
        1.81145,
        0.409476,
        None,
        None,
    ]
    frame_rates_coherent_no_vis = [
        1136.49,
        852.869,
        761.894,
        1126.77,
        1533.94,
        1562.2,
        1180.88,
        505.753,
        158.943,
        42.5598,
        10.9749,
        2.81504,
        0.744258,
    ]
    frame_rates_coherent_with_vis = [
        682.369,
        567.523,
        557.11,
        715.706,
        844.657,
        856.094,
        666.505,
        364.128,
        137.602,
        40.4236,
        10.6321,
        2.72167,
        0.682546,
    ]

    fig, ax = plt.subplots()
    for label, data in [
        ("Naive (w/o vis.)", frame_rates_naive_no_vis),
        ("Naive (w/ vis.)", frame_rates_naive_with_vis),
        ("Scattered grid (w/o vis.)", frame_rates_scattered_no_vis),
        ("Scattered grid (w/ vis.)", frame_rates_scattered_with_vis),
        ("Coherent grid (w/o vis.)", frame_rates_coherent_no_vis),
        ("Coherent grid (w/ vis.)", frame_rates_coherent_with_vis),
    ]:
        ax.plot(numbers_of_boids, data, marker="o", label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of boids")
    ax.set_ylabel("Frames per second")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        Path(__file__).parent.parent / "images/Frame rate vs number of boids.png"
    )


def plot_frame_rate_vs_block_size() -> None:
    block_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    frame_rates_naive = [
        12.7485,
        14.7857,
        17.6827,
        18.9289,
        18.9404,
        18.9422,
        19.0132,
        17.8544,
    ]
    frame_rates_scattered = [
        158.967,
        129.124,
        105.249,
        105.831,
        105.803,
        108.078,
        108.38,
        108.489,
    ]
    frame_rates_coherent = [
        453.725,
        743.291,
        1068.75,
        1176.1,
        1181.44,
        1200.36,
        1096.78,
        1110.18,
    ]

    fig, ax = plt.subplots()
    for label, data in [
        ("Naive", frame_rates_naive),
        ("Scattered grid", frame_rates_scattered),
        ("Coherent grid", frame_rates_coherent),
    ]:
        ax.plot(block_sizes, data, marker="o", label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Block size")
    ax.set_ylabel("Frames per second")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).parent.parent / "images/Frame rate vs block size.png")


if __name__ == "__main__":
    plot_frame_rate_vs_number_of_boids()
    plot_frame_rate_vs_block_size()
