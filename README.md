# University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 1 - Flocking

- Yunhao Qian
  - [LinkedIn](www.linkedin.com/in/yunhao-qian-026980170)
  - [GitHub](https://github.com/yunhao-qian)
- Tested on my personal computer:
  - OS: Windows 11, 24H2
  - CPU: 13th Gen Intel(R) Core(TM) i7-13700 (2.10 GHz)
  - GPU: NVIDIA GeForce RTX 4090
  - RAM: 32.0 GB

## Boids in Action

### Screenshot

![Screenshot of boids](images/Screenshot%20of%20boids.png)

The screenshot above is from an experiment with 100000 boids, 1x-sized cells, scattered uniform grid, and the block size of 128.

### Recording

![Recording of boids](images/Recording%20of%20boids.gif)

The screen recording above is from an experiment with 320000 boids, 1x-sized cells, scattered uniform grid, and the block size of 128.

## Changes to `CMakeLists.txt`

- To share code among different configurations, I used lambda functions and `constexpr` in CUDA `__device__` code. To enable these features, I turned on the `--extended-lambda` and `--expt-relaxed-constexpr` flags using `target_compile_options()`.
- To measure frame rates programmatically, all compile-time configurations are controlled by CMake definitions. These include `N_FOR_VIS`, `VISUALIZE`, `FINE_GRAINED_CELLS`, `UNIFORM_GRID`, `COHERENT_GRID`, and `CUDA_BLOCK_SIZE`.
- The modified program has a special timed mode for measuring frame rates. To support this, CMake options are added, including `FPS_MEASURE`, `FPS_MEASURE_START`, and `FPS_MEASURE_DURATION`.

## Performance Analysis

### Methodology

All experiments in this section are launched programmatically using [`measure_fps.py`](./scripts/measure_fps.py), which re-compiles the CMake project using specific arguments, runs the program, and captures the average frame rate (in frames per second, or FPS) from stdout. The measurement of each experiment starts 2 seconds after the program's launch and lasts for 20 seconds. After that, the program exits automatically. To avoid crashing the computer, more resource-consuming configurations are skipped once the frame rate drops below 1 FPS or the program takes over 100 seconds to execute. Statistics used to create the following plots are omitted here for conciseness, and please refer to [`measurements.json`](./scripts/measurements.json) for those detailed numbers.

### Number of Boids

![Frame rate vs. number of boids](images/Frame%20rate%20vs%20number%20of%20boids.png)

All experiments in this set use 1x-sized cells and the block size of 128.

Discussion:

- The naive implementation should be heavily compute-bound, and the time complexity is $O(N^2)$, where $N$ is the number of boids. Experiments confirm this trend in general, as the FPS drops consistently as $N$ increases.
- The scattered & coherent uniform grid implementations do not show a monotonous trend at small $N$'s. The worst performance appears at $N = ~20000$. This may be caused by thread divergence in warps as not all neighbor cells are occupied by boids.
- The two uniform-grid implementations reach their best performance at $N = ~100000$. The GPU's compute capability is probably saturated at this point.
- When $N$ is very large, the time complexity is $O(N^2)$ for all implementations with or without uniform grids, as large $N$'s always increase the number of boids within the effective distance. This is confirmed by the linear trends in the log-log plot.

### OpenGL Visualization

While visualizing a point cloud in OpenGL has $O(N)$ time complexity, its amount of work is much lighter than boid simulations in general. When $N$ is small, experiments with visualization turned on have slightly lower FPS. When $N$ is large, the difference becomes barely detectable.

### Block Size

![Frame rate vs. block size](images/Frame%20rate%20vs%20block%20size.png)

All experiments in this set use 32000 boids, no visualization, and 1x-sized cells.

Discussion:

- For the naive implementation, FPS improves as the block size increases from 8 to 64. This is because increased computation helps cover the latency of random global memory accesses. There is a small drop at 1024, which is probably due to reaching register limits.
- The scattered uniform grid performs best at very small block sizes from 8 to 32, and then flattens. This is because the execution time is dominated by the latency of random global memory accesses. Increasing the block size further does not address this problem, and reduces the scheduler's ability to distribute blocks evenly among MP's.
- The coherent uniform grid has strong gains for 8 to 16, as increased computation covers the latency of semi-sequential global memory accesses. Once the program is compute-bound, increasing the block size further does not bring consistent improvements.

### Coherent Uniform Grids

The coherent uniform grid brings significant performance improvements, and this is the expected outcome. Sorting boid data by cells makes neighbor loops touch contiguous memory. Those data reside in $\leq 9$ pieces of contiguous memory if the cells are 1x-sized, and $\leq 4$ pieces if the cells are 2x-sized. When $N$ is large, the semi-sequential accesses are much faster than thousands of random accesses. When $N$ is small, the overhead of sorting boid data tends to dominate, resulting in the slightly lower FPS of the coherent uniform grid.

### Cell Size

![Frame rate vs. fine-grained cells](images/Frame%20rate%20vs%20fine-grained%20cells.png)

All experiments in this set use no visualization, coherent uniform grids, and the block size of 128.

When $N$ is small, 2x-sized cells should be slightly better, as the smaller number of neighbor cells means lighter looping overhead and better memory coherence. However, this advantage is not obvious is experiments, as it is hard to tell that one configuration is clearly better than the other.

When $N$ is very large, 1x-sized cells should be much better, as more fine-grained cells allow the program to check fewer neighboring boids. If the boid density is $\rho$, and the effective distance is $R$, 1x-sized cells result in $(3 R)^3 = 27 R^3$ neighboring boids, while 2x-sized cells result in $(4 R)^3 = 64 R^3$ neighboring boids. This discrepancy is confirmed by the roughly constant offset in the log-log plot.
