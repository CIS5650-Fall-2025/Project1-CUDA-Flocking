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

The screenshot above is from an experiment with the following configurations:

- Block size: 128
- Number of boids: 100000
- Mode: scattered uniform grid (with visualization)

### Recording

![Recording of boids](images/Recording%20of%20boids.gif)

The screen recording above is from an experiment with the following configurations:

- Block size: 128
- Number of boids: 320000
- Mode: scattered uniform grid (with visualization)

## Performance Analysis

### Frame Rate vs. Number of Boids

![Frame rate vs. number of boids](images/Frame%20rate%20vs%20number%20of%20boids.png)

Numbers for plotting are listed in the following table. Each number is the average over measurements of 15 seconds. All experiments used the block size of 128. Configurations that are too resource-consuming to launch with are left as N/A.

| Number of Boids | Naive (w/o vis.) | Naive (w/ vis.) | Scattered Grid (w/o vis.) | Scattered Grid (w vis.) | Coherent Grid (w/o vis.) | Coherent Grid (w/ vis.) |
| --------------- | ---------------- | --------------- | ------------------------- | ----------------------- | ------------------------ | ----------------------- |
| 5000            | 1057.18          | 669.401         | 954.948                   | 637.855                 | 1136.49                  | 682.369                 |
| 10000           | 578.73           | 430.536         | 606.269                   | 448.857                 | 852.869                  | 567.523                 |
| 20000           | 303.208          | 246.626         | 450.065                   | 378.916                 | 761.894                  | 557.11                  |
| 40000           | 156.814          | 133.908         | 547.381                   | 420.64                  | 1126.77                  | 715.706                 |
| 80000           | 80.9807          | 66.5314         | 754.17                    | 500.578                 | 1533.94                  | 844.657                 |
| 160000          | 33.699           | 21.914          | 298.219                   | 250.635                 | 1562.2                   | 856.094                 |
| 320000          | 18.9407          | 7.64789         | 105.776                   | 96.8395                 | 1180.88                  | 666.505                 |
| 640000          | 12.8056          | 2.02074         | 28.1882                   | 27.1921                 | 505.753                  | 364.128                 |
| 1280000         | 10.4033          | 0.521029        | 7.32915                   | 7.16215                 | 158.943                  | 137.602                 |
| 2560000         | N/A              | N/A             | 1.88598                   | 1.81145                 | 42.5598                  | 40.4236                 |
| 5120000         | N/A              | N/A             | 0.463071                  | 0.409476                | 10.9749                  | 10.6321                 |
| 10240000        | N/A              | N/A             | N/A                       | N/A                     | 2.81504                  | 2.72167                 |
| 20480000        | N/A              | N/A             | N/A                       | N/A                     | 0.744258                 | 0.682546                |

Naive:

- FPS falls ~inversely with $N$ because the work is $O(N^2)$. With vis it collapses earlier.  
- Explanation: each boid compares against all others; doubling N roughly quadruples work. "(w/ vis.)" adds extra $O(N)$ draw/transfer/sync overhead, which hurts more at small–mid $N$.

Uniform scattered grid: non-monotonic

- Down at small $N$: grid build + under-utilization
- Up at mid $N$: better SM occupancy
- Down at large $N$: random neighbor reads saturate memory bandwidth

Uniform coherent grid:

- Rises at small–mid $N$, then drops at very large $N$.  
- Explanation: algorithmic work is $~O(N)$ if cell occupancy stays bounded, and the coherent layout (sorted boids) makes neighbor reads cache- and coalescing-friendly. At very large $N$ the program hits memory bandwidth + per-frame sort overhead ($N log N$) + heavier per-cell occupancy.

### Frame Rate vs. Block Size

![Frame rate vs. block size](images/Frame%20rate%20vs%20block%20size.png)

Numbers for plotting are listed in the following table. Each number is the average over measurements of 15 seconds. All experiments used 320000 boids and turned off visualization.

| Block Size | Naive   | Scattered Grid | Coherent Grid |
| ---------- | ------- | -------------- | ------------- |
| 8          | 12.7485 | 158.967        | 453.725       |
| 16         | 14.7857 | 129.124        | 743.291       |
| 32         | 17.6827 | 105.249        | 1068.75       |
| 64         | 18.9289 | 105.831        | 1176.1        |
| 128        | 18.9404 | 105.803        | 1181.44       |
| 256        | 18.9422 | 108.078        | 1200.36       |
| 512        | 19.0132 | 108.38         | 1096.78       |
| 1024       | 17.8544 | 108.489        | 1110.18       |

Recall: `blocksPerGrid = ceil(N / blockSize)`. Increasing block size ⇒ more threads per block but fewer blocks. Effects differ by memory pattern:

Naive:

- 8→512 improves, 1024 dips slightly.
- Explanation: larger CTAs raise occupancy and hide latency until register/shared-mem limits reduce concurrent CTAs/SM; then gains flatten or regress.

Uniform scattered grid:

- Best at very small blocks, then flattens.
- Explanation: the kernel is memory-latency dominated with random accesses. Many small CTAs give the scheduler more to interleave and better hide long stalls; bigger CTAs don’t improve coalescing but reduce CTA multiplicity.

Coherent grid:

- Strong gains to 128–256, then slight drop at 512–1024.
- Explanation: coherent layout benefits from larger CTAs (better coalescing and L2 reuse across warps) up to the point where CTA resources limit concurrency.

### Q3: Coherent Uniform Grid

Yes, significantly, and this is the expected outcome.

Explanation: sorting by cell makes neighbor loops touch contiguous memory. The$ O(N log N)$ sort amortizes once N is moderate because neighbor traversal dominates frame time. At very small $N$ the gap is small because overheads dominate.

### Q4

Don’t count cells; count neighbors actually visited and memory locality.

Let dimension = d, cell width = w, density = ρ, stencil size = S (e.g., 27 vs 8).  
Neighbor-loop work per boid ~ `S × (ρ × w^d)`.

- Small cells (w ≈ interaction radius R) → larger stencil (e.g., 27 in 3D / 9 in 2D) but sparser cells.  
- Large cells (w ≈ 2R) → smaller stencil (e.g., 8 in 3D / 4 in 2D) but each cell holds ~`2^d` more boids.

3D intuition:  
27-cell @ w=R → cost ∝ 27·ρR³.  
8-cell @ w=2R → cost ∝ 8·ρ(2R)³ = 64·ρR³.  
Even though 27>8, the 27-cell case can be ~2.4× cheaper because cells are 8× less populated.
