**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Sirui Zhu
  * [LinkedIn](https://www.linkedin.com/in/sirui-zhu-28a24a260/)
* Tested on: Windows 11, i7-13620H, RTX 4060 (Personal)

## Demo

### 100,000 Boids Simulation
![100000 Boids GIF](from_start_to_end.gif)

The above shows the full simulation process with 100,000 boids.

---

## Simulation Screenshots

### 5,000 Boids
| Start | End |
|-------|-----|
| ![5000 Start](5000start.png) | ![5000 End](5000end.png) |

### 100,000 Boids
| Start | End |
|-------|-----|
| ![100000 Start](100000start.png) | ![100000 End](100000end.png) |

## Boids Simulation Performance 
![Framerate vs Boids (without visualization)](Boids_without_visualization.png)

![Framerate vs Boids (with visualization)](Boids_with_visualization.png)

![Framerate vs Blocksize (without visualization)](Blocksize.png)
---

### Q1. For each implementation, how does changing the number of boids affect performance? Why do you think this is?

- **Naive:**  
  Performance decreases extremely rapidly as the number of boids grows (e.g., ~4400 fps at 100 boids → ~3.5 fps at 100,000 boids without visualization). This is because the algorithm is **O(N²)**: every boid checks all others, so computation explodes as population increases.

- **Uniform grid:**  
  Performance decreases much more gradually. The grid restricts neighbor checks to nearby cells, so the complexity is closer to **O(N)** in practice. This allows it to scale far better than the naive method.

- **Coherent grid:**  
  Similar to the uniform grid at small scales, but consistently faster at larger scales. The improvement comes from reordering boids in memory so neighbors are contiguous, which enables coalesced global memory access and reduces latency.

- **With visualization:**  
  All values are lower due to rendering overhead, but the same trends hold: naive collapses at scale, while uniform and coherent grids degrade more gracefully.

---

### Q2. For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

- **Naive:**  
  Sensitive to block size. Performance peaks at block sizes around 128–256 (560–570 fps) but decreases for larger sizes (330 fps at 1024). This occurs because larger blocks reduce GPU occupancy: too many registers/shared memory per block means fewer active warps, which makes it harder to hide memory latency.

- **Uniform and coherent grid:**  
  Both are much less sensitive to block size. Performance remains consistently high (2100–2400 fps) with only a slight dip at 1024. This stability is due to smaller per-thread workloads and structured memory access, which reduce reliance on block configuration.

---

### Q3. For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

Yes. The coherent grid consistently outperformed the regular uniform grid.  
- Example: At 10,000 boids with visualization → 1450 fps (coherent) vs. 1200 fps (uniform).  
- Cause: Reordering ensures that boids in the same cell are contiguous in memory, improving **spatial locality** and **memory coalescing**.  
- This was the expected outcome. At higher boid counts, memory access patterns become the main bottleneck, so improving memory layout provides measurable gains.

---

### Q4. Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?

- **Cell width:**  
  - Too large → too many boids per cell, increasing unnecessary checks.  
  - Too small → many nearly empty cells, adding overhead.  
  - Optimal width is close to the boid interaction radius.

- **Neighbor count:**  
  - It is incorrect to assume that checking 27 neighbors is always slower.  
  - With **8 neighbors**, cell width must be larger to avoid missing boids, which increases per-cell density and the number of checks.  
  - With **27 neighbors**, cell width can be smaller, reducing boids per cell. Even though more cells are checked, the overall work can be less.  
  - Result: Depending on boid distribution, 27-cell checks can be as fast or even faster than 8-cell checks.

---