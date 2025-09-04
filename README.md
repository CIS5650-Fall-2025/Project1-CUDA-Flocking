**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Ruben Young
  * [LinkedIn](https://www.linkedin.com/in/rubenaryo/), [personal website](https://rubenaryo.com)
* Tested on: Windows 11, AMD Ryzen 7 7800X3D, RTX 4080 SUPER (Compute Capability 8.9)

![](images/boids.gif)

# Performance Analysis:

## Block size impact
Block size testing was done with N = 100,000. While block size certainly affects overall performance, there aren't significant gains or losses past a certain threshold (around 64 threads per-block).
However, having smaller blocks than this does have a marked, negative performance impact, especially on the naive boid implementation, where FPS can drop as low as 8 frames-per-second at blockSize=8.
![block size to fps](images/fps_blocksize.png)

## Boid count impact

### Naive
Predictably, the naive implementation's time complexity scales quadratically with the number of boids, as each boid must check every other boid in the scene causing performance to drop sharply past around 25,000 boids. 

### Scattered
The scattered uniform grid approach brings a major advantage in that each boid only has to check within cells that fall within its neighborhood range. While this comes in exchange to increased memory and performance overhead due to allocating and setting up the additional buffers, the performance is substantially improved when compared to the naive case.
Now, we improve from O(N^2) to O(NM) where M represents the number of possible neighbors in the boid's neighboring cells.

### Coherent
The vastly improved performance of the coherent implementation at higher boid counts was a surprise to me. 
We effectively trade some more overhead to reduce stalling from global memory reads in the inner loop. 
While FPS remains consistent with the scattered implementation up to around N=50,000, it significantly outperforms scattered at very high boid counts like 100,000 and 150,000. 

![boid count to fps novis](images/fps_boid_count_novis.png)
![boid count to fps vis](images/fps_boid_count_vis.png)

## Other observations
Two optimizations in the scattered and coherent implementations made massive differences in performance:
- Ensuring only the 8 relevant grid cells were checked, rather than 27
- Skipping empty cells with no boids in them. (Marked as a pre-process as gridCellStart = -1)

Checking 27 cells rather than 8 was particularly problematic due to the unnecessary global memory reads for gridCellStart/End. It is clearly worth it to go through the additional overhead of checking which 8 neighboring grid cells can actually contain neighbors.
Similarly, skipping empty cells saves on a lot of global memory reads, particularly due to the sheer number of cells in the simulation. 