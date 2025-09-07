Project 1 CUDA Flocking
====================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 1**

* Yannick Gachnang
  * [LinkedIn](https://www.linkedin.com/in/yannickga/)
* Tested on: Windows 10, EPYC 9354 @ 3.25GHz, 16GB RAM, RTX 2000 Ada 23GB VRAM

---

## Implementation

As per the project instruction, I implemented three different Boid flocking simulations.
But there were some additional things I decided to implement for the sake of completeness and to get more information for the performance testing.

### Naive

In the naive version, we launch a kernel for each boid in which we iterate over all the other boids. Then we calculate the pairwise distance between all of them and check to see if any of the interaction rules need to be applied.
We can then apply the rules as per the pseudo-code in the instructions. Later down the line, I changed the distance calculation to use the squared distance using `glm::dot` because this way we avoid the `sqrt` operation which is expensive (and improved performance here by about 10%).

### Uniform Grid

Calculating the distance between each pair of boids is expensive with $O(n^2)$ time complexity. But we can assign all boids into the cells of a uniform grid and then we only need to check neighboring grid cells.
At `cellWidth >= 2 * maximumInteractionDistance`, we can get away with only checking 4 cells in total in 2D and 8 cells in 3D. For anything between `2 * maximumInteractionDistance > cellWidth >= maximumInteractionDistance`, we need to check 27 cells and for anything below that we need to check even more.
I will investigate the effect this has on performance later on in the performance section.

### Coherent Grid

The uniform grid approach cuts out a lot of unnecessary computation (in exchange for a slightly bigger memory footprint). But for the coherent grid, instead of only sorting the arrays that keep track of which grid cell a boid is located in, we directly sort the boid data for better memory coherence.
In theory this should improve performance even further, which is also something I investigate in the performance section.

### CPU Version

For the sake of completeness, I also implemented a naive CPU version to see what the approximate scale of the difference is when moving from CPU to GPU for a simulation like this.

### Periodic Boundary Conditions

Aside from using the squared distance for slightly faster computation, I wasn't sure whether the distance calculation and grid calculations should wrap around the seemingly periodic boundary.
Therefore I implemented both versions which can be toggled between using the `periodicDistance` variable in `kernel.cu`.

---

## Performance Testing

### Testing Methodology

For all performance testing I turned off visualization (`#define VISUALIZE 0`). Each run consisted of a 50 frame warmup and then precisely 1000 frames of simulation. I averaged the results over three runs to smooth out any noise.  
The measurements are in frames per second (FPS). For reference I also included a naive CPU version, but I stopped testing that one after 5000 boids because it became too slow to be practical.  


### CPU vs GPU Scaling

The first test was to get a direct comparison between the naive CPU and naive GPU versions. I ran this from 1 boid up to 5000 boids.

### GPU Scaling

Next I compared only the GPU versions. Here I started at 1000 boids since the results below that are just dominated by overhead.

### FPS vs Neighbor Cell Count

I then tested the effect of different neighborhood sizes. I tested 8 cells, 27 cells, and 125 cells for both the uniform grid and the coherent grid. (Where 8 cells correspond to `cellWidth = 2 * maxDist`, 27 cells correspond to `cellWidth = maxDist` and 125 cells correspond to `cellWidth = 0.5 * maxDist`).

### FPS vs Block Size

Finally I tested the effect of block size. I used 50k boids and the 8-cell configuration for both the uniform and coherent grid. I tested block sizes of 32, 64, 128, 256 and 512.

