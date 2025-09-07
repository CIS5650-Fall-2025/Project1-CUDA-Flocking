**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Pavel Peev
  * [LinkedIn](https://www.linkedin.com/in/pavel-peev-5568561b9/), [personal website](www.cartaphil.com)
* Tested on: Windows 11, i7-1270, NVIDIA T1000

### Description
![BoidsScreenCapSmaller](https://github.com/user-attachments/assets/ce555eaa-3950-4a20-bdbe-53157f129ca2)

Flocking simulation based on the Reynolds Boids algorithm, run on an NVIDIA GPU using CUDA. 

Three implementations when searching for neighboring Boids when calculating cohesion, seperation, and alignment:

* Naive: Searches through all Boids
* Uniform: Uses a uniform grid to only search through boids in adjacent cells.
* Coherent: Uses a uniform grid and also aligns the boid spacial data for sequential access


### Performance Analysis

<img width="1920" height="1080" alt="BoidsPerformance" src="https://github.com/user-attachments/assets/16bc8ded-4f98-4dd6-997b-66aed1912da2" />
<img width="1920" height="1080" alt="BlockSizePerformance" src="https://github.com/user-attachments/assets/167cac06-ac04-4b7d-924f-2f3486163bdd" />
<img width="1920" height="1080" alt="CellPerformance" src="https://github.com/user-attachments/assets/b314a50e-bbb3-469f-9bf8-445be22ed4f6" />

### Analysis Questions

*For each implementation, how does changing the number of boids affect performance? Why do you think this is?

For the Naive implementation, increasing the number of Boids has an exponential decrease in performance, as each boid has to search through all it's neighbors. In contrast, the uniform and coherent grid implemention have an almost linear decrease in performance, as the performance loss from the increasing boids is spread to the grid cells, with the coherent implementation performing better as it avoids loading global memory onto the GPU more often

*For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

Increasing the block count past 128 leads to minor decreases in fps, which become more noticable at 1024 threads per block. This may be because 1024 is the maximum number of threads per block, and therefor the GPU cannot utilize the threads at maximum efficiency.

*For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

There was a significant performance increase with the coherent uniform grid. I was expecting this, as loading memory onto the GPU is a major bottleneck, and so any way of decreasing the amount of load memory calls should see a noticable performance improvement.

*Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

There was a slight performance increase when checking 27 smaller cells vs 8 larger cells. This is probably because by searching the smaller cells, there is less volume that can contain the boids, so there ends up being less boids to check overall.
