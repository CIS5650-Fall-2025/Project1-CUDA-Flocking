**University of Pennsylvania, CIS 5650: GPU Programming and Architecture**

![static image](images/cis5650_boids_VxUVwdcb4n.png)

![Preview](images/cis5650_boids_sL8pwY9mqT.gif)



# Project1-CUDA-Flocking 

* Xiaonan Pan
  * [LinkedIn](https://www.linkedin.com/in/xiaonan-pan-9b0b0b1a7), [My Blog](www.tsingloo.com), [GitHub](https://github.com/TsingLoo)
* Tested on: 
  * Windows 11 24H2
  * 13600KF @ 3.5Ghz
  * 4070 SUPER 12GB
  * 32GB RAM

# Overview

This is my first CUDA project focusing on the Boids flocking algorithm. This project implements a **naive** algorithm and also explores parallel optimization methods, including the **Scattered and Coherent Uniform Grid techniques**. The final implementation can support a simulation of **50,000 boids at 1,744 frames per second (FPS)**. The results, performance analysis, and the potential reasons for the speedups are discussed in this document.



# Performance  & Discussion

## FPS vs. Number of Boids （Q1、Q3）

![fps-method-#](images/linechart.png)

Both uniform grid implementations are further optimized with an adaptive neighboring cell search. This strategy dynamically determines the search range, checking a smaller region of 8 adjacent cells when the interaction radius is small relative to the cell width, and expanding to a larger 27-cell region when the radius is larger. The `blocksize` of these runs is 128. 

As shown in the performance chart, the naive method, with its around $O(n^2)$ complexity, exhibits a rapid performance drop as the number of boids (N) increases. It becomes unusable (under 60 FPS) for simulations with more than 50,000 boids. While **the scattered and coherent version gives a quite good result**,  They maintain high frame rates even as N increases by orders of magnitude. However, **a distinct turning point is observed for the scattered grid method at approximately 230,000 boids**, where its performance collapses. This is likely due to the dataset exceeding the capacity of the GPU's hardware cache resources, introducing a large latency.  The coherent grid method, which physically reorders boid data in memory, avoids this cache cliff and maintains high performance at a much larger scale. Interestingly. However, **for a small number of boids (e.g., N=1,000), the coherent method is slightly slower than the scattered method**. This is attributed to the overhead of the extra data-reordering kernel launch, the cost of which is not fully offset by the memory access benefits when the dataset is small.



## FPS vs. Blocksize & Blockcount （Q2）

![fps-method-#](images/blocksize.png)

Both uniform grid implementations are further optimized with an adaptive neighboring cell search. The `N` of these runs is 10000. The block count is calculated by ` dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);`

As shown in the above chart, these is no a significant impact on performance for either method. I think this is because the most bottleneck here is the heavy memory access instead of math calculation. There is a lot of the kernel's time is spent waiting for boid position and velocity data to be fetched, more or less `blocksize` does not increase or decease the memory access operation.



## 27 vs 8 neighboring cells ? (Q4)

Well, I think this really depends on the density of the boids. 

From my experiment, low densities (e.g., N=500), the 8-cell approach is approximately 10% more performant. In this case, most grid cells are empty or contain very few boids. The dominant cost is the "broad phase"—the overhead of looping through cells and looking up their memory indices. The 8-cell method wins because it minimizes these lookups. 

Conversely, at high densities (e.g., N=500,000), the 27-cell approach becomes nearly 40% faster. Here, every cell is packed with boids, and the dominant cost shifts to the the expensive process of calculating the distance to every potential neighbor found. The 27-cell method's smaller cells provide a much more accurate initial filter, significantly reducing the number of "false positive" boids that must be checked. While it checks more cells, the work done per cell is so much less that it results in a substantial overall performance gain. 
