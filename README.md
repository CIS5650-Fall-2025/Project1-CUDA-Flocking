**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Lu Men
  * [LinkedIn](https://www.linkedin.com/in/lu-m-673425323/)
* Tested on PC: Windows 11 Home, AMD Ryzen 7 5800HS @ 3.20GHz 16GB, NVIDIA GeForce RTX 3060 Laptop GPU 6GB (Compute Capability 8.6)

### 1000 Boids Simulation

<img src="./images/boids_1000.gif" alt="Screenshot" style="width:80%; height:auto;">

### 10000 Boids Simulation

<img src="./images/boids_10000.gif" alt="Screenshot" style="width:80%; height:auto;">

### 50000 Boids Simulation

<img src="./images/boids_50000.gif" alt="Screenshot" style="width:80%; height:auto;">


## **Performance Analysis**

<img src="./images/plot.png" alt="Screenshot" style="width:80%; height:auto;">

### **Effect of Boid Count on Performance**

For all three methods to detect neighboring boids, increasing the number of boids decreases the frames per second (FPS). This is a direct result of the computational complexity associated with a larger flock. Each boid must check its surroundings for neighbors to follow the flocking rules. As the number of boids increases, the number of potential interactions grows significantly. This leads to more threads to manage, more neighbors to consider for each boid, and a greater overall computational load, which in turn reduces the FPS. 

***

### **Effect of Block Count and Block Size on Performance**

Changing the **block count** and **block size** directly influences the performance of the GPU-based implementations. These parameters determine how the workload is distributed among the GPU's processing units.

* **Block Count:** Increasing the block count divides the work among more blocks, allowing for greater parallelism. However, if the number of blocks is too high, it can introduce overhead in managing these blocks, potentially hurting performance.
* **Block Size:** The block size affects how efficiently the GPU's streaming multiprocessors are utilized. An optimal block size fully occupies the SMs without exceeding their resource limits. A block size that is too small underutilizes the hardware, while one that is too large can lead to resource contention and decreased performance.

***

### **Coherent Uniform Grid Performance**

The **coherent uniform grid** implementation yielded an interesting performance outcome.

* **Expected Outcome:** It was anticipated that the coherent uniform grid would consistently outperform the naive and scattered methods, especially with a large number of boids. This is because the grid-based approach provides a more efficient way to locate neighbors, avoiding the need for a costly brute-force search.
* **Actual Results:** The results showed that for a small number of boids (less than 3000), the coherent uniform grid had a slightly lower FPS. This is likely due to the initial overhead of setting up and managing the grid and sorting/unsorting. The naive or scattered approaches, while less efficient in principle, may have a simpler, lower-overhead implementation that works better for small numbers of boids. However, as the number of boids scaled up (to over 50,000), the coherent uniform grid's efficiency became evident, performing multiple times better than the other methods. The naive approach, in particular, does not scale well because of its brute-force neighbor search strategy.

***

### **Effect of Cell Width and Neighboring Cells**

Changing the **cell width** and the number of neighboring cells checked (8 vs. 27) has an insignificant and nuanced effect on performance.

* **Cell Width:** The optimal cell width is twice the maximum interaction radius of a boid. If the cell width is too small, a boid's neighbors might be in adjacent cells, requiring more cell checks. If it's too large, it may contain many boids that are not neighbors, leading to unnecessary distance calculations. An optimal cell width minimizes the number of cells to check and the number of unnecessary distance checks.

* **Checking 27 vs. 8 Neighboring Cells:** Checking 27 cells (including the current cell and its immediate neighbors in 3D space) is typically slightly slower than checking 8 cells (in 2D space) because it involves more data access and comparisons. A well-defined 2D uniform grid ensures that all interacting neighbors for a boid are contained within its own cell and the 8 surrounding cells. Therefore, checking 27 cells is redundant and inefficient for a 2D simulation, as the additional 18 cells would not contain any relevant neighbors.