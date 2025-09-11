**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Zwe Tun
  * LinkedIn: https://www.linkedin.com/in/zwe-tun-6b7191256/
* Tested on: Intel(R) i7-14700HX, 2100 Mhz, RTX 5060 Laptop
![CUDA Flocking](images/CUDA-Flocking.gif)

100,000 boids with Coherant Grid search 
### Overview 
Boids are artificial agents that simulate the behavior of flocking animals. Introduced by Craig Reynolds in 1986, each boid follows three simple rules:
Cohesion - boids move towards the perceived center of mass of their neighbors
Separation - boids avoid getting to close to their neighbors
Alignment - boids generally try to move with the same direction and speed as their neighbors

### Implementation 

## Naive 
The naive implementation uses a straightforward algorithm: each boid iterates over every other boid in the system to calculate its updated velocity and position based on the three flocking rules (separation, alignment, and cohesion).
While conceptually simple, this approach results in O(n²) time complexity, making it highly inefficient for large numbers of boids. Every boid must check all others regardless of distance, leading to significant computational overhead.
![CUDA Flocking](images/Naive-CUDA-Flocking.gif)

10,000 boids with Naive  

## Uniform Grid Search
A more effcient algorithm is dividing into 3D cells, and each boid is assigned to a cell based on its position. By storing cell indices and mapping boid data accordingly, each thread can now limit its neighbor search to only nearby cells rather than the entire boid population.
In the scattered version, boid data (position, velocity) is stored in separate buffers, and additional lookup is required to gather information based on the grid cell. This greatly reduces the number of comparisons per boid, improving performance. 
![CUDA Flocking](images/Uniform-CUDA-Flocking.gif)

10,000 boids with Uniform Grid Search  

## Coherant Grid Search 
The coherent grid improves on the uniform grid approach by taking advantage of spatial locality to optimize memory access on the GPU. In the scattered grid version, boid data is stored in separate buffers and accessed via indirect lookups, causing threads to read from scattered memory locations. This leads to reduced cache utilization. The coherent grid reorganizes the boid data so that boids located in the same or neighboring grid cells are stored contiguously in memory. By reshuffling the boid data arrays to align with their cell indices, the GPU can access data in adjacent memory locations, further improving performance. 
![CUDA Flocking](images/Coherant-CUDA-Flocking.gif)

10,000 boids with Coherant Grid Search  

## Performance Analysis
The following perfomance analysis will be run on Windows 11, Intel(R) i7-14700HX, 2100 Mhz, RTX 5060 Laptop. Frames Per Second (FPS) will be the metric to determine performance of alogorthms discusssed above. Higher FPS can be thought of us better performance. 


