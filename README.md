# Project 1 - CUDA Boids
**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Rachel Lin

  * [LinkedIn](https://www.linkedin.com/in/rachel-lin-452834213/), [personal website](https://www.artstation.com/rachellin4), [Instagram](https://www.instagram.com/lotus_crescent/)

* Tested on: (TODO) Windows 11, 12th Gen Intel(R) Core(TM) i7-12700H @ 2.30GHz, NVIDIA GeForce RTX 3080 Laptop GPU (16 GB)

## Naive Neighbor Search

For each boid, this implementation naively checks every other existing boid to compute a new velocity each frame.

### 5,000 Boid Simulation Using Naive Algorithm

<img src="images/flocking1.gif" width="40%">

<img src="images/flocking2.gif" width="40%">

### 50,000 Boid Simulation Using Naive Algorithm

<img src="images/naive_50k.gif" width="40%">


## Uniform Grid Search

This implementation checks only boids that are within the same neighborhood (i.e. they are at a close enough distance to actually influence the current boid's velocity).

### 50,000 Boid Simulation Using Uniform Grid Algorithm

<img src="images/scattered_50k.gif" width="40%">


## Coherent Grid Search

This implementation is similar to the uniform grid search, but instead of using pointers from each boid to its position and velocity data index, it rearranges the boid data so that it can be directly accessed using the boid's grid cell start and end indices.

### 50,000 Boid Simulation Using Coherent Grid Algorithm

<img src="images/coherent_50k.gif" width="40%">


## Performance Analysis

# Average Simulation Time Without Visualization

| Number of Boids | Naive | Uniform Grid | Coherent |
| --------- | --------- | --------- | --------- |
| 2500 | 1.519 | 0.401 | 0.475 |
| 5000 | 2.288 | 0.627 | 0.760 |
| 10000 | 3.800 | 0.732 | 0.551 |
| 25000 | 6.073 | 0.964 | 0.662 |
| 50000 | 15.760 | 2.113 | 1.145 |

<img src="images/Average Simulation Time vs. Number of Boids.png" width="50%">

# Average Search Time Without Visualization

| Number of Boids | Naive | Uniform Grid | Coherent |
| --------- | --------- | --------- | --------- |
| 2500 | 1.255 | 0.173 | 0.103 |
| 5000 | 2.089 | 0.288 | 0.128 |
| 10000 | 3.601 | 0.331 | 0.154 |
| 25000 | 5.858 | 0.446 | 0.186 |
| 50000 | 15.524 | 1.495 | 0.308 |

<img src="images/Average Search Time vs. Number of Boids.png" width="50%">

