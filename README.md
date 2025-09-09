**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Christina Qiu
  * [LinkedIn](https://www.linkedin.com/in/christina-qiu-6094301b6/), [personal website](https://christinaqiu3.github.io/), [twitter](), etc.
* Tested on: Windows 11, Intel Core i7-13700H @ 2.40GHz, 16GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU (Personal laptop)

### Overview

This is an implementation of a flocking simulation based on the Reynolds Boids algorithm. This project involved writing CUDA kernels, using them, and analyzing their performance. 

1. Naive boids simulation.
2. Scattered boids simulation on uniform grid.
3. Coherent boids simulation on uniform grid.

## Naive Boids

At every timestep, a boid looks at each of its neighboring boids and computes the velocity change contribution from each of the three rules:

1. cohesion - boids move towards the perceived center of mass of their neighbors
2. separation - boids avoid getting to close to their neighbors
3. alignment - boids generally try to move with the same direction and speed as their neighbors

Thus, a brute force implementation has each boid check every other boid in the simulation.

![Simulation Demo](images/hw_1_100.gif)

<p align="center">
  <img src="images/100_1.png" alt="Before" width="45%">
  <img src="images/100_2.png" alt="After" width="45%">
</p>

50,000 Boids Simulation

Runtime O(n^2)

## Scattered Uniform Grid

Each boid is assigned to a grid cell based on its position. We store an array of particle indices, which are sorted by their grid cell. The actual position and velocity data remain in their original arrays so we must look up the sorted index to access a boid's data. This enables fast neighbor search but can cause memory access inefficiencies due to scattered data.

![Simulation Demo](images/hw_1_110.gif)

<p align="center">
  <img src="images/110_1.png" alt="Before" width="45%">
  <img src="images/110_2.png" alt="After" width="45%">
</p>

50,000 Boids Simulation

## Coherent Uniform Grid

The coherent uniform grid improves on the scattered version by also reordering the position and velocity arrays so that boids within the same grid cell are stored contiguously in memory. This cuts out the middle-man and significantly improves performance, especially on large numbers of boids.

![Simulation Demo](images/hw_1_111.gif)

<p align="center">
  <img src="images/111_1.png" alt="Before" width="45%">
  <img src="images/111_2.png" alt="After" width="45%">
</p>

50,000 Boids Simulation
