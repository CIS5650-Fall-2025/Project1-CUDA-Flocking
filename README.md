**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Pavel Peev
  * [LinkedIn](https://www.linkedin.com/in/pavel-peev-5568561b9/), [personal website](www.cartaphil.com)
* Tested on: Windows 11, i7-1270, NVIDIA T1000

### Description

Flocking simulation based on the Reynolds Boids algorithm, run on an NVIDIA GPU using CUDA. 

Three implementations when searching for neighboring Boids when calculating cohesion, seperation, and alignment:

* Naive: Searches through all Boids
* Uniform: Uses a uniform grid to only search through boids in adjacent cells.
* Coherent: Uses a uniform grid and also aligns the boid spacial data for sequential access


### Performance Analysis

<img width="1920" height="1080" alt="BoidsPerformance" src="https://github.com/user-attachments/assets/16bc8ded-4f98-4dd6-997b-66aed1912da2" />



