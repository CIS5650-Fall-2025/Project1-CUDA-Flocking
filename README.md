**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Name: Harry Guan (17885658)
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: Windows 11, Intel i7-14700 @ 2.10GHz 32GB, NVIDIA T1000 4GB (Moore 100B virtual labs)

### Introducton
In this project, I implemented a 3D Boids flocking simulation on the GPU using CUDA. Where each particle move around following the following three rules

1. Rule 1: Boids try to fly towards the centre of mass of neighbouring boids
2. Rule 2: Boids try to keep a small distance away from other objects (including other boids).
3. Rule 3: Boids try to match velocity with near boids.

This is based on Conard Parker's notes with slight adaptation.

### Visualizations
![](images/boids-1.gif)

![](images/boids-2.gif)

![](images/boids-3.gif)

## Peformance Analysis

### Testing methodology
For testing, I developed a custom PerfTimer class to measure the performance of each boids simulation method. This class uses CUDA events to accurately time each simulation step on the GPU. The tests were run for a duration of 10 seconds per implementation, and the average frames per second (FPS) was calculated from the total frames rendered during that period. This approach provides a stable and reliable performance metric for comparing the naive, scattered grid, and coherent grid implementations.

### Performance with different number of boids 

| # of boids | Avg FPS(Brute force) | Avg FPS(Scattered Grid) | Avg FPS(Coherent Grid) |
| :--- | :--- | :--- | :--- |
| 1000 | 51.3 | 292.9 | 291 |
| 5000 | 6.3 | 116.6 | 120.4 |
| 10000 | 1.1 | 66.9 | 68.9 |
| 50000 | 0.4 | 12 | 11.9 |
| 100000 | 0.00929732 | 4.93 | 4.82 |
| 500000 | 0 | 0.281402 | 0.267018 |