**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Raymond Feng
  * [LinkedIn](https://www.linkedin.com/in/raymond-ma-feng/), [personal website](https://www.rfeng.dev/)
* Tested on: Windows 11, i9-9900KF @ 3.60GHz 32GB, NVIDIA GeForce RTX 2070 SUPER (Personal Computer)

# Overview
In this project, we implemented a Boids flocking simulation. In the simulation space, particles move around according to three rules:
1. Cohesion: Boids move towards the center of mass of neighboring boids
2. Separation: Boids maintain some distance away from other boids
3. Alignment: Boids try to move in the same direction and speed as neighboring boids

In this gif, the simulation is run on 50,000 boids.
![](/images/main_gif.gif)

## More Simulations
Each of these simulations was run with the default parameters:
- Time step: 0.2
- Rule 1 distance: 5.0
- Rule 1 scale: 0.01
- Rule 2 distance: 2.0
- Rule 2 scale: 0.1
- Rule 3 distance: 5.0
- Rule 3 scale: 0.1
- Max speed: 1.0

### Naive
N=5000

![](/images/naive_gif.gif)

### Uniform Grid
N=50000

![](/images/uniform_gif.gif)

### Coherent Grid
N=50000

![](/images/coherent_gif.gif)

## Data
The average FPS was calculated over a period of 10 seconds, with varying numbers of boids.

![](/images/Graph1.png)

### Data Table
Additional boid counts for the naive implementation were not tested because it became impractical and slow.

![](/images/Graph2.png)

## Performance Analysis
Changing the amount of boids slowed down performance, decreasing the amount of frames per second. This is because with the increase of the amount of boids, more calculations per second will have to be made. This is especially apparent with the naive implementation, where you're comparing every boid to every other boid. Thanks to the optimizations of the grid implementations, you only have to compare every boid against a handful of other boids in the neighborhood, speeding up performance by a significant amount. 

I found that changing the block size did not dramatically increase or decrease performance. 

As you can see from the graph, as the amount of boids increased, the coherent implementation became more efficient than the uniform implementation. This surprised me somewhat because in the coherent implementation, I used an additional kernel calculation to line up the position and velocity vectors. I didn't expect for the table look-up in the uniform implementation to be so expensive compared to the additional kernel.
