**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Griffin Evans
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on lab computer: Windows 11 Education, i9-12900F @ 2.40GHz 64.0GB, NVIDIA GeForce RTX 3090 (Levine 057 #1)

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

![](images/Screenshot%202025-09-04%20191957.png)
![](images/Screenshot%202025-09-04%20192620.png)

![](gifs/gif2.gif)
![](gifs/gif5.gif)

#### Benchmarking Methodology:

Using CUDA events (drawing from [this article](http://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)) I tracked the number of milliseconds that the simulation step took in each loop, and summed that value over a set number of frames so that I could determine the mean length of a simulation step over those frames. I tracked this information for a variety of configurations and copied it into a spreadsheet to graph and compare.

#### Performance Analysis:

- For each implementation, how does changing the number of boids affect performance? Why do you think this is?

![](images/Time%20Per%20Step%20(s)%20vs.%20Boid%20Count.png)
![](images/Simulation%20Steps%20Per%20Second%20vs.%20Boid%20Count.png)

For all three implementations, increasing the boid count would slow the simulation, increasing the time each step took to complete. The rate at which this change affected each of them varied between the implementations however, with the implementations that ran already slower with the initial 5000 boids also worsening more severely with increased boids. For example, increasing boid count from 5,000 to 10,000, the naïve implementation has the time taken per simulation step increase from 1.65487 ms to 4.06693 ms (a 145% increase), while the non-coherent uniform grid implementation goes from 0.219143 ms to 0.250066 ms (a 14.1% increase) and the coherent uniform grid implementation goes from 0.162406 ms to 0.18808 ms (a 15.8% increase). Increasing to 50,000 boids the naïve implementation continues to drastically slow while both grid implementations see only slight changes, but once increasing to 100,000 the non-coherent uniform grid implementation starts to slow significantly (step time of 0.299942 ms to 0.691533 ms, a 131% increase) while the uniform grid suffers only a slight drop in performance (step time of 0.211083 ms to 0.227745 ms, a 7.89% increase). Once we increase the boid count again to 500,000 the coherent grid implementation begins to suffer more drastic performance losses as well, going from 0.227745 ms to 1.4712 ms per simulation step (a 546% increase).

The naïve implementation being the most severely affected by the increased boid count is to be expected as in that implementation every thread needs to loop through all of the boids in the simulation, so for example by doubling the number of boids (such as from 5,000 to 10,000) we double both the number of threads and how many iterations each of them needs to make to finish, leading to the simulation step length more than doubling. The grid implementations, where each thread only needs to iterate through the boids within the neighboring cells, seem to only have severe slowdown once the number of boids passes some threshold; this appears to occur for a lower number of boids in the non-coherent implementation. Since we're increasing the number of boids without changing the number of cells (leaving the scene scale and neighborhood radii consistent), the density of boids in each cell increases, meaning even with a thread only checking the boids in 8 neighboring cells, it still has to iterate through an increasing number of boids. This leads to longer-running threads, which eventually seem to become a bottleneck and cause stalls. This happens slightly sooner for the non-coherent uniform grid implementation as the scattered memory structure slows the threads further—we both have more reads from global memory (reading the index array alongside the other arrays) and the reads are less efficient as they require accessing very disjunct regions in memory.

- For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
- For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
- Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
