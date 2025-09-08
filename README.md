**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**




* Daniel Chen
  * https://www.linkedin.com/in/daniel-c-a02ba2229/
* Tested on: Windows 11, AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (3.80 GHz), RTX 4070 notebook

## Writeup
The following answers are as requested in [INSTRUCTION.md](/INSTRUCTION.md).

### Captures
![screenshot](/images/scrn.png)

[recording](/images/rec.mp4)

https://github.com/user-attachments/assets/5f65ead3-24a1-4437-a3c9-5485ea8a1f4e

### Performance analysis
After setting `VISUALIZE` to `0` and monitoring the window title, we estimate the following average framerates for each algorithm for various boid counts:

![fps graphs](/images/fps.png)


After setting `VISUALIZE` to `1`, we estimate:

![fps graphs](/images/fps-vis.png)


After setting `N_FOR_VIS` to `500000` with a coherent uniform grid, we estimate for various block sizes:

![fps graphs](/images/fps-bs.png)



> For each implementation, how does changing the number of boids affect performance? Why do you think this is?

Each implementation's measurement in seconds / frame increases roughly quadratically with the number of boids. This aligns with the idea that for each boid, we compare it against $\approx O(n)$ other boids, resulting in an $O(n^2)$ running time (should the number of boids surpass the number of those that can be updated in parallel).

> For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

Increasing the block size tended to improve performance, and the rate of improvement slowed as the block size grew, even decreasing from 512 to 1024. This could be because fewer blocks would need to be scheduled, although having larger blocks could increase the chance that there are threads that return immediately and are therefore waiting while not assigned to any boids.

> For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

The coherent uniform grid produced the highest frame rates for larger boid counts. Since it takes advantage of spatial locality in the cache I would expect a performance improvement, although the actual performance improvement is rather significant over the unoptimized grid, especially seeing as the coherent grid requires an extra kernel call to clone all the position and velocity vectors an additional time. 

> Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

Decreasing the cell width and forcing 27 cells to be checked instead of determining which 8 larger cells to check resulted in a slowdown by about one third. This could be because having 8 times as many grid cells could drastically increase the chance of cache misses on the `gridCellStartIndices` and `gridCellEndIndices` arrays, which are also accessed at least 3.3 times as often.


### Extra credit

The set of neighbor cells to check is determined based on a bounding box surrounding each boid, allowing up to 8 or 27 cells to be checked dynamically.
