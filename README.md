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

https://github.com/user-attachments/assets/901fc1e9-3a3c-4719-8fb4-4bc46a35e69f

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

Increasing the block size tended to improve performance, and the rate of improvement slowed as the block size grew.

> For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

The coherent uniform grid produced the highest frame rates for larger boid counts. Since it takes advantage of spatial locality in the cache I would expect a performance improvement, although the actual performance improvement is rather significant over the unoptimized grid, especially seeing as the coherent grid requires an extra kernel call to clone all the position and velocity vectors an additional time. 

> Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

Decreasing the cell width and comparing 27 cells instead of determining which 8 larger cells to check resulted in a slowdown by about one third. 