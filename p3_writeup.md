
## Part3 write-up
* For each implementation, how does changing the number of boids affect performance? Why do you think this is?
    
    As the number of boids increase, performance drop. Because more boids would require more frequent access to memory for accessing related position and velocity informatoin. Also once the number of boids exceed the maximum number of threads supported by hardware, we need more gpu cycles to run all boids.
* For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

    According to my experiments. Increasing the block count causes worse performance in naive implementation ( 25% less fps). While for uniform and coherent grids it barely has any impact on performance. <br>
    For naive implementation, i think it was because more branches occured inside one block, causing wasted cycles.
    For coherent and uniform grid, the branching is minizied since the number of neighbours to check decreased on each thread.

* For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

    I saw slightly worse performance with coherent uniform grids. Maybe my implementation is faulty. I suspect the additional sorting required more time, and the memory coherency isn't utilized to local SM cache partitions

* Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

    I observed checking 27 performs better than 8 about 2-3% with both uniform and coherent grids. I suppose because the total volume to check is smaller with 27, since 27 means checking 3^3 unit volumes while there is a chance that 8 end up checking (2*2)^3 unit volumes. But the difference is marginal. 