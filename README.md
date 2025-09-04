**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Avi Serebrenik
  * [https://www.linkedin.com/in/avi-serebrenik-a7386426b/](), [https://aviserebrenik.wixsite.com/cvsite]()
* Tested on: Windows 11, i7-13620H @ 2.4GHz 32GB, RTX 4070 8GB

### Overview
 ![](gif.gif)
 This gif was recorded on the machine described above with 50000 boids and a timestep of 0.02 to make the animation smooth.
 The boids follow a generic flocking algorithm based on Conrad Parker's notes: [https://vergenet.net/~conrad/boids/pseudocode.html]()
 
 ## 1. Naive Boid Flocking
  - Overview: This algorithm naively checks every single other boid to see whether they're close enough for flocking.
  - Implementation: The main algorithm runs on one kernel that simply implements each rule one by one, summing their velocities
    and clamping them by a maximum velocity. This is not performant, and the rules could be grouped, but I wanted this to be
    truly "naive" as a good benchmark for the more optimized algorithms.
    
 ## 2. Uniform Grid Flocking
  - Overview: This algorithm optimizes naive flocking by only checking grid cells close enough for the rules for boids.
  - Implementation: The optimization is achieved in multiple steps. First, each boid is labeled by its array index and grid cell
    index for tracking and for efficient access. Next, the array index is sorted by the grid cell indices using Thrust. I then check
    which boid index starts and ends each grid cell, storing these in two separate arrays. Importantly "gridCellStartIndices" array is
    first filled with "-1," marking which cells are empty so we can skip these later. Next, the main body of the algorithm gets our
    boid index from our array and finds which grid cell we're in based on its position. I then check which nearby grid cells are even
    close enough to be within a flocking neighborhood distance based on the position and our flocking rules, and loop over only these
    cells, skipping the empty ones. Using the start and end indices arrays, I then easily loop through each boid in each cell, accumulating
    their information for the flocking calculations. Finally, the positions are updated and the velocity arrays get ping-ponged, same as
    in the naive algorithm.
    
 ## 3. Coherent Grid Flocking
  - Overview: This algorithm further optimizes the Uniform Grid method by "cutting out the middle man" - the index array - by sorting
    the position and velocity arrays, allowing for direct lookup of nearby memory.
  - Implementation: This is much the same as the Uniform Grid algorithm, with the big difference being how I access positions and
    velocities in my main algorithm, and that I sort the position and velocity arrays after filling the start and end grid indices
    arrays. Ideally, I would have liked to use thrust for this sorting too, but that gave me some bugs, so I use a kernel to sort
    and another one to unsort before the next loop, as positions would get confused otherwise in the current setup. The performance
    of this method versus the one above is discussed below in the results section, but it is the most performant.

### Results
 I have plotted the runtime of these algorithms (in ms) with varying boid counts at 0.2s dt to see the difference these optimizations make,
 however I would focus on the comparisons between the algorithms and not the specific values, as I had other applications open, and my method
 for getting the milliseconds was counting to 5 in my head and then capturing an Nsight GPU Trace.
 The plots also compare searching all neighboring cells VS only searching neighboring cells that could have boids in the flocking distance.
 These plots have the same data, but the bottom one uses a log scale for time to see the differences between the fast algorithms clearly.
 The naive method also can not handle more than 100k boids on my laptop, which is why its data points end there.
 The block size was 128 for these plots and the data was visualized. For different block size comparisons and non-visualization performance
 comparisons, please see the graphs further down.
 
  ![](plotVis.png)
  ![](plotLogVis.png)
  
 As we can clearly see, the naive implementation is by far the worst; optimizing neighboring cell searches provides a significant boost in
 performance; and the Coherent Grid algorithm is faster than the Uniform Grid one. However, the difference between Uniform and Coherent grids
 is quite small when dealing with a small number of boids, and Uniform is actually faster when searching all the neighbors. Furthermore,
 besides the naive algorithm, they all experience quite a considerable performance boost between 100k and 500k boids.
 For the first result, I believe that this is due to memory searches still not being that far off when dealing with these lower numbers,
 and the Coherent algorithm relies on two extra kernels to sort and unsort my position and velocity arrays, leading to slightly more operations.
 However, as boid numbers increase, these operations pay off and we can clearly see the increase in performance.
 For the second result, I am a little baffled. I only have a hypothesis, which might be a little nonsensical. My theory is that after a certain
 number of boids (somewhere between 100k and 500k), we overload the grid cells to such a degree that the boids become relatively evenly distributed
 in each cell. This means that each thread will have to generally check the same number of boids, and thus they rarely stall. However, with 100k, we
 we still have some sparse areas which lead to some threads having much more work than others, creating some massive stalling. We can see these
 boid values in the image below:
 
  ![](combined.png)
  
  When comparing between visualizing VS not visualizing, we see the expected result that not rendering the visuals speeds up the simulation.
  This is consistent accross the non-naive implementations and is about a factor of 2. For the naive implementation, the difference is quite
  minimal, since the algorithm is throttled by other problems, to the point where visualizing came out faster for 10k boids.
  
  ![](plotNoVis.png)
  
  When comparing block size, the performance stays consistent. I was curious about why this was the case and checked in Nsight. I found that
  my bottlenecks were glClear and swapping buffers, which are not affected by block size changes, so this lack of performance change is expected.
  
  ![](plotBlocks.png)
  ![](profiler.png)
 
### Question Answers
These are the questions and answers as part of this homework.
 - For each implementation, how does changing the number of boids affect performance? Why do you think this is?
     I discussed this above, but the number of boids decreases performance as there are simply more calculations to do. Interestingly, there is an
     increase in performance between 100k and 500k boids. I discuss my theory about why this is right above the 100k 500k boid image.
 - For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
     Block size change is also discussed right above this section. In short, block size doesn't really affect performance because the main bottlenecks
     in the algorithm are glClear and buffer swapping. As for block count, right now we are maximizing the number of threads per block, so we can't
     really decrease block count. If we increase block count we will have less threads per block. The naive instinct is to say that this would decrease
     performance, since we are waiting on thread execution more, but depending on the number of boids, these values could sync to actually increase performance
     by a bit. However, I think we can say that as a general rule of thumb for an unknown amount of boids we wouldn't really want to increase our blocks.
 - For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
     Yes, the performance improved which was the outcome that I expected because the lookup was nearer in memory due to the sorted arrays. However, the difference
     was quite minimal with a small number of boids, which makes sense, since the increase in performance is based on boid # and comes at a slight cost of extra sorting,
     meaning that at small boid numbers this extra sorting can take longer than the time we save by near memory lookup.
 - Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
    As I show in my results above, I first checked all 27 neighboring cells and then only the relevant ones (generally 8). Generally speaking, checking the 8 cells
    was more performant, although this difference is quite minimal when the boids are a small number. This is because of the 27 cells, only 8 will actually have the
    relevant boids, so we have 19 cells of boid comparisons (the neighbor rule distance checks) per each rule that we completely waste, decreasing performance.
