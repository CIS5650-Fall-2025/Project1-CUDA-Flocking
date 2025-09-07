**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Neha Thumu
  * [LinkedIn](https://www.linkedin.com/in/neha-thumu/)
* Tested on: Windows 11 Pro, i9-13900H @ 2.60GHz 32GB, Nvidia GeForce RTX 4070

### Project 1: Details 

[fancy picture/gif here!!]

## Implementation

In this project, I implemented a flocking simulation based on the Reynolds Boids algorithm that have two levels of optimization.  

## Performance Analysis 
![blocksizevsframerate](https://github.com/thumun/Project1-CUDA-Flocking/blob/main/images/blocksizevsframerate.png)
This chart shows how as the block size increases, the frame rate drops however the optimal block size appears to be 128 or 32 which can be seen by the little spike. 

![numboidsvsframerate](https://github.com/thumun/Project1-CUDA-Flocking/blob/main/images/numboidsvsframerate.png)
This chart shows how as the number of boids increases, the frame rate generally drops (as expected). The coherent case outperforms the scattered optimization in general and both optimizations vastly outperform the number of boids. It is interesting to note that for the case of 75k boids, the coherent case has a rather large spike in frame rate. This may be due it being well optimized for the number of threads per block.

![numboidsvsframeratenovisualization](https://github.com/thumun/Project1-CUDA-Flocking/blob/main/images/numboidsvsframeratenovisualization.png)
This chart is the same as above however there is no visualization. The general trends are the same as above but there are a few things to note. In the naive case, it is much better than the other two for the 2k boid amount which I found surprising. It is also interesting to see that scattered and coherent are quite similar in performance until there is more than 50k boids. 

## Responses 
For each implementation, how does changing the number of boids affect performance? Why do you think this is?
- ***Naive implementation***: In this, implementation, increasing the number of boids drastically decreases the performance which can be seen by the dramatic loss in framerate from 600 when there were only 2000 boids to only 14 when there were 100,000 (in the visualization scenario). This is due to having to check all other boids in the solution space in order to calculate each boid's velocity.
- ***Uniform implementation***: In this implementation, increasing the number of boids does lower the frame rate (as expected) but the amount it is lower is much better in comparison to the naive method. This is due to the optimization only considering boids that are neighboring and not looking at the complete solution space. (A more thorough explanation can be found in the implementation explanation above.) 
- ***Uniform Coherent implementation***: In this implementation, increasing the number of boids does lower the frame rate (as expected) but it is noticeably different than the uniform variant-at least in the case of no visualization. This is due to cutting out the middle man (a lookup into an array). (A more thorough explanation can be found in the implementation explanation above.) 
  
For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
- In general for each implementation, as the block size goes down the performance gets worse. In the naive case, there is not a noticeable change due how low it already is in the start but this can be seen in the two optimizations. I believe this is due to not using the warps to their full potential or due to not having enough registers for the data which may result in stalls. 

For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
- Yes, there were performance improvements if you compare them generally. This was the outcome I expected as the coherent uniform grid cuts out a lookup to the particlesArrayGrid and instead directlyy access the position and velocity for the neighbor boid. It is interesting to note that the two of them perform similarly until the number of boids becomes quite big. It is also rather odd (in my opinion) that coherent out performs by a rather significant amount for the case where there is 75000 boids. 
  
Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
- Yes, having a max of 8 neighboring cells was more accurate in getting the proper/true neighboring boids. In the 27 neighboring cells case, I was look at cells that may have been empty/devoid of boids and there may have been boids that are out of range for the rules) which are both a waste of a check. In the 8 cells case, there is logic where there is -1 for invalid indivies which prevents the issue of considering invalid boids. 
