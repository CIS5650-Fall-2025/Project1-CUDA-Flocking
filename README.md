**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking - LATE DAY USED**

* Henry Han
  * https://github.com/sirenri2001
  * https://www.linkedin.com/in/henry-han-a832a6284/
* Tested on: Windows 11 Pro 24H2, i7-9750H @ 2.60GHz 16GB, RTX 2070 Max-Q

## Screenshots

![](/profiles/11.gif)

Tested with coherent grid algorithm with particle spawn number of 1,000,000

Average FPS 130

## Performance Analysis

### Analysis Conclusion

This boid algorithm works well under a certain ratio of particles per cell grid. More particle in a grid, more performance impact on particle interacts with neighbors. Also, more grid means more cost on sorting particles along with grid indices. Therefore, this algorithm can demostrate best performance when average number of particle in each grid cell stays within 0.1 ~ 10. 

### Changing of Factors

I tested the following factors that may have an impact on performance

- Particle Count
- Neighbor Searching Strategy (8x or 27x grid method)
- Simulation Domain (larger domain contains more grid)

Also I tested on blockSize, but it seems no impact on performance.

Here is a diagram of my test result.

![](/profiles/FrameRateFactors.png)


### Performance Analysis

I use following setting as baseline: 1 million particles with initial grid size and domain size, as well as using coherent method and 27x neighbor grids. This yields an FPS of 142.599, as shown in bold text in the diagram below.

![](/profiles/ProfileBaseLine.png)

Also, note that yellow line shows when there are fewer grid cells, more particles will stay in the same cell, hence more cost will occur when computing neighbor particles (see picture below). This profile record shows `kernUpdateVelNeighborSearchCoherent` takes 92% of time per frame, which means a lot of cost when a particle finding its neighbor to update its velocity. 

![](/profiles/1MillionWith10KGrid.png)

The grey line shows when there are too many grid cells. This leads to unnecessary cost for sorting the grid index. 

![](/profiles/1MillionWith515Million.png)

The green line is more optimal setting compared to base line. It has a more reasonalbe particle / cell ratio which significantly improve FPS. 

### Future Works

Future works may contains testing on more initial settings, with different grid sizes or particles. Also, interact ranges for particles might be another factors that affect the performance, since if particle pull other neighbors around them to themselves, there will be a dense area that is full of particles, which may drag down performance.