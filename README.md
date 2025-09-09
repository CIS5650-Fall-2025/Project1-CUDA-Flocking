**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Yuning Wen
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc (Really sorry but still not ready yet, will update in the future).
* Tested on: Windows 11, i9-12900H @ 2.50GHz 16GB, NVIDIA GeForce RTX 3060 Laptop GPU (Personal Laptop)

### README

**1 Late Day used here!**

First of all, a gif for what I've done within this project writing for boids.

![tryGIF](./images/runtime%20boids.gif)

There are naive method, uniform grid, and coherent grid, in total three methods that implement Boids flocking simulation here!


#### Answering the Questions

* For each implementation, changing the number of boids larger will make the performance worse, or specifically, lower the framerate. I believe that it is because we have more data to analysis and thus takes more time to calculate with the neighbors. The more neighbors a boid has, or potentially has, the more time one thread need to use to calculate the final result for the certain boid. See graphs and analysis below.

* For each implementation, changing the block size does not affect a lot to the performance, or framerate. I believe that it is because the smaller the block size is, the more block it will take. However, the total number of threads should still be similar, and we will always have all the threads calculating the same frame together at the same time, which is not affected by the block size. As a result, changing the block size in this case does not really affect the performance.

* For the coherent uniform grid, I do experience performance improvement with the more coherent uniform grid. It is an outcome I expected, but I do think that the improvement it provides is huge, as I was not so care with the memory size when coding on CPU only. Here I believe that the less memory one thread uses (especially large memory to allocate such as size as large as the number of threads), the better it performs. I think space is more valuable here in parallel programming than time? As parallel programming saves a lot of time but not so enough space for large usage.

* By changing the cell width of the uniform grid to be the neighborhood distance instead of twice the neighborhood distance, I have perform both the two cases for test and result recorded (but no graph record below), and it seems like the modified version, or the smaller cell, performs better. I'm not sure if that is always the case, as this time I'm testing with 5000000 boids, and the smaller cells in this case may exclude more neighbors that are not in range, which thus finally result in the smaller cells performance twice the framrate of the default setting.

#### Performance Analysis

* Here is the graph of the relationship between frame per second (framerate) and the number of boids

![Vis no](./images/Vis%20no.png)

  From this graph I'd like to conclude that, in general, no matter what method we are using, FPS decreases as the number of boids gets larger.

![Vis yes](./images/Vis%20yes.png)

  Similar things happens to the one with visualization. However, since the visualized version has the upper bound limited by the hardware, and I currently may only have access to at most 165 fps, good algorithms (especially Coherent Grid) may not show a significant trend here.

* Here is the graph of the relationship between the block size and frame per second (framerate).

![block_size](./images/Block%20Size.png)

  The line seems to be weird, but actually the data changes only a little bit, which I may assume that the size of the block does not show a strong correlation with frame per second (framerate), at least correlation not shown with this amount of data. (Testing with 500000 boids)