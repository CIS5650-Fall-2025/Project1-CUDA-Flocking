**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Qirui (Chiray) Fu
  * [personal website](https://qiruifu.github.io/)
* Tested on my own laptop: Windows 11, i5-13500HX @ 2.5GHz 16GB, GTX 4060 8GB

### README

#### Simulation Results
3k boids:
<p>
  <img src="/images/p3k.png" alt="Image 1" width="45%">
  <img src="/images/g3k.gif" alt="Image 2" width="45%">
</p>

100k boids:
<p>
  <img src="/images/p100k.png" alt="Image 1" width="45%">
  <img src="/images/g100k.gif" alt="Image 2" width="45%">
</p>

#### Performance Analysis
##### 1. Framerate change with increasing # of boids

<img src="/images/chart1.png" alt="Image 1" width="80%">

We can see that the framerates of methods with searching grids are much higher than the naive method. However, one thing noticable is for coherent uniform method, the framerate at 100k boids is higher than that at 70k and 50k boids. This phenomenon is counter-intuitive, I think the number 100k might triger something of GPU and gets a better performance.

##### 2. Framerate change with different block size

For each method, we test its performance with block size 32, 128, 512, 1024. We test on 30k boids for naive method, 500k boids for scattered grids and 1M boids for coherent grids. It's because these numbers can make framerate fall into range of 0 to 100. Here is the result:

<img src="/images/chart2.png" alt="Image 1" width="80%">

We can find that for coherent uniform grids and naive mothed, they have the best performance with block size of 128. But for scattered uniform grids, the performance differences are really low, framerate at 512 is a little higher than other options.

##### 3. Improvement of coherent memory access

First of all, it does increase the performance of program as we expected. However, the increase is more significant than I expected. For 500k boids, the framerate after optimization is almost twice than old method. This shows that the better memory access would lead to a much better performance.

##### 4. Framerate change with # of neighbors

<img src="/images/chart3.png" alt="Image 1" width="80%">

The fact is, searching 27 neighbor grids would have a better performance. I think it's because we have less particles to judge whether they are the neighbor boid. If our neighbor radius is $R$, then we only need to visit boids in a $(3R)^3$ cube in this method. If we use grids whose width is $2R$ and search 8 neighbors, this number is $(4R)^3$.