**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

+ Hongyi Ding
  * [LinkedIn](https://www.linkedin.com/in/hongyi-ding/), [personal website](https://johnnyding.com/)

* Tested on: Windows 11, i7-14700@ 2.10GHz 32GB, NVIDIA T1000 8GB (AGH 200 SEAS-LAB0403-WD)

### Screenshots

The screenshots are taken when the flock size is 20000. The default setting is

```
rule1Distance = 5.0f; rule1Scale = 0.01f
rule1Distance = 3.0f; rule1Scale = 0.1f
rule1Distance = 5.0f; rule1Scale = 0.1f
```

with `maxSpeed = 1.0f`. Some interesting patterns can be observed if we tune these parameters.

![video1](images/video1.gif)

If we increase `rule1Distance` to `25.0f`, the flock will form groups in a sphere. The distance is set to a larger value, then the group is also larger and thus more stable.

![video1](images/video2.gif)

If we increase `rule1Scale` to `0.1f`, the flock will move a a different way. They will be more likely to form small groups, because the bond between members in the same group become stronger.

![video1](images/video3.gif)

If we increase `rule2Scale` to `0.3f`, the boids will look crazy because we make them keep away with others. The larger`rule2Scale` is, the more the boids hate others.

![video1](images/video4.gif)

If we increase `rule3Distance` to `25.0f`, the flock will behave like some fluid as a whole. This is because the boids will follow the pace of the majority as `rule3Distance` increases.

![video1](images/video5.gif)

If we increase `rule3Scale` to `0.5f`, the flock will form large group that move with the same speed. As `rule3Scale` increases, the boids are more likely to follow their neighbors' speed.

![video1](images/video6.gif)

If we keep investigating into these parameters, I believe more interesting patterns will be found. That's why this algorithm is so widely used in 3D effects, games and movies.

### Performance Analysis

This table reflects how the **average fps** changes as we change the **flock size** and acceleration method (with visualization disabled). Note: the statistics are a rough estimate with ±50fps deviation, as it takes too long for the avgfps to converge.

| flock size         | Naive | Uniform Grid | Coherent Grid |
| ------------------ | ----- | ------------ | ------------- |
| 5000               | 1020  | 2010         | 2300          |
| 20000              | 105   | 1400         | 1800          |
| 50000              | 19    | 520          | 860           |
| 100000             | 5     | 240          | 460           |
| 20000 (visualized) | 100   | 910          | 1180          |

As we can conclude from the table above, in a certain range, if the flock size if `n` and the average fps is `p`, then for naive simulation,
$$
p\sim \frac{1}{n^2}
$$
and for grid simulation, in a certain range of setting,
$$
p\sim \frac{1}{n}
$$
However, due to limitation of statistics, the conclusion is only a rough guess. We may need more test and data to prove the relationship of fps and flock size and method used.

This table reflects how the **average fps** changes as we change the block size (with visualization disabled). The flock size is set to 20000, and the simulation method is coherent grid. The visualization is enabled.

| block size | avg fps |
| ---------- | ------- |
| 64         | 1160    |
| 128        | 1150    |
| 256        | 1180    |
| 512        | 1100    |

### Answer to Questions

* For each implementation, how does changing the number of boids affect performance? Why do you think this is?

  Roughly speaking, for naive simulation, the performance is proportional to the inverse square of the number of boids. Because we need to calculate the impact between any pair of boids, then the computation amount is `O(n^2)`; While for grid simulation, since we only need to check a few neighbor boids for each boid, the performance is proportional to the inverse of the number of boids.

* For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

  The performance doesn't change a lot as we increase block size (block count is decreased as a result). I think this is because the number of total threads needed is large, and the GPU performance is optimized whichever way we organize them. And we didn't do any implementations for shared memory or other same-block optimization.

* For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

  Yes, there is much improvement in performance when we apply coherent grid. The improvement is smaller than from naive simulation to uniform grid, which is as expected. The performance increased because we avoid some random memory access in GPU kernels, which may cause a lot of delay because the accesses happen in global memory. The performance does not change dramatically because the algorithm complexity stays the same.

* Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

  I don't think this affect the performance a lot, so I didn't implement the 8 neighboring algorithm. The number of neighboring grids we calculate differs by 3, but this doesn't mean the performance also differs by 3 times. For those unnecessary grids calculated, we won't take them into account, because after validation, the boids are to far from the center grid. There are only some extra memory accesses. However, they won't take much time because the memory addresses are very close to other grids we are accessing.