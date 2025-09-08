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