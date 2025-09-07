**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Joanne Li
  * [LinkedIn](https://www.linkedin.com/in/zhuoran-li-856658244/)
* Tested on: Windows 11, AMD Ryzen 5 5600H @ 3.30 GHz 16.0GB, NVIDIA GeForce RTX 3050 Laptop GPU 4GB

## README
### Part 1 - Brute force
For Part 1, I implemented the 3 rules of Boids, and used brute force to compute each particle's new velocity - for each particle, the algorithm iterates over all existing particles. 
For Rule 3, I added the little missing detail following Conrad Parker's notes. I wrote:
```
velocity += (pv - vel[iSelf]) * rule3Scale;
```
And it works just fine, except the particles's velocities become noticeably slower. I increased `rule3Scale` and `rule3Distance` to 8.0 and 0.5. The behavior is slightly different from the reference, but it still correctly represents Boids.

![](images/proj1_1.gif)

Observation: For 5,000 particles, the simulation runs smoothly at 60 fps. However, when the number of particles increases to 50,000, the frame rate drops to 13-14 fps.