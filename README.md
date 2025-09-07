**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Joanne Li
  * [LinkedIn](https://www.linkedin.com/in/zhuoran-li-856658244/)
* Tested on: Windows 11, AMD Ryzen 5 5600H @ 3.30 GHz 16.0GB, NVIDIA GeForce RTX 3050 Laptop GPU 4GB

## README
### Final results
![](images/proj1_1.gif)

### Performance analysis





### Note
For Boids Rule 3, I added the little missing detail following Conrad Parker's notes. I wrote:
```
velocity += (pv - vel[iSelf]) * rule3Scale;
```
And it works just fine, except the particles's velocities become noticeably slower. I increased `rule3Scale` and `rule3Distance` to 8.0 and 0.5. The behavior is slightly different from the reference, but it still correctly represents Boids.