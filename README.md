**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Griffin Evans
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on lab computer: Windows 11 Education, i9-12900F @ 2.40GHz 64.0GB, NVIDIA GeForce RTX 3090 (Levine 057 #1)

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)


Benchmarking Methodology:

Using CUDA events (drawing from [this article](http://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)) I tracked the number of milliseconds that the simulation step took in each loop, and summed that value over a set number of frames so that I could determine the mean length of a simulation step over those frames. I tracked this information for a variety of configurations and copied it into a spreadsheet to graph and compare.
