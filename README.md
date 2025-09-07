**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,  
Project 1 - Flocking**

* Lewis Ghrist  
* [LinkedIn](https://www.linkedin.com/in/lewis-ghrist-4b1b3728b/), [personal website](https://siwel-cg.github.io/siwel.cg_websiteV1/index.html#home)  
* Tested on: Windows 11, AMD Ryzen 9 5950X 16-Core Processor, 64GB, NVIDIA GeForce RTX 3080 10GB (Personal PC)

---

## Overview

### Example Simulation Runs 
These were all run with the coherent grid implementation. 
Rule 1 distance: 8.0 | Rule 1 strength: 0.05
Rule 2 distance: 3.0 | Rule 2 strength: 0.1
Rule 3 distance: 4.0 | Rule 3 strength: 0.1

- **10,000 boids**  
  ![10k boids](BOIDS_834_10000.gif)

- **50,000 boids**  
  ![50k boids](BOIDS_834_50000.gif)

- **100,000 boids**  
  ![100k boids](BOIDS_834_100000.gif)

---
Rule 1 distance: 10.0 | Rule 1 strength: 0.01
Rule 2 distance: 2.0 | Rule 2 strength: 0.1
Rule 3 distance: 8.0 | Rule 3 strength: 0.1

- **1,000,000 boids**  
  ![1M boids](BOIDS_1028_1000000.gif)

---
Rule 1 distance: 5.0 | Rule 1 strength: 0.01
Rule 2 distance: 2.0 | Rule 2 strength: 0.1
Rule 3 distance: 10.0 | Rule 3 strength: 0.1

- **2,000,000 boids**  
  ![1M boids](BOIDS_5210_1000000.gif)

---

## Performance Results
The average fps was calculated over a 10 second window for each simulation method with varying numbers of boids.
### FPS Line Graph
![FPS Line Graph](FPS_LineGrpah_V1.png)

### FPS Breakdown by Block Size
![Block Size Graph](FPS_BlockGraph_V1.png)

### FPS Table
![FPS Table](FPS_Table_V1.png)

At **1M+ boids**, only the coherent grid remains practical.
