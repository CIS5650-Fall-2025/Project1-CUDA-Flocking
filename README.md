**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Jacqueline (Jackie) Li
  * [LinkedIn](https://www.linkedin.com/in/jackie-lii/), [personal website](https://sites.google.com/seas.upenn.edu/jacquelineli/home), [Instagram](https://www.instagram.com/sagescherrytree/), etc.
* Tested on: Windows 10, 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz, NVIDIA GeForce RTX 3060 Laptop GPU (6 GB)

# Program Overview

This program is an implementation of boids simulation on the GPU, in which three methods are implemented and subsequently compared. 
1. Naive boids simulation. 
2. Scattered boids simulation on uniform grid.
3. Coherent boids simulation on uniform grid.

#### 50,000 Boids Simulation

| ![](images/50000Boids.gif) |
|:--:|

| ![](images/50000Boids.png) | ![](images/50000Boids2.png) | ![](images/50000Boids3.png) |
|:--:|:--:|:--:|


### Naive Boids Algorithm

Idea: For each boid, naively check every other boid to compute velocity according to rules.

Pseudocode:

```
for each boid i in boids array:
    check all other boids against curr boid
    vel_new[i] = compute_new_velocity(pos, vel)

for each boid i in boids array:
    pos[i] += vel_new[i] * dt

swap(vel_old, vel_new)
```

Runtime: O(N^2)

#### Naive Algorithm Images

| ![](images/HW1.2_naiveBoidsUpdate.png) | ![](images/HW1.2_naiveBoids.gif) |
|:--:|:--:|

### Scattered Uniform Grid Algorithm

Idea: Sets up uniform spacial grid so each boid only checks grid cells close enough for rules of boids. Positions/velocities remain in original arrays.

Pseudocode:

```
1. Compute grid indices for all boids
for each boid i in parallel:
    label boid by grid cell indices
    label boid by particle array indices

2. Sort boids by grid index (so same-cell boids are grouped)
sort_by_key(gridIndex, particleArrayIndex)

3. Identify start & end indices of each grid cell
for each sorted boid i in parallel:
    if gridIndex[i] changes:
        mark start/end of cell

4. Velocity update with neighbor search (scattered)
for each boid i in parallel:
    for each neighbour in range of boid i:
	vel_new[i] = compute_new_velocity(pos, vel)

5. Update positions
for each boid i in parallel:
    pos[i] += vel_new[i] * dt

swap(vel_old, vel_new)
```

Runtime: O(N) * max_neighbours of boid

#### Scattered Algorithm Images

| ![](images/HW2.1_scatteredBoids.png) | ![](images/HW2.1_scatteredBoids2.png) |
|:--:|:--:|

| ![](images/HW2.1_scatteredBoids.gif) |
|:--:|

### Coherent Uniform Grid Algorithm

Idea: Same as scattered grid, but after sorting, reshuffle particle data so boid data in same grid are updated in memory.

Pseudocode:
```
1. Compute grid indices for all boids
for each boid i in parallel:
    label boid by grid cell indices
    label boid by particle array indices

2. Sort boids by grid index
sort_by_key(gridIndex, particleArrayIndex)

3. Identify start & end indices of each grid cell
for each sorted boid i in parallel:
    if gridIndex[i] changes:
        mark start/end of cell

4. Additional step: Reshuffle pos/vel arrays into coherent order
for each sorted boid i in parallel:
    reshufflePos of boid = position of boid's corresponding particle index
    reshuffleVel of boid = position of boid's corresponding particle index

5. Velocity update with neighbor search (scattered)
for each boid i in parallel:
    for each neighbour in range of boid i:
	vel_new[i] = compute_new_velocity(pos, vel)

6. Update positions
for each boid i in parallel:
    reshufflePos[i] += vel_new[i] * dt

7. Restore coherent positions back to original array order
for each sorted boid i in parallel:
    position of particle index = reshufflePos of boid
    velocity of particle index = reshuffleVel of boid

swap(vel_old, vel_new)
```

Runtime: O(N) * neighbours
Additional strength over Scattered: for boids in same cell, memory is consequently gathered together, better GPU cache.

#### Coherent Algorithm Images

| ![](images/HW2.3_coherentBoids.png) | ![](images/HW2.3_coherentBoids2.png) |
|:--:|:--:|

| ![](images/HW2.3_coherentBoids.gif) |
|:--:|


## Runtime Analysis

