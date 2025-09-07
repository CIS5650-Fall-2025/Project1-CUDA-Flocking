#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?

// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_pos_coherent;
glm::vec3* dev_vel1_coherent;
glm::vec3* dev_vel2_coherent;

// Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  cudaMalloc((void**)&dev_pos_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_coherent failed!");

  cudaMalloc((void**)&dev_vel1_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1_coherent failed!");

  cudaMalloc((void**)&dev_vel2_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2_coherent failed!");

  // This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/*
* Helper function for boids rules based on readme
* Edits passed in data
*/
__device__ void RuleLogic(int minBounds, int maxBounds, int currB,
    const glm::vec3* pos, const glm::vec3* vel, int& ruleOne_neighbours,
    int& ruleThree_neighbours, glm::vec3& separate, glm::vec3& perceived_velocity,
    glm::vec3& perceived_center, int naiveFlag = 0, int * particleArrayIndices = nullptr) {

    for (int i = minBounds; i < maxBounds; i++) {
        // skipping comparison of same boid
        if (currB == i) {
            continue;
        }

        int compareBoid;

        if (naiveFlag == 1) {
            compareBoid = particleArrayIndices[i];
        }
        else {
            compareBoid = i;
        }

        float distance = glm::distance(pos[currB], pos[compareBoid]);

        // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
        if (distance < rule1Distance) {
            perceived_center += pos[compareBoid];
            ruleOne_neighbours++;
        }

        // Rule 2: boids try to stay a distance d away from each other
        if (distance < rule2Distance) {
            separate -= (pos[compareBoid] - pos[currB]);
        }

        // Rule 3: boids try to match the speed of surrounding boids
        if (distance < rule3Distance) {
            perceived_velocity += vel[compareBoid];
            ruleThree_neighbours++;
        }
    }
}

/*
* Helper function to calculate output velocity of Boid 
* Inputs calculated in RuleLogic
*/
__device__ glm::vec3 ComputeFinalVelocity(int currB, const glm::vec3* pos, int ruleOne_neighbours,
    int ruleThree_neighbours, glm::vec3 separate, glm::vec3 perceived_velocity,
    glm::vec3 perceived_center, glm::vec3 velocity_out) {

    if (ruleOne_neighbours > 0) {
        perceived_center /= ruleOne_neighbours;
        velocity_out += (perceived_center - pos[currB]) * rule1Scale;
    }

    if (ruleThree_neighbours > 0) {
        perceived_velocity /= ruleThree_neighbours;
        velocity_out += perceived_velocity * rule3Scale;
    }

    velocity_out += separate * rule2Scale;

    return velocity_out;
}

/**
* You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {

    glm::vec3 perceived_center = glm::vec3(0);
    int ruleOne_neighbours = 0;

    glm::vec3 separate = glm::vec3(0);

    glm::vec3 perceived_velocity = glm::vec3(0);
    int ruleThree_neighbours = 0;

    glm::vec3 velocity_out = vel[iSelf];

    RuleLogic(0, N, iSelf, pos, vel, ruleOne_neighbours,
        ruleThree_neighbours, separate, perceived_velocity,
        perceived_center);

    return ComputeFinalVelocity(iSelf, pos, ruleOne_neighbours,
        ruleThree_neighbours, separate, perceived_velocity,
        perceived_center, velocity_out);
}

/**
* implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
   
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    // Check if idx is out of bounds. If yes, return.
    if (index >= N)
        return;

    // Compute a new velocity based on pos and vel1
    glm::vec3 new_velocity = computeVelocityChange(N, index, pos, vel1);

    // Clamp the speed
    if (glm::length(new_velocity) > maxSpeed) {
        new_velocity = (new_velocity / glm::length(new_velocity)) * maxSpeed;
    }

    // Record the new velocity into vel2. Question: why NOT vel1?
    // using vel2 because don't want read/write to be from same arr
    vel2[index] = new_velocity;
}

/**
* Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// Consider this method of computing a 1D index from a 3D grid index.
// Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

/*
* fills two arrays for non-naive methods
* dev_particleGridIndices = array that maps boid to the grid cell that it is in
* dev_particleArrayIndices = array that keeps track of boid indices (used in scattered logic)
*/
__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // Label each boid with the index of its grid cell.
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // finding cell boid is in by converting world pos to pos in grid space
    // gridMin = disp from origin
    // mult by inverseCellWidth for grid space
    // floor to get int index val
    glm::vec3 gridIndex_3d = floor((pos[index] - gridMin) * inverseCellWidth);

    // converting to 1D index
    int gridIndex = gridIndex3Dto1D(gridIndex_3d[0], 
                                    gridIndex_3d[1], 
                                    gridIndex_3d[2], 
                                    gridResolution);

    // Set up a parallel array of integer indices as pointers to the actual
    // boid data in pos and vel
    indices[index] = index;
    gridIndices[index] = gridIndex;
}

// Consider how this could be useful for indicating that a cell
// does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

/*
* sets up the start and end index arrays 
* these tell which boids make up the range of boids in each cell
* helps in processing the neighboring boids in non-naive methods
*/
__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

    int currIndex = particleGridIndices[index];

    // boid can be BOTH the start and the end index of a grid cell if only one there!
    // edge case
    if (index == 0) {
        gridCellStartIndices[currIndex] = index;
    }
    else if (currIndex > particleGridIndices[index - 1]) {
        gridCellStartIndices[currIndex] = index;
        gridCellEndIndices[particleGridIndices[index - 1]] = index-1;
    }

    // edge case
    if (index == N - 1) {
        gridCellEndIndices[currIndex] = index;
    }
}

/*
* uniform grid, non-naive implementation
* instead of checking entire solution space, only look at neighbors
* find neighbors by using grid-based method (neighboring cells)
*/
__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth, 
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // Identify the grid cell that this particle is in
    glm::vec3 gridIndex_3d = floor((pos[index] - gridMin) * inverseCellWidth);
    
    // get distance from the rules!
    float distance = rule1Distance > rule2Distance ? rule1Distance : rule2Distance;
    distance = distance > rule3Distance ? distance : rule3Distance;

    // moving from world space to grid space
    distance *= inverseCellWidth;

    // variables to be used for velocity calculation/updating
    glm::vec3 perceived_center = glm::vec3(0);
    int ruleOne_neighbours = 0;

    glm::vec3 separate = glm::vec3(0);

    glm::vec3 perceived_velocity = glm::vec3(0);
    int ruleThree_neighbours = 0;

    glm::vec3 velocity_out = vel1[index];

    // Identify which cells may contain neighbors. This isn't always 8.
    // get the min/max bounds of each dim using the distance from the rules
    for (int x = imax(0, (gridIndex_3d[0] - distance)); x <= imin(gridResolution - 1, (gridIndex_3d[0] + distance)); x++) {
        for (int y = imax(0, (gridIndex_3d[1] - distance)); y <= imin(gridResolution - 1, (gridIndex_3d[1] + distance)); y++) {
            for (int z = imax(0, (gridIndex_3d[2] - distance)); z <= imin(gridResolution - 1, (gridIndex_3d[2] + distance)); z++) {

                // For each cell, read the start/end indices in the boid pointer array.
                int cellIndx = gridIndex3Dto1D(x, y, z, gridResolution);

                // skipping invalid indicies
                if (gridCellStartIndices[cellIndx] == -1 || gridCellEndIndices[cellIndx] == -1)
                {
                    continue;
                }

                RuleLogic(gridCellStartIndices[cellIndx], gridCellEndIndices[cellIndx], index, pos, 
                    vel1, ruleOne_neighbours, ruleThree_neighbours, separate, perceived_velocity,
                    perceived_center, 1, particleArrayIndices);
            }
        }
    }

    // Clamp the speed change before putting the new speed in vel2

    glm::vec3 new_vel = ComputeFinalVelocity(index, pos, ruleOne_neighbours,
        ruleThree_neighbours, separate, perceived_velocity,
        perceived_center, velocity_out);

    if (glm::length(new_vel) > maxSpeed) {
        new_vel = (new_vel / glm::length(new_vel)) * maxSpeed;
    }

    vel2[index] = new_vel;
}

/*
* above but better b/c cutting out middleman (particleArrayIndicies)
*/
__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // Identify the grid cell that this particle is in
    glm::vec3 gridIndex_3d = floor((pos[index] - gridMin) * inverseCellWidth);

    // get distance from the rules!
    float distance = rule1Distance > rule2Distance ? rule1Distance : rule2Distance;
    distance = distance > rule3Distance ? distance : rule3Distance;
    // moving from world space to grid space
    distance *= inverseCellWidth;

    // variables to be used for velocity calculation/updating
    glm::vec3 perceived_center = glm::vec3(0);
    int ruleOne_neighbours = 0;

    glm::vec3 separate = glm::vec3(0);

    glm::vec3 perceived_velocity = glm::vec3(0);
    int ruleThree_neighbours = 0;

    glm::vec3 velocity_out = vel1[index];

//   DIFFERENCE: For best results, consider what order the cells should be
//   checked in to maximize the memory benefits of reordering the boids data.
//   Access each boid in the cell and compute velocity change from
//   the boids rules, if this boid is within the neighborhood distance.

    // Identify which cells may contain neighbors. This isn't always 8.
    // get the min/max bounds of each dim using the distance from the rules
    for (int x = imax(0, (gridIndex_3d[0] - distance)); x <= imin(gridResolution - 1, (gridIndex_3d[0] + distance)); x++) {
        for (int y = imax(0, (gridIndex_3d[1] - distance)); y <= imin(gridResolution - 1, (gridIndex_3d[1] + distance)); y++) {
            for (int z = imax(0, (gridIndex_3d[2] - distance)); z <= imin(gridResolution - 1, (gridIndex_3d[2] + distance)); z++) {

                // For each cell, read the start/end indices in the boid pointer array.
                int cellIndx = gridIndex3Dto1D(x, y, z, gridResolution);

                // skipping invalid indicies
                if (gridCellStartIndices[cellIndx] == -1 || gridCellEndIndices[cellIndx] == -1)
                {
                    continue;
                }

                RuleLogic(gridCellStartIndices[cellIndx], gridCellEndIndices[cellIndx], index, pos,
                    vel1, ruleOne_neighbours, ruleThree_neighbours, separate, perceived_velocity,
                    perceived_center);
            }
        }
    }

    // Clamp the speed change before putting the new speed in vel2

    glm::vec3 new_vel = ComputeFinalVelocity(index, pos, ruleOne_neighbours,
        ruleThree_neighbours, separate, perceived_velocity,
        perceived_center, velocity_out);

    if (glm::length(new_vel) > maxSpeed) {
        new_vel = (new_vel / glm::length(new_vel)) * maxSpeed;
    }

    vel2[index] = new_vel;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {

  int blocks = (numObjects - 1 + blockSize) / blockSize;

  // use the kernels you wrote to step the simulation forward in time.
  kernUpdateVelocityBruteForce<<<blocks, threadsPerBlock>>>(numObjects, dev_pos, dev_vel1, dev_vel2);

  // update positions
  kernUpdatePos << <blocks, threadsPerBlock >> > (numObjects, dt, dev_pos, dev_vel2);

  // ping-pong the velocity buffers
  // ping pong = swapping the buffers 
  // b/c don't want to read/write to same one
  glm::vec3* ptr = dev_vel2;
  dev_vel2 = dev_vel1;
  dev_vel1 = ptr;
}

/*
* above idea but integrating using neighbors to alter vel
* using thrust to sort grid & array in order to set up start/end grid index arrs 
*/
void Boids::stepSimulationScatteredGrid(float dt) {
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  
    int boidsBlocks = (numObjects - 1 + blockSize) / blockSize;

    kernComputeIndices << <boidsBlocks, threadsPerBlock >> > (numObjects,
        gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices,
        dev_particleGridIndices);

  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.

    thrust::device_ptr<int> dev_thrust_particleGridIndices(dev_particleGridIndices);
    thrust::device_ptr<int> dev_thrust_particleArrayIndices(dev_particleArrayIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices

    int cellBlocks = (gridCellCount - 1 + blockSize) / blockSize;

    // creates invalid vals!!
    kernResetIntBuffer << <cellBlocks, threadsPerBlock >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <cellBlocks, threadsPerBlock >> > (gridCellCount, dev_gridCellEndIndices, -1);

    kernIdentifyCellStartEnd << <boidsBlocks, threadsPerBlock >> > (numObjects,
        dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  
    // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered << <boidsBlocks, threadsPerBlock >> > (numObjects,
        gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices,
        dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

  // - Update positions
    kernUpdatePos << <boidsBlocks, threadsPerBlock >> > (numObjects, dt, dev_pos, dev_vel2);

  // - Ping-pong buffers as needed
    glm::vec3* ptr = dev_vel2;
    dev_vel2 = dev_vel1;
    dev_vel1 = ptr;
}

// creating a copy of vel1 & pos to cut out middle man
__global__ void sortArray (int N, glm::vec3 * pos, glm::vec3 * posCopy, glm::vec3 * vel, glm::vec3 * velCopy, int * particleArray) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    int origIndx = particleArray[index];

    posCopy[index] = pos[origIndx];
    velCopy[index] = vel[origIndx];
}

// writing above arrays back into originals to keep rest of logic consistent
__global__ void placeBackArray(int N, glm::vec3* pos, glm::vec3* posCopy, glm::vec3* vel, glm::vec3* velCopy, int* particleArray) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    int origIndx = particleArray[index];

    pos[origIndx] = posCopy[index];
    vel[origIndx] = velCopy[index];
}

/*
* same as scattered but no middleman
* remove lookup for particleArrayIndicies by sorting pos & vel arrays
* based on the same sorting as gridIndicies
*/
void Boids::stepSimulationCoherentGrid(float dt) {
  // start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  //   Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
    int boidsBlocks = (numObjects - 1 + blockSize) / blockSize;

    kernComputeIndices << <boidsBlocks, threadsPerBlock >> > (numObjects,
        gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices,
        dev_particleGridIndices);
  //   Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.

    thrust::device_ptr<int> dev_thrust_particleGridIndices(dev_particleGridIndices);
    thrust::device_ptr<int> dev_thrust_particleArrayIndices(dev_particleArrayIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  //   Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices

    int cellBlocks = (gridCellCount - 1 + blockSize) / blockSize;

    kernResetIntBuffer << <cellBlocks, threadsPerBlock >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <cellBlocks, threadsPerBlock >> > (gridCellCount, dev_gridCellEndIndices, -1);

    kernIdentifyCellStartEnd << <boidsBlocks, threadsPerBlock >> > (numObjects,
        dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  //   BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED

    sortArray <<<boidsBlocks, threadsPerBlock >>> (numObjects, dev_pos, dev_pos_coherent, dev_vel1, dev_vel1_coherent, dev_particleArrayIndices);

  // Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchCoherent << <boidsBlocks, threadsPerBlock >> > (numObjects,
        gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices,
        dev_gridCellEndIndices, dev_pos_coherent, dev_vel1_coherent, dev_vel2_coherent);
  
    placeBackArray << <boidsBlocks, threadsPerBlock >> > (numObjects, dev_pos, dev_pos_coherent, dev_vel2, dev_vel2_coherent, dev_particleArrayIndices);

  //   Update positions
    kernUpdatePos << <boidsBlocks, threadsPerBlock >> > (numObjects, dt, dev_pos, dev_vel2);

  // Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
  // Ping pong afterwards because this should be the last step!! 
    glm::vec3* ptr = dev_vel2;
    dev_vel2 = dev_vel1;
    dev_vel1 = ptr;
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_pos_coherent);
  cudaFree(dev_vel1_coherent);
  cudaFree(dev_vel2_coherent);
}

void Boids::unitTest() {
  // Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
