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

// LOOK-2.1 potentially useful for doing grid-based neighbor search
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
#define blockSize 1024

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 2.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

glm::vec3* dev_pos_sorted;
glm::vec3* dev_vel1_sorted;
glm::vec3* dev_vel2_sorted;

// LOOK-2.1 - Grid parameters based on simulation parameters.
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
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
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

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  // Grid-Looping Optimization: Here change to 1¡Ádistance, Dynamic boundary calculation works better with smaller cells.
  // When cell_width = 2¡Ádistance, saves only 0-2 cells (8¡ú6-8); when cell_width = 1¡Ádistance, saves 12-19 cells (27¡ú8-15).
  // Optimization overhead is justified only when grid cells are small enough to create significant search space reduction.
  gridCellWidth = 1.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  // Setup thrust pointers for sorting
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
  cudaMalloc((void**)&dev_pos_sorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_sorted failed!");

  cudaMalloc((void**)&dev_vel1_sorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1_sorted failed!");

  cudaMalloc((void**)&dev_vel2_sorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2_sorted failed!");

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

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
    glm::vec3 rule1_velocity = glm::vec3(0.0f);
    glm::vec3 rule2_velocity = glm::vec3(0.0f);
    glm::vec3 rule3_velocity = glm::vec3(0.0f);

    glm::vec3 myPos = pos[iSelf];

    int rule1_neighbors = 0;
    int rule3_neighbors = 0;

    // Loop through all other boids
    for (int i = 0; i < N; i++) {
        if (i == iSelf) continue; // Skip self

        glm::vec3 otherPos = pos[i];
        float distance = glm::length(myPos - otherPos);

        // Rule 1: Cohesion - boids fly towards their local perceived center of mass
        if (distance < rule1Distance) {
            rule1_velocity += otherPos;
            rule1_neighbors++;
        }

        // Rule 2: Separation - boids try to stay a distance d away from each other
        if (distance < rule2Distance) {
            rule2_velocity -= (otherPos - myPos);
        }

        // Rule 3: Alignment - boids try to match velocity with near boids
        if (distance < rule3Distance) {
            rule3_velocity += vel[i];
            rule3_neighbors++;
        }
    }

    // Apply the rules
    if (rule1_neighbors > 0) {
        rule1_velocity /= rule1_neighbors; // Get average position
        rule1_velocity = (rule1_velocity - myPos) * rule1Scale; // Direction towards center
    }

    rule2_velocity *= rule2Scale;

    if (rule3_neighbors > 0) {
        rule3_velocity /= rule3_neighbors; // Get average velocity
        rule3_velocity *= rule3Scale;
    }

    return rule1_velocity + rule2_velocity + rule3_velocity;
}


/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3* pos,
    glm::vec3* vel1, glm::vec3* vel2) {

    // Get the index of the current thread
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    // Compute velocity change based on the three rules
    glm::vec3 velocityChange = computeVelocityChange(N, index, pos, vel1);

    // Update velocity
    glm::vec3 newVelocity = vel1[index] + velocityChange;

    // Clamp the speed to maxSpeed
    float speed = glm::length(newVelocity);
    if (speed > maxSpeed) {
        newVelocity = (newVelocity / speed) * maxSpeed;
    }

    // Record the new velocity into vel2
    vel2[index] = newVelocity;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
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

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
    glm::vec3 gridMin, float inverseCellWidth,
    glm::vec3* pos, int* indices, int* gridIndices) {

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    // Set up parallel array of integer indices as pointers to the actual boid data
    indices[index] = index;

    // Label each boid with the index of its grid cell
    glm::vec3 boidPos = pos[index];

    // Convert world position to grid coordinates
    glm::vec3 gridPos = (boidPos - gridMin) * inverseCellWidth;

    // Convert to integer grid coordinates (with bounds checking)
    int gridX = (int)gridPos.x;
    int gridY = (int)gridPos.y;
    int gridZ = (int)gridPos.z;

    // Clamp to valid grid range
    gridX = gridX < 0 ? 0 : (gridX >= gridResolution ? gridResolution - 1 : gridX);
    gridY = gridY < 0 ? 0 : (gridY >= gridResolution ? gridResolution - 1 : gridY);
    gridZ = gridZ < 0 ? 0 : (gridZ >= gridResolution ? gridResolution - 1 : gridZ);

    // Convert 3D grid coordinates to 1D index
    int gridIndex = gridIndex3Dto1D(gridX, gridY, gridZ, gridResolution);

    gridIndices[index] = gridIndex;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    int currentGridIndex = particleGridIndices[index];

    // Check if this is the start of a new cell
    if (index == 0 || particleGridIndices[index - 1] != currentGridIndex) {
        gridCellStartIndices[currentGridIndex] = index;
    }

    // Check if this is the end of a cell
    if (index == N - 1 || particleGridIndices[index + 1] != currentGridIndex) {
        gridCellEndIndices[currentGridIndex] = index;
    }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2


    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    // Get current boid's data
    glm::vec3 myPos = pos[index];

    // Initialize rule velocities
    glm::vec3 rule1_velocity = glm::vec3(0.0f);
    glm::vec3 rule2_velocity = glm::vec3(0.0f);
    glm::vec3 rule3_velocity = glm::vec3(0.0f);

    int rule1_neighbors = 0;
    int rule3_neighbors = 0;

    // Extra Credit : Grid-Looping Optimization: Calculate dynamic search bounds
    float maxNeighborDistance = fmaxf(fmaxf(rule1Distance, rule2Distance), rule3Distance);

    // Calculate bounding box for neighbor search
    glm::vec3 minBound = myPos - glm::vec3(maxNeighborDistance);
    glm::vec3 maxBound = myPos + glm::vec3(maxNeighborDistance);

    // Convert to grid coordinates
    glm::vec3 minGridCoord = (minBound - gridMin) * inverseCellWidth;
    glm::vec3 maxGridCoord = (maxBound - gridMin) * inverseCellWidth;

    // Convert to integer grid indices with bounds checking
    int minGridX = max(0, (int)minGridCoord.x);
    int maxGridX = min(gridResolution - 1, (int)maxGridCoord.x);
    int minGridY = max(0, (int)minGridCoord.y);
    int maxGridY = min(gridResolution - 1, (int)maxGridCoord.y);
    int minGridZ = max(0, (int)minGridCoord.z);
    int maxGridZ = min(gridResolution - 1, (int)maxGridCoord.z);

    // Dynamic loop - only check cells that could contain neighbors
    for (int gz = minGridZ; gz <= maxGridZ; gz++) {
        for (int gy = minGridY; gy <= maxGridY; gy++) {
            for (int gx = minGridX; gx <= maxGridX; gx++) {

                int neighborGridIndex = gridIndex3Dto1D(gx, gy, gz, gridResolution);

                int startIndex = gridCellStartIndices[neighborGridIndex];
                int endIndex = gridCellEndIndices[neighborGridIndex];

                // Skip empty cells
                if (startIndex == -1 || endIndex == -1) {
                    continue;
                }

                // Check all boids in this neighboring cell
                for (int i = startIndex; i <= endIndex; i++) {
                    int neighborBoidIndex = particleArrayIndices[i];

                    if (neighborBoidIndex == index) continue; // Skip self

                    glm::vec3 otherPos = pos[neighborBoidIndex];
                    float distance = glm::length(myPos - otherPos);

                    // Apply the three rules with distance checks
                    if (distance < rule1Distance) {
                        rule1_velocity += otherPos;
                        rule1_neighbors++;
                    }

                    if (distance < rule2Distance) {
                        rule2_velocity -= (otherPos - myPos);
                    }

                    if (distance < rule3Distance) {
                        rule3_velocity += vel1[neighborBoidIndex];
                        rule3_neighbors++;
                    }
                }
            }
        }
    }

    // Apply the rules (same logic as before)
    if (rule1_neighbors > 0) {
        rule1_velocity /= rule1_neighbors;
        rule1_velocity = (rule1_velocity - myPos) * rule1Scale;
    }

    rule2_velocity *= rule2Scale;

    if (rule3_neighbors > 0) {
        rule3_velocity /= rule3_neighbors;
        rule3_velocity *= rule3Scale;
    }

    glm::vec3 velocityChange = rule1_velocity + rule2_velocity + rule3_velocity;
    glm::vec3 newVelocity = vel1[index] + velocityChange;

    // Clamp speed
    float speed = glm::length(newVelocity);
    if (speed > maxSpeed) {
        newVelocity = (newVelocity / speed) * maxSpeed;
    }

    vel2[index] = newVelocity;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    // Get current boid's data (now directly from sorted arrays)
    glm::vec3 myPos = pos[index];

    // Initialize rule velocities
    glm::vec3 rule1_velocity = glm::vec3(0.0f);
    glm::vec3 rule2_velocity = glm::vec3(0.0f);
    glm::vec3 rule3_velocity = glm::vec3(0.0f);

    int rule1_neighbors = 0;
    int rule3_neighbors = 0;

    // Extra Credit : Grid-Looping Optimization Calculate dynamic search bounds
    float maxNeighborDistance = fmaxf(fmaxf(rule1Distance, rule2Distance), rule3Distance);

    // Calculate bounding box for neighbor search
    glm::vec3 minBound = myPos - glm::vec3(maxNeighborDistance);
    glm::vec3 maxBound = myPos + glm::vec3(maxNeighborDistance);

    // Convert to grid coordinates
    glm::vec3 minGridCoord = (minBound - gridMin) * inverseCellWidth;
    glm::vec3 maxGridCoord = (maxBound - gridMin) * inverseCellWidth;

    // Convert to integer grid indices with bounds checking
    int minGridX = max(0, (int)minGridCoord.x);
    int maxGridX = min(gridResolution - 1, (int)maxGridCoord.x);
    int minGridY = max(0, (int)minGridCoord.y);
    int maxGridY = min(gridResolution - 1, (int)maxGridCoord.y);
    int minGridZ = max(0, (int)minGridCoord.z);
    int maxGridZ = min(gridResolution - 1, (int)maxGridCoord.z);

    // Dynamic loop with memory-efficient order (z-y-x for better cache locality)
    for (int gz = minGridZ; gz <= maxGridZ; gz++) {
        for (int gy = minGridY; gy <= maxGridY; gy++) {
            for (int gx = minGridX; gx <= maxGridX; gx++) {

                int neighborGridIndex = gridIndex3Dto1D(gx, gy, gz, gridResolution);

                int startIndex = gridCellStartIndices[neighborGridIndex];
                int endIndex = gridCellEndIndices[neighborGridIndex];

                // Skip empty cells
                if (startIndex == -1 || endIndex == -1) {
                    continue;
                }

                // Check all boids in this neighboring cell
                for (int i = startIndex; i <= endIndex; i++) {
                    if (i == index) continue; // Skip self

                    glm::vec3 otherPos = pos[i];
                    float distance = glm::length(myPos - otherPos);

                    // Apply the three rules with distance checks
                    if (distance < rule1Distance) {
                        rule1_velocity += otherPos;
                        rule1_neighbors++;
                    }

                    if (distance < rule2Distance) {
                        rule2_velocity -= (otherPos - myPos);
                    }

                    if (distance < rule3Distance) {
                        rule3_velocity += vel1[i];
                        rule3_neighbors++;
                    }
                }
            }
        }
    }

    // Apply the rules
    if (rule1_neighbors > 0) {
        rule1_velocity /= rule1_neighbors;
        rule1_velocity = (rule1_velocity - myPos) * rule1Scale;
    }

    rule2_velocity *= rule2Scale;

    if (rule3_neighbors > 0) {
        rule3_velocity /= rule3_neighbors;
        rule3_velocity *= rule3Scale;
    }

    glm::vec3 velocityChange = rule1_velocity + rule2_velocity + rule3_velocity;
    glm::vec3 newVelocity = vel1[index] + velocityChange;

    // Clamp speed
    float speed = glm::length(newVelocity);
    if (speed > maxSpeed) {
        newVelocity = (newVelocity / speed) * maxSpeed;
    }

    vel2[index] = newVelocity;
}




__global__ void kernReshuffleData(int N, int* particleArrayIndices,
    glm::vec3* pos_unsorted, glm::vec3* vel_unsorted,
    glm::vec3* pos_sorted, glm::vec3* vel_sorted) {

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    // Get the original boid index from the sorted array
    int originalIndex = particleArrayIndices[index];

    // Copy data from unsorted to sorted arrays
    pos_sorted[index] = pos_unsorted[originalIndex];
    vel_sorted[index] = vel_unsorted[originalIndex];
}



/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    // Update velocities using brute force neighbor search
    kernUpdateVelocityBruteForce << <fullBlocksPerGrid, threadsPerBlock >> > (
        numObjects, dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

    // Update positions based on new velocities
    kernUpdatePos << <fullBlocksPerGrid, threadsPerBlock >> > (
        numObjects, dt, dev_pos, dev_vel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    // Ping-pong the velocity buffers
    glm::vec3* temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed


    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 fullBlocksPerGridCell((gridCellCount + blockSize - 1) / blockSize);

    // Label each particle with its array index and grid index
    kernComputeIndices << <fullBlocksPerGrid, threadsPerBlock >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    //  Reset grid cell start/end indices
    kernResetIntBuffer << <fullBlocksPerGridCell, threadsPerBlock >> > (
        gridCellCount, dev_gridCellStartIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer start failed!");

    kernResetIntBuffer << <fullBlocksPerGridCell, threadsPerBlock >> > (
        gridCellCount, dev_gridCellEndIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer end failed!");

    //  Sort particles by grid index using Thrust
    thrust::sort_by_key(dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices);

    //  Identify start and end indices for each cell
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, threadsPerBlock >> > (
        numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    //  Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, threadsPerBlock >> > (
        numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
        dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

    //  Update positions
    kernUpdatePos << <fullBlocksPerGrid, threadsPerBlock >> > (
        numObjects, dt, dev_pos, dev_vel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    //  Ping-pong buffers
    glm::vec3* temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.


    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 fullBlocksPerGridCell((gridCellCount + blockSize - 1) / blockSize);

    //  Label each particle with its array index and grid index
    kernComputeIndices << <fullBlocksPerGrid, threadsPerBlock >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    //  Reset grid cell start/end indices
    kernResetIntBuffer << <fullBlocksPerGridCell, threadsPerBlock >> > (
        gridCellCount, dev_gridCellStartIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer start failed!");

    kernResetIntBuffer << <fullBlocksPerGridCell, threadsPerBlock >> > (
        gridCellCount, dev_gridCellEndIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer end failed!");

    // Sort particles by grid index using Thrust
    thrust::sort_by_key(dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices);

    // Identify start and end indices for each cell
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, threadsPerBlock >> > (
        numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    // BIG DIFFERENCE - Reshuffle the particle data for coherent access
    kernReshuffleData << <fullBlocksPerGrid, threadsPerBlock >> > (
        numObjects, dev_particleArrayIndices,
        dev_pos, dev_vel1,
        dev_pos_sorted, dev_vel1_sorted);
    checkCUDAErrorWithLine("kernReshuffleData failed!");

    // Perform velocity updates using coherent neighbor search
    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, threadsPerBlock >> > (
        numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_pos_sorted, dev_vel1_sorted, dev_vel2_sorted);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

    //  Update positions using sorted data
    kernUpdatePos << <fullBlocksPerGrid, threadsPerBlock >> > (
        numObjects, dt, dev_pos_sorted, dev_vel2_sorted);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    // Ping-pong buffers

    glm::vec3* temp_pos = dev_pos;
    dev_pos = dev_pos_sorted;
    dev_pos_sorted = temp_pos;

    glm::vec3* temp_vel = dev_vel1;
    dev_vel1 = dev_vel2_sorted;
    dev_vel2_sorted = dev_vel1_sorted;
    dev_vel1_sorted = temp_vel;
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);


  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  cudaFree(dev_pos_sorted);
  cudaFree(dev_vel1_sorted);
  cudaFree(dev_vel2_sorted);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

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
