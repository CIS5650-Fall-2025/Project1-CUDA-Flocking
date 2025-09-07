#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
// ORIGINALLY 5, 3, 5
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

// ORIGINALLY 0.01, 0.1, 0.1
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

glm::vec3* dev_sorted_pos;
glm::vec3* dev_sorted_vel1;
glm::vec3* dev_sorted_vel2;

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
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x = -halfGridWidth;
  gridMinimum.y = -halfGridWidth;
  gridMinimum.z = -halfGridWidth;

  // TODO-2.1 : Allocate buffers
  
  cudaMalloc((void**)&dev_particleArrayIndices, numObjects * sizeof(int));
  cudaMalloc((void**)&dev_particleGridIndices, numObjects * sizeof(int));

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));


  //TODO - 2.3 - Allocate additional buffers here

  cudaMalloc((void**)&dev_sorted_pos, numObjects * sizeof(glm::vec3));
  cudaMalloc((void**)&dev_sorted_vel1, numObjects * sizeof(glm::vec3));
  cudaMalloc((void**)&dev_sorted_vel2, numObjects * sizeof(glm::vec3));

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

__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
    glm::vec3 finalVel = glm::vec3(0.0f);

    glm::vec3 centerPos = glm::vec3(0.0f);
    glm::vec3 averageVel = glm::vec3(0.0f);
    glm::vec3 seperation = glm::vec3(0.0f);

    float numNeighbors1 = 0.0;
    float numNeighbors3 = 0.0;

    for (int boidIdx = 0; boidIdx < N; boidIdx++) {
        if (boidIdx != iSelf) {
            // RULE 1:
            if (glm::length(pos[boidIdx] - pos[iSelf]) < rule1Distance) {
                centerPos += pos[boidIdx];
                numNeighbors1++;
            }

            // RULE 2:
            if (glm::length(pos[boidIdx] - pos[iSelf]) < rule2Distance) {
                seperation -= (pos[boidIdx] - pos[iSelf]);
            }

            // RULE 3:
            if (glm::length(pos[boidIdx] - pos[iSelf]) < rule3Distance) {
                averageVel += vel[boidIdx];
                numNeighbors3++;
            }
        }
    }
    if (numNeighbors1 != 0) {
        centerPos /= numNeighbors1;
        finalVel += (centerPos - pos[iSelf]) * rule1Scale;
    }

    finalVel += seperation * rule2Scale;

    if (numNeighbors3 != 0) {
        averageVel /= numNeighbors3;
        finalVel += averageVel * rule3Scale;
    }

    return finalVel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1? 

    int boidIdx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (boidIdx >= N) {
        return;
    }

    vel2[boidIdx] = vel1[boidIdx] + computeVelocityChange(N, boidIdx, pos, vel1);
    if (glm::length(vel2[boidIdx]) > maxSpeed) {
        vel2[boidIdx] = glm::normalize(vel2[boidIdx]) * maxSpeed;
    }
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
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

    int boidIdx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (boidIdx >= N) {
        return;
    }

    glm::ivec3 gridIdx3D = (pos[boidIdx] - gridMin) * inverseCellWidth;

    int gridIdx1D = gridIndex3Dto1D(gridIdx3D.x, gridIdx3D.y, gridIdx3D.z, gridResolution);

    indices[boidIdx] = boidIdx;
    gridIndices[boidIdx] = gridIdx1D;

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

    // CALLED WITH N = NUM GRIP CELLS

    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= N) {
        return;
    }

    if (idx == 0 || particleGridIndices[idx] != particleGridIndices[idx - 1]) {
        gridCellStartIndices[particleGridIndices[idx]] = idx;
    }
    if (idx == N - 1 || particleGridIndices[idx] != particleGridIndices[idx + 1]) {
        gridCellEndIndices[particleGridIndices[idx]] = idx;
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
     //    - For this use pos[iSelf] calculate the corrisponding grid cell
     // - Identify which cells may contain neighbors. This isn't always 8.
     //    - Loop over 27 grid cells (account for boundaries) and check start / end 
     //      arrays. If a cell has -1 then there are no boids in it
     // - For each cell, read the start/end indices in the boid pointer array.
     //    - I cell != -1, then go through between start and end in boid array
     // - Access each boid in the cell and compute velocity change from
     //   the boids rules, if this boid is within the neighborhood distance.
     //    - Loop over all the boids you find
     // - Clamp the speed change before putting the new speed in vel2

       // STEP 1:  USE POS TO GET CURRENT CELL
    int boidIdx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (boidIdx >= N) {
        return;
    }

    glm::vec3 finalVel = glm::vec3(0.0f);
    glm::vec3 centerPos = glm::vec3(0.0f);
    glm::vec3 averageVel = glm::vec3(0.0f);
    glm::vec3 seperation = glm::vec3(0.0f);

    float numNeighbors1 = 0.0;
    float numNeighbors3 = 0.0;

    // STEP 2: LOOP OVER SURROUNDING GRID
    glm::ivec3 baseGridCell = (pos[boidIdx] - gridMin) * inverseCellWidth;
    glm::ivec3 curGridCell = baseGridCell;
    int curGridIdx1D;

    for (int xOff = -1; xOff <= 1; xOff++) {
        for (int yOff = -1; yOff <= 1; yOff++) {
            for (int zOff = -1; zOff <= 1; zOff++) {
                curGridCell.x = baseGridCell.x + xOff;
                curGridCell.y = baseGridCell.y + yOff;
                curGridCell.z = baseGridCell.z + zOff;

                if (curGridCell.x < 0 || curGridCell.x >= gridResolution ||
                    curGridCell.y < 0 || curGridCell.y >= gridResolution ||
                    curGridCell.z < 0 || curGridCell.z >= gridResolution) {
                    continue;
                }

                curGridIdx1D = gridIndex3Dto1D(curGridCell.x, curGridCell.y, curGridCell.z, gridResolution);

                // CHECK IF NO BOIDS ARE IN CELL
                if (gridCellStartIndices[curGridIdx1D] == -1 || gridCellEndIndices[curGridIdx1D] == -1) {
                    continue;
                }

                int sortedIdx = 0;
                // FOR EACH BOID IN THIS CELL (start -> end idx) do the three rules
                for (int idx = gridCellStartIndices[curGridIdx1D]; idx <= gridCellEndIndices[curGridIdx1D]; idx++) {
                    sortedIdx = particleArrayIndices[idx];
                    if (sortedIdx != boidIdx) {
                        // RULE 1:
                        if (glm::length(pos[sortedIdx] - pos[boidIdx]) < rule1Distance) {
                            centerPos += pos[sortedIdx];
                            numNeighbors1++;
                        }
                        // RULE 2:
                        if (glm::length(pos[sortedIdx] - pos[boidIdx]) < rule2Distance) {
                            seperation -= (pos[sortedIdx] - pos[boidIdx]);
                        }
                        // RULE 3:
                        if (glm::length(pos[sortedIdx] - pos[boidIdx]) < rule3Distance) {
                            averageVel += vel1[sortedIdx];
                            numNeighbors3++;
                        }
                    }
                }
            }
        }
    }

    // RULE 1:
    if (numNeighbors1 != 0) {
        centerPos /= numNeighbors1;
        finalVel += (centerPos - pos[boidIdx]) * rule1Scale;
    }

    // RULE 2:
    finalVel += seperation * rule2Scale;

    // RULE 3:
    if (numNeighbors3 != 0) {
        averageVel /= numNeighbors3;
        finalVel += averageVel * rule3Scale;
    }

    finalVel += vel1[boidIdx];

    if (glm::length(finalVel) > maxSpeed) {
        finalVel = glm::normalize(finalVel) * maxSpeed;
    }

    // UPDATE VEL2 WITH NEW CALCULATED VEL
    vel2[boidIdx] = finalVel;
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

    int boidIdx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (boidIdx >= N) {
        return;
    }

    glm::vec3 finalVel = glm::vec3(0.0f);
    glm::vec3 centerPos = glm::vec3(0.0f);
    glm::vec3 averageVel = glm::vec3(0.0f);
    glm::vec3 seperation = glm::vec3(0.0f);

    float numNeighbors1 = 0.0;
    float numNeighbors3 = 0.0;

    // STEP 2: LOOP OVER SURROUNDING GRID
    glm::ivec3 baseGridCell = (pos[boidIdx] - gridMin) * inverseCellWidth;
    glm::ivec3 curGridCell = baseGridCell;
    int curGridIdx1D;

    for (int xOff = -1; xOff <= 1; xOff++) {
        for (int yOff = -1; yOff <= 1; yOff++) {
            for (int zOff = -1; zOff <= 1; zOff++) {
                curGridCell.x = baseGridCell.x + xOff;
                curGridCell.y = baseGridCell.y + yOff;
                curGridCell.z = baseGridCell.z + zOff;

                if (curGridCell.x < 0 || curGridCell.x >= gridResolution ||
                    curGridCell.y < 0 || curGridCell.y >= gridResolution ||
                    curGridCell.z < 0 || curGridCell.z >= gridResolution) {
                    continue;
                }

                curGridIdx1D = gridIndex3Dto1D(curGridCell.x, curGridCell.y, curGridCell.z, gridResolution);

                // CHECK IF NO BOIDS ARE IN CELL
                if (gridCellStartIndices[curGridIdx1D] == -1 || gridCellEndIndices[curGridIdx1D] == -1) {
                    continue;
                }


                // FOR EACH BOID IN THIS CELL (start -> end idx) do the three rules
                for (int idx = gridCellStartIndices[curGridIdx1D]; idx <= gridCellEndIndices[curGridIdx1D]; idx++) {
                    if (idx != boidIdx) {
                        // RULE 1:
                        if (glm::length(pos[idx] - pos[boidIdx]) < rule1Distance) {
                            centerPos += pos[idx];
                            numNeighbors1++;
                        }
                        // RULE 2:
                        if (glm::length(pos[idx] - pos[boidIdx]) < rule2Distance) {
                            seperation -= (pos[idx] - pos[boidIdx]);
                        }
                        // RULE 3:
                        if (glm::length(pos[idx] - pos[boidIdx]) < rule3Distance) {
                            averageVel += vel1[idx];
                            numNeighbors3++;
                        }
                    }
                }
            }
        }
    }

    // RULE 1:
    if (numNeighbors1 != 0) {
        centerPos /= numNeighbors1;
        finalVel += (centerPos - pos[boidIdx]) * rule1Scale;
    }

    // RULE 2:
    finalVel += seperation * rule2Scale;

    // RULE 3:
    if (numNeighbors3 != 0) {
        averageVel /= numNeighbors3;
        finalVel += averageVel * rule3Scale;
    }

    finalVel += vel1[boidIdx];

    if (glm::length(finalVel) > maxSpeed) {
        finalVel = glm::normalize(finalVel) * maxSpeed;
    }

    // UPDATE VEL2 WITH NEW CALCULATED VEL
    vel2[boidIdx] = finalVel;

}

__global__ void kernCopyByIndex(int N, int* particleArrayIndices,
                glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2,
                glm::vec3* sorted_pos, glm::vec3* sorted_vel1, glm::vec3* sorted_vel2) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
        return;
    }
    int sortedIdx = particleArrayIndices[index];

    sorted_pos[index] = pos[sortedIdx];
    sorted_vel1[index] = vel1[sortedIdx];
    sorted_vel2[index] = vel2[sortedIdx];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers

    int gridSize = (numObjects + blockSize - 1) / blockSize;
    kernUpdateVelocityBruteForce<<<gridSize, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos<<<gridSize, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

    std::swap(dev_vel1, dev_vel2);
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

    int gridSizeBoids = (numObjects + blockSize - 1) / blockSize;
    int gridSizeCells = (gridCellCount + blockSize - 1) / blockSize; 


    // RESET START AND END BUFFERS TO -1
    kernResetIntBuffer << <gridSizeCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <gridSizeCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    // COMPUTE GRID / BOID INDEXING
    kernComputeIndices << <gridSizeBoids, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
                                                          dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // SORT BY KEY
    dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);

    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

    // FIND START AND END IDXS (each thread is a cell)
    kernIdentifyCellStartEnd << < gridSizeBoids, blockSize >> > (numObjects, dev_particleGridIndices,
                                                                 dev_gridCellStartIndices, dev_gridCellEndIndices);

    // UPDATE VELOCITY BASED ON 3 RULES USING GRID
    kernUpdateVelNeighborSearchScattered << <gridSizeBoids, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
                                                                                dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
                                                                                dev_pos, dev_vel1, dev_vel2);

    // UPDATE POSITIONS 
    kernUpdatePos << <gridSizeBoids, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

    // PING-PONG VEL BUFFERS
    std::swap(dev_vel1, dev_vel2);
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

    int gridSizeBoids = (numObjects + blockSize - 1) / blockSize;
    int gridSizeCells = (gridCellCount + blockSize - 1) / blockSize;


    // RESET START AND END BUFFERS TO -1
    kernResetIntBuffer << <gridSizeCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <gridSizeCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    // COMPUTE GRID / BOID INDEXING
    kernComputeIndices << <gridSizeBoids, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // SORT BY KEY
    dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);

    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

    // REORDER POS AND VEL DATA
    kernCopyByIndex << <gridSizeBoids, blockSize >> > (numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2,
                                                        dev_sorted_pos, dev_sorted_vel1, dev_sorted_vel2);

    // FIND START AND END IDXS (each thread is a cell)
    kernIdentifyCellStartEnd << < gridSizeBoids, blockSize >> > (numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices, dev_gridCellEndIndices);

    // UPDATE VELOCITY BASED ON 3 RULES USING GRID
    kernUpdateVelNeighborSearchCoherent << <gridSizeBoids, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_sorted_pos, dev_sorted_vel1, dev_sorted_vel2);

    // UPDATE POSITIONS 
    kernUpdatePos << <gridSizeBoids, blockSize >> > (numObjects, dt, dev_sorted_pos, dev_sorted_vel2);


    kernCopyByIndex << <gridSizeBoids, blockSize >> > (numObjects, dev_particleArrayIndices, dev_sorted_pos, dev_sorted_vel1, dev_sorted_vel2,
        dev_pos, dev_vel1, dev_vel2);

    // PING-PONG VEL BUFFERS
    std::swap(dev_vel1, dev_vel2);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.

  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);

  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  cudaFree(dev_sorted_pos);
  cudaFree(dev_sorted_vel1);
  cudaFree(dev_sorted_vel2);
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
