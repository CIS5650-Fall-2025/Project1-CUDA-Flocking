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

/**
* Runtime check for kernels function.
**/
#define CUDA_CHECK(err) { gpuAssert((err), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
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
glm::vec3* dev_reshuffledPos;
glm::vec3* dev_reshuffledVel1;
glm::vec3* dev_reshuffledVel2;

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
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaDeviceSynchronize();

  // Pointer from boid to data in pos, vel1, and vel2 arrays.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);

  // Pointer from grid index to boid.
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);

  // Pointer from cell to start index in dev_particleArrayIndices.
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  // Pointer from cell to end index in dev_particleArrayIndices.
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_reshuffledPos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_coherent failed!");

  cudaMalloc((void**)&dev_reshuffledVel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1_coherent failed!");

  cudaMalloc((void**)&dev_reshuffledVel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2_coherent failed!");
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
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids
    glm::vec3 perceived_center = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 c = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 perceived_velocity = glm::vec3(0.f, 0.f, 0.f);

    glm::vec3 currBoidPos = pos[iSelf];
    glm::vec3 currBoidVel = vel[iSelf];

    // Count num neighbours.
    int numNeighboursRule1 = 0;
    int numNeighboursRule3 = 0;

    // N = num boids.
    for (int i = 0; i < N; i++)
    {
        if (i == iSelf)
        {
            continue;
        }

        // Get dist for every other boid and curr boid.
        float dist = glm::distance(pos[i], currBoidPos);
        // Rule 1 calc.
        if (dist < rule1Distance)
        {
            perceived_center += pos[i];
            numNeighboursRule1++;
        }
        //Rule 2 calc.
        if (dist < rule2Distance)
        {
            c -= pos[i] - currBoidPos;
        }
        // Rule 3 calc.
        if (dist < rule3Distance)
        {
            perceived_velocity += vel[i];
            numNeighboursRule3++;
        }
    }

    // Process Rules.
    glm::vec3 resultBoids = glm::vec3(0.f, 0.f, 0.f);
    // Rule 1:
    if (numNeighboursRule1 > 0)
    {
        perceived_center /= numNeighboursRule1;
        resultBoids += (perceived_center - currBoidPos) * rule1Scale;
    }
    // Rule 2:
    resultBoids += c * rule2Scale;
    // Rule 3:
    if (numNeighboursRule3 > 0)
    {
        perceived_velocity /= numNeighboursRule3;
        resultBoids += perceived_velocity * rule3Scale;
    }

    return resultBoids;
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
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    glm::vec3 velChange = computeVelocityChange(N, index, pos, vel1);
    glm::vec3 newVelocity = vel1[index] + velChange;

    // Clamp speed.
    float speed = glm::length(newVelocity);
    if (speed > maxSpeed)
    {
        newVelocity = glm::normalize(newVelocity) * maxSpeed;
    }

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
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= N) {
        return;
    }

    indices[index] = index;

    glm::vec3 gridPos = (pos[index] - gridMin) * inverseCellWidth;

    int index1D = gridIndex3Dto1D(gridPos.x, gridPos.y, gridPos.z, gridResolution);

    // Parallel array indices. 
    gridIndices[index] = index1D;
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

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) {
        return;
    }

    int currBoidGridIndex = particleGridIndices[index];

    // Set up start and end boid.
    if (index == 0) {
        gridCellStartIndices[currBoidGridIndex] = 0;
    }
    else if (index == N - 1) {
        gridCellEndIndices[currBoidGridIndex] = N - 1;
    }

    // The rest of the boids.
    else {
        int prevBoidGridIndex = particleGridIndices[index - 1];

        if (currBoidGridIndex != prevBoidGridIndex) {
            // Start of current boid and end of previous boid.
            gridCellStartIndices[currBoidGridIndex] = index;
            gridCellEndIndices[prevBoidGridIndex] = index - 1;
        }
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
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) {
        return;
    }

    glm::vec3 perceived_center = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 c = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 perceived_velocity = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 resultBoids = glm::vec3(0.f, 0.f, 0.f);
    int numNeighboursRule1 = 0;
    int numNeighboursRule3 = 0;

    int boidIdx = particleArrayIndices[index];
    glm::vec3 currPos = pos[index];
    glm::vec3 currVel = vel1[index];
    glm::vec3 gridPos = glm::floor((currPos - gridMin) * inverseCellWidth);

    for (int x = imax(gridPos.x - 1, 0); x < imin(gridPos.x + 1, gridResolution - 1); x++) {
        for (int y = imax(gridPos.y - 1, 0); y < imin(gridPos.y + 1, gridResolution - 1); y++) {
            for (int z = imax(gridPos.z - 1, 0); z < imin(gridPos.z + 1, gridResolution - 1); z++) {
                int currGridIdx = gridIndex3Dto1D(x, y, z, gridResolution);
                if (currGridIdx < 0 || currGridIdx >= pow(gridResolution, 3)) {
                    continue;
                }

                int startIdx = gridCellStartIndices[currGridIdx];
                int endIdx = gridCellEndIndices[currGridIdx];

                if (startIdx == -1 || endIdx == -1)
                {
                    continue;
                }

                for (int i = startIdx; i <= endIdx; i++) {
                    // Current neighbouring cell.
                    int currBoidIdx = particleArrayIndices[i];

                    if (currBoidIdx == boidIdx) {
                        continue;
                    }

                    // Get dist for every other boid and curr boid.
                    float dist = glm::distance(pos[currBoidIdx], currPos);
                    // Rule 1 calc.
                    if (dist < rule1Distance)
                    {
                        perceived_center += pos[currBoidIdx];
                        numNeighboursRule1++;
                    }
                    //Rule 2 calc.
                    if (dist < rule2Distance)
                    {
                        c -= pos[currBoidIdx] - currPos;
                    }
                    // Rule 3 calc.
                    if (dist < rule3Distance)
                    {
                        perceived_velocity += vel1[currBoidIdx];
                        numNeighboursRule3++;
                    }
                }
            }
        }
    }

    // Rule 1:
    if (numNeighboursRule1 > 0)
    {
        perceived_center /= numNeighboursRule1;
        resultBoids += (perceived_center - currPos) * rule1Scale;
    }
    // Rule 2:

    resultBoids += c * rule2Scale;
    // Rule 3:
    if (numNeighboursRule3 > 0)
    {
        perceived_velocity /= numNeighboursRule3;
        resultBoids += perceived_velocity * rule3Scale;
    }

    resultBoids += currVel;
    float speed = glm::length(resultBoids);
    if (speed > maxSpeed) {
        resultBoids = glm::normalize(resultBoids) * maxSpeed;
    }

    vel2[index] = resultBoids;
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
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) {
        return;
    }

    glm::vec3 perceived_center = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 c = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 perceived_velocity = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 resultBoids = glm::vec3(0.f, 0.f, 0.f);
    int numNeighboursRule1 = 0;
    int numNeighboursRule3 = 0;

    glm::vec3 currPos = pos[index];
    glm::vec3 gridPos = glm::floor((currPos - gridMin) * inverseCellWidth);

    for (int x = imax(gridPos.x - 1, 0); x < imin(gridPos.x + 1, gridResolution - 1); x++) {
        for (int y = imax(gridPos.y - 1, 0); y < imin(gridPos.y + 1, gridResolution - 1); y++) {
            for (int z = imax(gridPos.z - 1, 0); z < imin(gridPos.z + 1, gridResolution - 1); z++) {
                int currGridIdx = gridIndex3Dto1D(x, y, z, gridResolution);

                if (currGridIdx < 0 || currGridIdx >= pow(gridResolution, 3)) {
                    continue;
                }

                int startIdx = gridCellStartIndices[currGridIdx];
                int endIdx = gridCellEndIndices[currGridIdx];

                if (startIdx == -1 || endIdx == -1)
                {
                    continue;
                }

                for (int i = startIdx; i <= endIdx; i++) {
                    // I is current neighbour cell's index.
                    if (i == index) {
                        continue;
                    }

                    // Get dist for every other boid and curr boid.
                    float dist = glm::distance(currPos, pos[i]);
                    // Rule 1 calc.
                    if (dist < rule1Distance)
                    {
                        perceived_center += pos[i];
                        numNeighboursRule1++;
                    }
                    //Rule 2 calc.
                    if (dist < rule2Distance)
                    {
                        c -= pos[i] - currPos;
                    }
                    // Rule 3 calc.
                    if (dist < rule3Distance)
                    {
                        perceived_velocity += vel1[i];
                        numNeighboursRule3++;
                    }
                }
            }
        }
    }

    // Rule 1:
    if (rule1Distance > 0.f && numNeighboursRule1 > 0)
    {
        perceived_center /= numNeighboursRule1;
        resultBoids += (perceived_center - currPos) * rule1Scale;
    }
    // Rule 2:
    if (rule2Distance > 0.f)
    {
        resultBoids += c * rule2Scale;
    }
    // Rule 3:
    if (rule3Distance > 0.f && numNeighboursRule3 > 0)
    {
        perceived_velocity /= numNeighboursRule3;
        resultBoids += perceived_velocity * rule3Scale;
    }

    resultBoids += vel1[index];
    float speed = glm::length(resultBoids);
    if (speed > maxSpeed) {
        resultBoids = glm::normalize(resultBoids) * maxSpeed;
    }

    vel2[index] = resultBoids;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
    
    // TIMER FOR RUNTIME TESTING.
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start timing.
    CUDA_CHECK(cudaEventRecord(start));

    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    // Time naive neighbour search.
    cudaEventRecord(start);

    kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("kernUpdateVelocityBruteForce = %f ms\n", milliseconds);

    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
    cudaDeviceSynchronize();

    std::swap(dev_vel1, dev_vel2);

    // Stop timing.
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("[Naive] stepSimulation time = %f ms\n", milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
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

    // TIMER FOR RUNTIME TESTING.
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start timing.
    CUDA_CHECK(cudaEventRecord(start));

    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 fullCellsPerGrid((gridCellCount + blockSize - 1) / blockSize);

    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum
        , gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    // Step 2: Sort particles by grid index.
    // Sort the boids based on the grid cell they are in using thrust::sort_by_key.
    thrust::device_ptr<int> dev_key(dev_particleGridIndices);
    thrust::device_ptr<int> dev_val(dev_particleArrayIndices);
    thrust::sort_by_key(dev_key, dev_key + numObjects, dev_val);

    // Step 3: Reset grid cell start/end index buffers.
    kernResetIntBuffer << <fullCellsPerGrid, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <fullCellsPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    // Step 4: Identify start and end indices of grid cells.
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    // Time scattered neighbour search.
    cudaEventRecord(start);

    // Step 5: Update velocities using neighbour search.
    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount,
        gridMinimum, gridInverseCellWidth,
        gridCellWidth, dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        dev_particleArrayIndices,
        dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("kernUpdateVelNeighborSearchScattered = %f ms\n", milliseconds);

    // Step 6: Update positions based on new velocities.
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    glm::vec3* temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;

    // Stop timing.
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("[Scattered] stepSimulation time = %f ms\n", milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Reshuffle pos and vel data.
__global__ void kernReshuffle(int N, int* particleArrayIndices, glm::vec3* pos, glm::vec3* vel, glm::vec3* reshufflePos, glm::vec3* reshuffleVel) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    if (index >= N) {
        return;
    }

    int reshuffleIdx = particleArrayIndices[index];
    reshufflePos[index] = pos[reshuffleIdx];
    reshuffleVel[index] = vel[reshuffleIdx];
}

// Restore reshuffled pos.
__global__ void kernRestore(int N, int* particleArrayIndices, glm::vec3* pos, glm::vec3* vel, glm::vec3* reshufflePos, glm::vec3* reshuffleVel) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    if (index >= N) {
        return;
    }

    int restoreIdx = particleArrayIndices[index];
    pos[restoreIdx] = reshufflePos[index];
    vel[restoreIdx] = reshuffleVel[index];
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

    // TIMER FOR RUNTIME TESTING.
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start timing.
    CUDA_CHECK(cudaEventRecord(start));

    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 fullCellsPerGrid((gridCellCount + blockSize - 1) / blockSize);

    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum
        , gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    // Step 2: Sort particles by grid index. Same as 2.1.
    thrust::device_ptr<int> dev_key(dev_particleGridIndices);
    thrust::device_ptr<int> dev_val(dev_particleArrayIndices);
    thrust::sort_by_key(dev_key, dev_key + numObjects, dev_val);

    // Step 3: Reset grid cell start/end index buffers.
    kernResetIntBuffer << <fullCellsPerGrid, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <fullCellsPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    // Step 4: Identify start and end indices of grid cells.
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    // Step 5: Reshuffle data in particles array..
    kernReshuffle << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_reshuffledPos, dev_reshuffledVel1);

    // Time scattered neighbour search.
    cudaEventRecord(start);

    // Step 6: Update velocities using neighbor search.
    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount,
        gridMinimum, gridInverseCellWidth,
        gridCellWidth, dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        dev_reshuffledPos, dev_reshuffledVel1, dev_reshuffledVel2);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("kernUpdateVelNeighborSearchCoherent = %f ms\n", milliseconds);

    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_reshuffledPos, dev_reshuffledVel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    // Step 7: Restore reshuffled coherent positions.
    kernRestore << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleArrayIndices, dev_pos, dev_vel2, dev_reshuffledPos, dev_reshuffledVel2);

    glm::vec3* temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;

    // Stop timing.
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("[Coherent] stepSimulation time = %f ms\n", milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
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
  cudaFree(dev_reshuffledPos);
  cudaFree(dev_reshuffledVel1);
  cudaFree(dev_reshuffledVel2);
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
