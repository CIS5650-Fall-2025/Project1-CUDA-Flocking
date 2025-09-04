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
#include <thrust/gather.h>

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
glm::vec3* dev_sortedPos;
glm::vec3* dev_sortedVel1;
glm::vec3* dev_sortedVel2;
// thrust for the above buffers
thrust::device_ptr<glm::vec3> dev_thrust_particlePosIndices;
thrust::device_ptr<glm::vec3> dev_thrust_particleVel1Indices;


// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;
// extra param added
float maxRuleDist;

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
  maxRuleDist = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  gridCellWidth = 2.0f * maxRuleDist;
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

  // 2.3 buffers
  cudaMalloc((void**)&dev_sortedPos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_sortedPos failed!");
  cudaMalloc((void**)&dev_sortedVel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_sortedVel1 failed!");
  cudaMalloc((void**)&dev_sortedVel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_sortedVel2 failed!");

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

__device__ glm::vec3 rule1(int N, int iSelf, const glm::vec3* pos)
{
    glm::vec3 perceived_center;         // 0, 0, 0 atm, but maybe should be smth else?
    int num_neighbors = 0;

    for (int i = 0; i < N; ++i)
    {
        if (i != iSelf && glm::distance(pos[i], pos[iSelf]) < rule1Distance)
        {
            perceived_center += pos[i];
            ++num_neighbors;
        }
    }
    if (num_neighbors == 0) return glm::vec3(0);
    perceived_center /= num_neighbors;
    return (perceived_center - pos[iSelf]) * rule1Scale;
}

__device__ glm::vec3 rule2(int N, int iSelf, const glm::vec3* pos)
{
    glm::vec3 c;
    for (int i = 0; i < N; ++i)
    {
        if (iSelf != i && glm::distance(pos[i], pos[iSelf]) < rule2Distance)
        {
            c -= (pos[i] - pos[iSelf]);
        }
    }

    return c * rule2Scale;
}

__device__ glm::vec3 rule3(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel)
{
    glm::vec3 perceived_vel;
    int num_neighbors = 0;

    for (int i = 0; i < N; ++i)
    {
        if (i != iSelf && glm::distance(pos[i], pos[iSelf]) < rule3Distance)
        {
            perceived_vel += vel[i];
            ++num_neighbors;
        }
    }
    if (num_neighbors == 0) return glm::vec3(0);
    perceived_vel /= num_neighbors;
    return perceived_vel * rule3Scale;
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
    glm::vec3 vel_change(0);
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    vel_change += rule1(N, iSelf, pos);
  // Rule 2: boids try to stay a distance d away from each other
    vel_change += rule2(N, iSelf, pos);
  // Rule 3: boids try to match the speed of surrounding boids
    vel_change += rule3(N, iSelf, pos, vel);
  return vel_change;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3* pos,
    glm::vec3* vel1, glm::vec3* vel2) {
    // Compute a new velocity based on pos and vel1
    // Clamp the speed
    // Record the new velocity into vel2. Question: why NOT vel1?
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    glm::vec3 new_vel = vel1[index] + computeVelocityChange(N, index, pos, vel1);
    new_vel = new_vel.length() > maxSpeed ? glm::normalize(new_vel) * maxSpeed : new_vel;
    vel2[index] = new_vel;
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
    if (index < N)
    {
        indices[index] = index;
        // we care about relative position from the start of the grid, then bucket based on cell width
        glm::vec3 relativePos = (pos[index] - gridMin) * inverseCellWidth;
        gridIndices[index] = gridIndex3Dto1D(relativePos.x, relativePos.y, relativePos.z, gridResolution);
    }
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
   if (index >= N) return;
   int thisGrid = particleGridIndices[index];

   if (index > 0 && index < N - 1 && thisGrid != particleGridIndices[index - 1])
   {
       gridCellStartIndices[thisGrid] = index;
       gridCellEndIndices[thisGrid - 1] = index - 1;
   }
   else if (index == 0)
   {
       gridCellStartIndices[thisGrid] = index;
   }
   else if (index == N - 1)
   {
       gridCellEndIndices[thisGrid] = index;
   }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2, float maxRuleDist) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
    int tIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tIdx >= N) return;
    int currBoid = particleArrayIndices[tIdx];
    glm::vec3 relativePos = glm::vec3((pos[currBoid] - gridMin) * inverseCellWidth);
  // - Identify which cells may contain neighbors. This isn't always 8.

    // vars for the rules
    int numNeighbors1 = 0;      // different neighbor sum for rules 1 and 3 in case their neighbor dist is different
    int numNeighbors3 = 0;
    glm::vec3 perceived_center(0.f), c(0.f), perceived_vel(0.f);

    float cellOffset = maxRuleDist * inverseCellWidth;

    int minX = imax((int)relativePos.x - cellOffset, 0);
    int minY = imax((int)relativePos.y - cellOffset, 0);
    int minZ = imax((int)relativePos.z - cellOffset, 0);
    int maxX = imin((int)relativePos.x + cellOffset, gridResolution - 1);
    int maxY = imin((int)relativePos.y + cellOffset, gridResolution - 1);
    int maxZ = imin((int)relativePos.z + cellOffset, gridResolution - 1);

    for (int z = minZ; z <= maxZ; ++z)
    {
        for (int y = minY; y <= maxY; ++y)
        {
            for (int x = minX; x <= maxX; ++x)
            {
                glm::ivec3 neighborPos = glm::ivec3(x, y, z);
                int neighborGrid = gridIndex3Dto1D(neighborPos.x, neighborPos.y, neighborPos.z, gridResolution);
                // - For each cell, read the start/end indices in the boid pointer array.
                // - Access each boid in the cell and compute velocity change from
                //   the boids rules, if this boid is within the neighborhood distance.
                int start = gridCellStartIndices[neighborGrid];
                int end = gridCellEndIndices[neighborGrid];
                if (start < 0 || end < 0) continue;
                for (int i = start; i <= end; ++i)
                {
                    int boid = particleArrayIndices[i];
                    if (boid != currBoid) // don't want to check this boid
                    {
                        float dist = glm::distance(pos[currBoid], pos[boid]);
                        if (dist < rule1Distance)
                        {
                            perceived_center += pos[boid];
                            ++numNeighbors1;
                        }
                        if (dist < rule2Distance)
                        {
                            c -= (pos[boid] - pos[currBoid]);
                        }
                        if (dist < rule3Distance)
                        {
                            perceived_vel += vel1[boid];
                            ++numNeighbors3;
                        }
                    }
                }
            }
        }
    }
    // rule adjustments and summing up rule results
    glm::vec3 newVel = vel1[currBoid];
    if (numNeighbors1 > 0)
    {
        perceived_center /= numNeighbors1;
        newVel += (perceived_center - pos[currBoid]) * rule1Scale;
    }
    newVel += c * rule2Scale;
    if (numNeighbors3 > 0)
    {
        perceived_vel /= numNeighbors3;
        newVel += perceived_vel * rule3Scale;
    }
  // - Clamp the speed change before putting the new speed in vel2
    newVel = newVel.length() > maxSpeed ? glm::normalize(newVel) * maxSpeed : newVel;
    vel2[currBoid] = newVel;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2, float maxRuleDist) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= N) return;
    glm::vec3 relativePos = glm::vec3((pos[idx] - gridMin) * inverseCellWidth);
  // - Identify which cells may contain neighbors. This isn't always 8.
    // vars for the rules
    int numNeighbors1 = 0;      // different neighbor sum for rules 1 and 3 in case their neighbor dist is different
    int numNeighbors3 = 0;
    glm::vec3 perceived_center(0.f), c(0.f), perceived_vel(0.f);

    float cellOffset = maxRuleDist * inverseCellWidth;

    int minX = imax((int)relativePos.x - cellOffset, 0);
    int minY = imax((int)relativePos.y - cellOffset, 0);
    int minZ = imax((int)relativePos.z - cellOffset, 0);
    int maxX = imin((int)relativePos.x + cellOffset, gridResolution - 1);
    int maxY = imin((int)relativePos.y + cellOffset, gridResolution - 1);
    int maxZ = imin((int)relativePos.z + cellOffset, gridResolution - 1);
    for (int z = minZ; z <= maxZ; ++z)
    {
        for (int y = minY; y <= maxY; ++y)
        {
            for (int x = minX; x <= maxX; ++x)
            {
                glm::ivec3 neighborPos = glm::ivec3(x, y, z);
                int neighborGrid = gridIndex3Dto1D(neighborPos.x, neighborPos.y, neighborPos.z, gridResolution);
                // - For each cell, read the start/end indices in the boid pointer array.
                //   DIFFERENCE: For best results, consider what order the cells should be
                //   checked in to maximize the memory benefits of reordering the boids data.
                int start = gridCellStartIndices[neighborGrid];
                int end = gridCellEndIndices[neighborGrid];
                if (start < 0 || end < 0) continue;

                // - Access each boid in the cell and compute velocity change from
                for (int boid = start; boid <= end; ++boid)
                {
                    if (boid != idx) // don't want to check this boid
                    {
                        float dist = glm::distance(pos[idx], pos[boid]);
                        if (dist < rule1Distance)
                        {
                            perceived_center += pos[boid];
                            ++numNeighbors1;
                        }
                        if (dist < rule2Distance)
                        {
                            c -= (pos[boid] - pos[idx]);
                        }
                        if (dist < rule3Distance)
                        {
                            perceived_vel += vel1[boid];
                            ++numNeighbors3;
                        }
                    }
                }
            }
        }
    }
  //   the boids rules, if this boid is within the neighborhood distance.
    glm::vec3 newVel = vel1[idx];
    if (numNeighbors1 > 0)
    {
        perceived_center /= numNeighbors1;
        newVel += (perceived_center - pos[idx]) * rule1Scale;
    }
    newVel += c * rule2Scale;
    if (numNeighbors3 > 0)
    {
        perceived_vel /= numNeighbors3;
        newVel += perceived_vel * rule3Scale;
    }
  // - Clamp the speed change before putting the new speed in vel2
    newVel = newVel.length() > maxSpeed ? glm::normalize(newVel) * maxSpeed : newVel;
    vel2[idx] = newVel;
}

// Helper kernel to sort our positions and velocities. This method worked over thrust
__global__ void kernelSortArrays(int N, glm::vec3* pos, glm::vec3* sorted_pos, glm::vec3* vel, glm::vec3* sorted_vel, int* particleArrayIndices) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < N) {
        sorted_pos[idx] = pos[particleArrayIndices[idx]];
        sorted_vel[idx] = vel[particleArrayIndices[idx]];
    }
}
// Helper kernel to unsort our positions, making it work for the next loop
__global__ void kernelUnsortArrays(int N, glm::vec3* pos, glm::vec3* sorted_pos, glm::vec3* vel, glm::vec3* sorted_vel, int* particleArrayIndices) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < N) {
        pos[particleArrayIndices[idx]] = sorted_pos[idx];
        vel[particleArrayIndices[idx]] = sorted_vel[idx];
    }
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
    kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  // TODO-1.2 ping-pong the velocity buffers
    cudaMemcpy(dev_vel1, dev_vel2, sizeof(float) * 3 * numObjects, cudaMemcpyDeviceToDevice);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    // Wrap device vectors in thrust iterators for use with thrust.
    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
    // first we tag all empty cells as -1
    kernResetIntBuffer << <((gridCellCount + blockSize - 1) / blockSize), blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    //kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1); // no need to also do end, sinze start -1 already signals
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
                                                                                dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2, maxRuleDist);
  // - Update positions
    kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  // - Ping-pong buffers as needed
    cudaMemcpy(dev_vel1, dev_vel2, sizeof(float) * 3 * numObjects, cudaMemcpyDeviceToDevice);
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grid
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
    // first we tag all empty cells as -1
    kernResetIntBuffer << <((gridCellCount + blockSize - 1) / blockSize), blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    //kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
    kernelSortArrays << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_sortedPos, dev_vel1, dev_sortedVel1, dev_particleArrayIndices);
  // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
                                                                                dev_gridCellStartIndices, dev_gridCellEndIndices, dev_sortedPos, dev_sortedVel1, dev_sortedVel2, maxRuleDist);
  // - Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_sortedPos, dev_sortedVel2);
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
    kernelUnsortArrays << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_sortedPos, dev_vel2, dev_sortedVel2, dev_particleArrayIndices);
    cudaMemcpy(dev_vel1, dev_vel2, sizeof(float) * 3 * numObjects, cudaMemcpyDeviceToDevice);
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

  cudaFree(dev_sortedPos);
  cudaFree(dev_sortedVel1);
  cudaFree(dev_sortedVel2);
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