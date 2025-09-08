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

// CUSTOM: Simple helper function to calculate distance squared.
__device__ float distance2(glm::vec3 a, glm::vec3 b)
{
  return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]);
}

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef imin
#define imin(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAError(const char *msg, int line = -1)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    if (line >= 0)
    {
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

#define maxSpeed 0.5f

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
int *dev_particleGridIndices;  // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

glm::vec3 *dev_posCoherent;
glm::vec3 *dev_vel1Coherent;

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

__host__ __device__ unsigned int hash(unsigned int a)
{
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
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index)
{
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
 * LOOK-1.2 - This is a basic CUDA kernel.
 * CUDA kernel for generating boids with a specified mass randomly around the star.
 */
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 *arr, float scale)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N)
  {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
 * Initialize memory, update some globals
 */
void Boids::initSimulation(int N)
{
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void **)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void **)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void **)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
                                                               dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  const int CELL_TO_NEIGHBORHOOD_RATIO = 2;

  // LOOK-2.1 computing grid params
  gridCellWidth = (float)CELL_TO_NEIGHBORHOOD_RATIO * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = CELL_TO_NEIGHBORHOOD_RATIO * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.

  // uniform grid additional buffers
  cudaMalloc((void **)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("CUDA Malloc for particle array indices failed.");
  cudaMalloc((void **)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("CUDA Malloc for particle grid indices failed.");
  cudaMalloc((void **)&dev_gridCellStartIndices, N * sizeof(int));
  checkCUDAErrorWithLine("CUDA Malloc for grid cell start indices failed.");
  cudaMalloc((void **)&dev_gridCellEndIndices, N * sizeof(int));
  checkCUDAErrorWithLine("CUDA Malloc for grid cell end indices failed.");

  cudaMalloc((void **)&dev_posCoherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("CUDA Malloc for coherent positions failed.");

  cudaMalloc((void **)&dev_vel1Coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("CUDA Malloc for coherent velocities failed.");

  cudaDeviceSynchronize();
}

/******************
 * copyBoidsToVBO *
 ******************/

/**
 * Copy the boid positions into the VBO so that they can be drawn by OpenGL.
 */
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N)
  {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N)
  {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
 * Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
 */
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities)
{
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

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
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel)
{
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids
  const glm::vec3 iSelfPos = pos[iSelf];
  const glm::vec3 iSelfVel = vel[iSelf];

  // initiate values for each rules
  glm::vec3 cohesion = glm::vec3(0.f);
  glm::vec3 separation = glm::vec3(0.f);
  glm::vec3 alignment = glm::vec3(0.f);

  int count1 = 0;
  int count3 = 0;

  for (int i = 0; i < N; i++)
  {
    if (iSelf != i)
    {
      glm::vec3 currIPos = pos[i];
      glm::vec3 currIVel = vel[i];

      float dist2 = distance2(iSelfPos, currIPos);
      // create mask values to eliminate branching
      float mask1 = dist2 < rule1Distance * rule1Distance ? 1.f : 0.f;
      float mask2 = dist2 < rule2Distance * rule2Distance ? 1.f : 0.f;
      float mask3 = dist2 < rule3Distance * rule3Distance ? 1.f : 0.f;

      cohesion += mask1 * currIPos;
      separation -= mask2 * (currIPos - iSelfPos);
      alignment += mask3 * currIVel;

      // update number of neighbours for rules 1 & 3
      count1 += mask1;
      count3 += mask3;
    }
  }

  cohesion /= glm::max(count1, 1);
  cohesion = (cohesion - iSelfPos) * rule1Scale;

  separation *= rule2Scale;

  alignment /= glm::max(count3, 1);
  alignment = alignment * rule3Scale;

  return iSelfVel + cohesion + separation + alignment;
}

/**
 *
 * TODO-1.1 implement basic flocking
 * For each of the `N` bodies, update its position based on its current velocity.
 */
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
                                             glm::vec3 *vel1, glm::vec3 *vel2)
{
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
  const int iSelf = blockIdx.x * blockDim.x + threadIdx.x;
  if (iSelf >= N)
  {
    return;
  }

  glm::vec3 newVel = computeVelocityChange(N, iSelf, pos, vel1);

  // clamp speed
  float speed2 = glm::dot(newVel, newVel);
  newVel = speed2 > maxSpeed * maxSpeed ? newVel * rsqrtf(speed2) * maxSpeed : newVel;

  vel2[iSelf] = newVel;
}

/**
 * LOOK-1.2 Since this is pretty trivial, we implemented it for you.
 * For each of the `N` bodies, update its position based on its current velocity.
 */
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel)
{
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N)
  {
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
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution)
{
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
                                   glm::vec3 gridMin, float inverseCellWidth,
                                   glm::vec3 *pos, int *indices, int *gridIndices)
{
  // TODO-2.1
  // - Label each boid with the index of its grid cell.
  // - Set up a parallel array of integer indices as pointers to the actual
  //   boid data in pos and vel1/vel2

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N)
  {
    return;
  }

  indices[index] = index;

  glm::vec3 gridIndex3D = glm::floor((pos[index] - gridMin) * inverseCellWidth);
  int gridIndex = gridIndex3Dto1D(gridIndex3D.x, gridIndex3D.y, gridIndex3D.z, gridResolution);

  gridIndices[index] = gridIndex;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N)
  {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
                                         int *gridCellStartIndices, int *gridCellEndIndices)
{
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= N - 1)
  { // stop at N-1 as N-2 would give us an out of bounds error
    return;
  }

  const int currGridIndex = particleGridIndices[index];
  const int nextGridIndex = particleGridIndices[index + 1];

  if (index == 0)
  {
    gridCellStartIndices[currGridIndex] = index;
  }
  if (index == N - 2)
  { // use (N-2)th particle to set the (N-1)th classification
    gridCellEndIndices[nextGridIndex] = index + 1;
  }

  if (currGridIndex != nextGridIndex)
  {
    gridCellStartIndices[nextGridIndex] = index + 1; // mark that the next particle starts their grid cell
    gridCellEndIndices[currGridIndex] = index;       // mark that this particle ends this grid cell
  }
}

__global__ void kernUpdateVelNeighborSearchScattered(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int *gridCellStartIndices, int *gridCellEndIndices,
    int *particleArrayIndices,
    glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2)
{
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  int iSelf = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (iSelf >= N)
  {
    return;
  }

  // get pos and vel for index
  const glm::vec3 iSelfPos = pos[iSelf];
  const glm::vec3 iSelfVel = vel1[iSelf];

  // get 3D grid index
  glm::vec3 iSelfGridIndex3D = glm::floor((iSelfPos - gridMin) * inverseCellWidth);

  // initiate values for each rules
  glm::vec3 cohesion = glm::vec3(0.f);
  glm::vec3 separation = glm::vec3(0.f);
  glm::vec3 alignment = glm::vec3(0.f);

  int count1 = 0;
  int count3 = 0;

  float maxDist = fmaxf(fmaxf(rule1Distance, rule2Distance), rule3Distance);
  glm::vec3 maxDistVec = glm::vec3(maxDist);

  // find components of min and max grid cells that need to be searched
  glm::ivec3 gridIndex3DMin = glm::floor((iSelfPos - gridMin - maxDistVec) * inverseCellWidth);
  glm::ivec3 gridIndex3DMax = glm::floor((iSelfPos - gridMin + maxDistVec) * inverseCellWidth);

  for (int z = gridIndex3DMin.z; z <= gridIndex3DMax.z; z++)
  {
    for (int y = gridIndex3DMin.y; y <= gridIndex3DMax.y; y++)
    {
      for (int x = gridIndex3DMin.x; x <= gridIndex3DMax.x; x++)
      {
        if (z >= 0 && y >= 0 && x >= 0 && z < gridResolution && y < gridResolution && x < gridResolution)
        { // check for out of bounds

          // get info for neighbouring grid cell
          int gridIndex = gridIndex3Dto1D(x, y, z, gridResolution);
          int gridStart = gridCellStartIndices[gridIndex];
          int gridEnd = gridCellEndIndices[gridIndex];

          if (gridStart == -1)
          {
            // assert(gridEnd == -1); // if gridStart == -1 then gridEnd should also never have been initialized
            continue;
          }

          for (int i = gridStart; i <= gridEnd; i++)
          {
            // rule 1, rule 2, rule 3 logic
            int boidIndex = particleArrayIndices[i];
            if (iSelf != boidIndex)
            {
              glm::vec3 currBoidPos = pos[boidIndex];
              glm::vec3 currBoidVel = vel1[boidIndex];

              float dist2 = distance2(iSelfPos, currBoidPos);

              if (dist2 < rule1Distance)
              {
                cohesion += currBoidPos;
                count1 += 1;
              }
              if (dist2 < rule2Distance)
              {
                separation -= currBoidPos - iSelfPos;
              }
              if (dist2 < rule3Distance)
              {
                alignment += currBoidVel;
                count3 += 1;
              }
            }
          }
        }
      }
    }
  }

  // post-process affectors
  glm::vec3 rule1Vel = glm::vec3(0.f);
  glm::vec3 rule2Vel = separation * rule2Scale;
  glm::vec3 rule3Vel = glm::vec3(0.f);

  if (count1 > 0)
  {
    cohesion /= count1;
    rule1Vel = (cohesion - iSelfPos) * rule1Scale;
  }

  if (count3 > 0)
  {
    alignment /= count3;
    rule3Vel = alignment * rule3Scale;
  }

  glm::vec3 newVel = iSelfVel + rule1Vel + rule2Vel + rule3Vel;

  // clamp speed
  float speed2 = glm::dot(newVel, newVel);
  newVel = speed2 > maxSpeed * maxSpeed ? newVel * rsqrtf(speed2) * maxSpeed : newVel;

  vel2[iSelf] = newVel;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int *gridCellStartIndices, int *gridCellEndIndices,
    glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2)
{
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

  int iSelf = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (iSelf >= N)
  {
    return;
  }

  // get pos and vel for index
  const glm::vec3 iSelfPos = pos[iSelf];
  const glm::vec3 iSelfVel = vel1[iSelf];

  // get 3D grid index
  glm::vec3 iSelfGridIndex3D = glm::floor((iSelfPos - gridMin) * inverseCellWidth);

  // initiate values for each rules
  glm::vec3 cohesion = glm::vec3(0.f);
  glm::vec3 separation = glm::vec3(0.f);
  glm::vec3 alignment = glm::vec3(0.f);

  int count1 = 0;
  int count3 = 0;

  float maxDist = fmaxf(fmaxf(rule1Distance, rule2Distance), rule3Distance);
  glm::vec3 maxDistVec = glm::vec3(maxDist);

  // find components of min and max grid cells that need to be searched
  glm::ivec3 gridIndex3DMin = glm::floor((iSelfPos - gridMin - maxDistVec) * inverseCellWidth);
  glm::ivec3 gridIndex3DMax = glm::floor((iSelfPos - gridMin + maxDistVec) * inverseCellWidth);

  for (int z = gridIndex3DMin.z; z <= gridIndex3DMax.z; z++)
  {
    for (int y = gridIndex3DMin.y; y <= gridIndex3DMax.y; y++)
    {
      for (int x = gridIndex3DMin.x; x <= gridIndex3DMax.x; x++)
      {
        if (z >= 0 && y >= 0 && x >= 0 && z < gridResolution && y < gridResolution && x < gridResolution)
        { // check for out of bounds

          // get info for neighbouring grid cell
          int gridIndex = gridIndex3Dto1D(x, y, z, gridResolution);
          int gridStart = gridCellStartIndices[gridIndex];
          int gridEnd = gridCellEndIndices[gridIndex];

          if (gridStart == -1)
          {
            // assert(gridEnd == -1); // if gridStart == -1 then gridEnd should also never have been initialized
            continue;
          }

          for (int i = gridStart; i <= gridEnd; i++)
          { // i refers to boid within grid
            // rule 1, rule 2, rule 3 logic
            glm::vec3 currBoidPos = pos[i];
            glm::vec3 currBoidVel = vel1[i];

            float dist2 = distance2(iSelfPos, currBoidPos);

            if (dist2 < rule1Distance)
            {
              cohesion += currBoidPos;
              count1 += 1;
            }
            if (dist2 < rule2Distance)
            {
              separation -= currBoidPos - iSelfPos;
            }
            if (dist2 < rule3Distance)
            {
              alignment += currBoidVel;
              count3 += 1;
            }
          }
        }
      }
    }
  }

  // post-process affectors
  glm::vec3 rule1Vel = glm::vec3(0.f);
  glm::vec3 rule2Vel = separation * rule2Scale;
  glm::vec3 rule3Vel = glm::vec3(0.f);

  if (count1 > 0)
  {
    cohesion /= count1;
    rule1Vel = (cohesion - iSelfPos) * rule1Scale;
  }

  if (count3 > 0)
  {
    alignment /= count3;
    rule3Vel = alignment * rule3Scale;
  }

  glm::vec3 newVel = iSelfVel + rule1Vel + rule2Vel + rule3Vel;

  // clamp speed
  float speed2 = glm::dot(newVel, newVel);
  newVel = speed2 > maxSpeed * maxSpeed ? newVel * rsqrtf(speed2) * maxSpeed : newVel;

  vel2[iSelf] = newVel;
}

/**
 * Step the entire N-body simulation by `dt` seconds.
 */
void Boids::stepSimulationNaive(float dt)
{
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

  // ping-pong
  glm::vec3 *temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;
}

void Boids::stepSimulationScatteredGrid(float dt)
{
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

  int N = numObjects;

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(N, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_particleGridIndices(dev_particleGridIndices);
  thrust::device_ptr<int> dev_thrust_particleArrayIndices(dev_particleArrayIndices);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + N, dev_thrust_particleArrayIndices);

  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer for grid cell start indices failed.");

  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer for grid cell end indices failed.");

  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(N, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(
      N, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

  glm::vec3 *temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;
}

// process pos and vel based on sorted particle indices
__global__ void kernProcessParticlePosVel(int N, int *particleArrayIndices, glm::vec3 *pos, glm::vec3 *vel, glm::vec3 *posOut, glm::vec3 *velOut)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N)
  {
    int newIdx = particleArrayIndices[index];
    posOut[index] = pos[newIdx];
    velOut[index] = vel[newIdx];
  }
}

// restore particle velocities to their non-sorted index
__global__ void kernRestoreParticleVel(int N, int *particleArrayIndices, glm::vec3 *vel, glm::vec3 *velOut)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N)
  {
    int newIdx = particleArrayIndices[index];
    velOut[newIdx] = vel[index];
  }
}

void Boids::stepSimulationCoherentGrid(float dt)
{
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

  int N = numObjects;

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // DIFFERENCE: update positions first on unsorted positions
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(N, dt, dev_pos, dev_vel1);

  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(N, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_particleGridIndices(dev_particleGridIndices);
  thrust::device_ptr<int> dev_thrust_particleArrayIndices(dev_particleArrayIndices);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + N, dev_thrust_particleArrayIndices);

  // DIFFERENCE: process the pos and vel based on the sorted indices
  kernProcessParticlePosVel<<<fullBlocksPerGrid, blockSize>>>(N, dev_particleArrayIndices, dev_pos, dev_vel1, dev_posCoherent, dev_vel1Coherent);

  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer for grid cell start indices failed.");

  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer for grid cell end indices failed.");

  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(N, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  // DIFFERENCE: pass in coherent position and velocity
  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(
      N, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_posCoherent, dev_vel1Coherent, dev_vel1);

  // DIFFERENCE: ping-pong restore velocities to nonsorted index in preparation for next timestep
  kernRestoreParticleVel<<<fullBlocksPerGrid, blockSize>>>(N, dev_particleArrayIndices, dev_vel1, dev_vel2);

  glm::vec3 *temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;
}

void Boids::endSimulation()
{
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  // free 2.1 buffers
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  cudaFree(dev_posCoherent);
  cudaFree(dev_vel1Coherent);
}

void Boids::unitTest()
{
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]> intKeys{new int[N]};
  std::unique_ptr<int[]> intValues{new int[N]};

  intKeys[0] = 0;
  intValues[0] = 0;
  intKeys[1] = 1;
  intValues[1] = 1;
  intKeys[2] = 0;
  intValues[2] = 2;
  intKeys[3] = 3;
  intValues[3] = 3;
  intKeys[4] = 0;
  intValues[4] = 4;
  intKeys[5] = 2;
  intValues[5] = 5;
  intKeys[6] = 2;
  intValues[6] = 6;
  intKeys[7] = 0;
  intValues[7] = 7;
  intKeys[8] = 5;
  intValues[8] = 8;
  intKeys[9] = 6;
  intValues[9] = 9;

  cudaMalloc((void **)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void **)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++)
  {
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
  for (int i = 0; i < N; i++)
  {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
