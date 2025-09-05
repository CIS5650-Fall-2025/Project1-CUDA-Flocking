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
#include <device_launch_parameters.h>

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

// Coherent (cell-sorted) copies
glm::vec3 *dev_posCoherent;
glm::vec3 *dev_vel1Coherent;
glm::vec3 *dev_vel2Coherent;


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
  cudaMalloc(&dev_particleArrayIndices, N * sizeof(int));
  cudaMalloc(&dev_particleGridIndices, N * sizeof(int));
  cudaMalloc(&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  cudaMalloc(&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  cudaMalloc(&dev_posCoherent, N * sizeof(glm::vec3));
  cudaMalloc(&dev_vel1Coherent, N * sizeof(glm::vec3));
  cudaMalloc(&dev_vel2Coherent, N * sizeof(glm::vec3));
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
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  glm::vec3 p = pos[iSelf], v = vel[iSelf];
  glm::vec3 centre(0), separation(0), average_vel(0), deltaV(0);
  int count1 = 0, count3 = 0;

  for (int i = 0; i < N; i++) {
    if (i == iSelf) continue;
    glm::vec3 pos_dif = pos[i] - p;
    float distance = glm::length(pos_dif);
    if (distance < rule1Distance) {
      centre += pos[i];
      count1++;
    }
    if (distance < rule2Distance) {
      separation -= (pos[i] - p);
    }
    if (distance < rule3Distance) {
      average_vel += vel[i];
      count3++;
    }
  }
  if (count1 > 0) {
    centre /= float(count1);
    deltaV += (centre - p) * rule1Scale;
  }
  deltaV += separation * rule2Scale;
  if (count3 > 0) {
    average_vel /= float(count3);
    deltaV += average_vel * rule3Scale;
  }
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids
  return deltaV;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  glm::vec3 newV = vel1[i] + computeVelocityChange(N, i, pos, vel1);
  float s = glm::length(newV);
  if (s > maxSpeed) newV = (newV / s) * maxSpeed;
  vel2[i] = newV;

  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  glm::vec3 rel = (pos[i] - gridMin) * inverseCellWidth;
  int grid_x = imax(0, imin(gridResolution - 1, int(floorf(rel.x))));
  int grid_y = imax(0, imin(gridResolution - 1, int(floorf(rel.y))));
  int grid_z = imax(0, imin(gridResolution - 1, int(floorf(rel.z))));
  int cell = gridIndex3Dto1D(grid_x, grid_y, grid_z, gridResolution);
  gridIndices[i] = cell;
  // - Set up a parallel array of integer indices as pointers to the actual
	//   boid data in pos and vel1/vel2
  indices[i] = i;
	
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernReorderCoherent(int N, const int* particleArrayIndices,     
  const glm::vec3* posIn, const glm::vec3* vel1In, glm::vec3* posOut, glm::vec3* vel1Out) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int src = particleArrayIndices[i];
  posOut[i] = posIn[src];
  vel1Out[i] = vel1In[src];
}


__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
   
  int curr = particleGridIndices[i];

  if (i == 0) {
    gridCellStartIndices[curr] = 0; 
  } else
  {
    int prev = particleGridIndices[i - 1];
    if (prev != curr)
    {
      gridCellEndIndices[prev] = i - 1;
      gridCellStartIndices[curr] = i;
    }
  }
  if (i == N - 1)
  {
    gridCellEndIndices[curr] = i;
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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int boidIdx = particleArrayIndices[i];
  glm::vec3 p = pos[boidIdx];

  glm::vec3 rel = (p - gridMin) * inverseCellWidth;
  int grid_x = imax(0, imin(gridResolution - 1, int(floorf(rel.x))));
  int grid_y = imax(0, imin(gridResolution - 1, int(floorf(rel.y))));
  int grid_z = imax(0, imin(gridResolution - 1, int(floorf(rel.z))));

  float frac_x = rel.x - floorf(rel.x);
  float frac_y = rel.y - floorf(rel.y);
  float frac_z = rel.z - floorf(rel.z);

  // - Identify which cells may contain neighbors. This isn't always 8.
  int nx = grid_x + ((frac_x >= 0.5f) ? 1 : -1);
  int ny = grid_y + ((frac_y >= 0.5f) ? 1 : -1);
  int nz = grid_z + ((frac_z >= 0.5f) ? 1 : -1);
  nx = imax(0, imin(gridResolution - 1, nx));
  ny = imax(0, imin(gridResolution - 1, ny));
  nz = imax(0, imin(gridResolution - 1, nz));

  int xs[2] = { grid_x, nx };
  int ys[2] = { grid_y, ny};
  int zs[2] = { grid_z, nz};
	//check for duplicates
  int xCount = (nx == grid_x) ? 1 : 2;
  int yCount = (ny == grid_y) ? 1 : 2;
  int zCount = (nz == grid_z) ? 1 : 2;

  glm::vec3 centre(0), separation(0), averageV(0);
  int count1 = 0, count3 = 0;

  // - For each cell, read the start/end indices in the boid pointer array.
  for (int x = 0; x < xCount; x++) {
    for (int y = 0; y < yCount; y++) {
	    for (int z = 0; z < zCount; z++) {
        int cx = xs[x], cy = ys[y], cz = zs[z];
        int cell = gridIndex3Dto1D(cx, cy, cz, gridResolution);

        int start = gridCellStartIndices[cell];
        if (start == -1) continue;
        int end = gridCellEndIndices[cell];

        for (int k = start; k <= end; k++) {
          int j = particleArrayIndices[k];
          if (j == boidIdx) continue;

          glm::vec3 d = pos[j] - p;
          float dist = glm::length(d);

          if (dist < rule1Distance) {
            centre += pos[j];
            count1++;
          }
          if (dist < rule2Distance) {
            separation -= d;
          }
          if (dist < rule3Distance) {
            averageV += vel1[j];
            count3++;
          }
        }
	    }
    }
  }// - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  glm::vec3 deltaV(0);
  if (count1 > 0) {
    centre /= (float)count1;
    deltaV += (centre - p) * rule1Scale;
  }
  deltaV += separation * rule2Scale;
  if (count3 > 0) {
    averageV /= (float)count3;
    deltaV += averageV * rule3Scale;
  }
  // - Clamp the speed change before putting the new speed in vel2
  glm::vec3 v = vel1[boidIdx] + deltaV;
  float s = glm::length(v);
  if (s > maxSpeed) v = (v / s) * maxSpeed;

  vel2[boidIdx] = v;
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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  glm::vec3 p = pos[i];

  glm::vec3 rel = (p - gridMin) * inverseCellWidth;
  int grid_x = imax(0, imin(gridResolution - 1, int(floorf(rel.x))));
  int grid_y = imax(0, imin(gridResolution - 1, int(floorf(rel.y))));
  int grid_z = imax(0, imin(gridResolution - 1, int(floorf(rel.z))));

  float frac_x = rel.x - floorf(rel.x);
  float frac_y = rel.y - floorf(rel.y);
  float frac_z = rel.z - floorf(rel.z);

  // - Identify which cells may contain neighbors. This isn't always 8.
  int nx = grid_x + ((frac_x >= 0.5f) ? 1 : -1);
  int ny = grid_y + ((frac_y >= 0.5f) ? 1 : -1);
  int nz = grid_z + ((frac_z >= 0.5f) ? 1 : -1);
  nx = imax(0, imin(gridResolution - 1, nx));
  ny = imax(0, imin(gridResolution - 1, ny));
  nz = imax(0, imin(gridResolution - 1, nz));

  //int xs[2] = { grid_x, nx };
  //int ys[2] = { grid_y, ny };
  //int zs[2] = { grid_z, nz };
  ////check for duplicates
  //int xCount = (nx == grid_x) ? 1 : 2;
  //int yCount = (ny == grid_y) ? 1 : 2;
  //int zCount = (nz == grid_z) ? 1 : 2;
  int xMin = (nx < grid_x) ? nx : grid_x;
  int xMax = (nx < grid_x) ? grid_x : nx;
  int yMin = (ny < grid_y) ? ny : grid_y;
  int yMax = (ny < grid_y) ? grid_y : ny;
  int zMin = (nz < grid_z) ? nz : grid_z;
  int zMax = (nz < grid_z) ? grid_z : nz;

  glm::vec3 centre(0), separation(0), averageV(0);
  int count1 = 0, count3 = 0;

  // - For each cell, read the start/end indices in the boid pointer array.
  for (int z = zMin; z <= zMax; z++) {
    int zBase = z * gridResolution * gridResolution;
    for (int y = yMin; y <= yMax; y++) {
      int base = zBase + y * gridResolution + xMin;
      for (int x = xMin; x <= xMax; x++, base++) {

        int start = gridCellStartIndices[base];
        if (start == -1) continue;
        int end = gridCellEndIndices[base];

        for (int k = start; k <= end; k++) {
          if (k == i) continue;
          glm::vec3 d = pos[k] - p;
          float dist = glm::length(d);

          if (dist < rule1Distance) {
            centre += pos[k];
            count1++;
          }
          if (dist < rule2Distance) {
            separation -= d;
          }
          if (dist < rule3Distance) {
            averageV += vel1[k];
            count3++;
          }
        }
      }
    }
  }// - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  glm::vec3 deltaV(0);
  if (count1 > 0) {
    centre /= (float)count1;
    deltaV += (centre - p) * rule1Scale;
  }
  deltaV += separation * rule2Scale;
  if (count3 > 0) {
    averageV /= (float)count3;
    deltaV += averageV * rule3Scale;
  }
  // - Clamp the speed change before putting the new speed in vel2
  glm::vec3 v = vel1[i] + deltaV;
  float s = glm::length(v);
  if (s > maxSpeed) v = (v / s) * maxSpeed;

  vel2[i] = v;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
  dim3 blocks((numObjects + blockSize - 1) / blockSize);

  kernUpdateVelocityBruteForce << <blocks, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("Update vel brute force failed");

  glm::vec3* temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;

  kernUpdatePos << <blocks, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
  checkCUDAErrorWithLine("Update pos failed");
  cudaDeviceSynchronize();
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  dim3 blocksParticles((numObjects + blockSize - 1) / blockSize);
  dim3 blocksCells((gridCellCount + blockSize - 1) / blockSize);

  // Label boids' array and grid indices
  kernComputeIndices << <blocksParticles, blockSize >> > (numObjects, gridSideCount, gridMinimum, 
    gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices error");
	
  //Sort by key using thrust
  thrust::sort_by_key(dev_thrust_particleGridIndices, 
    dev_thrust_particleGridIndices + numObjects, dev_particleArrayIndices);
  checkCUDAErrorWithLine("Thrust sort by key failed");

  //Reset start and end buffers to -1
  kernResetIntBuffer << <blocksCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer << <blocksCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer error");

  //Identify cell ranges
  kernIdentifyCellStartEnd << <blocksParticles, blockSize >> > (numObjects, dev_particleGridIndices,
    dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd error");

  //Neighbour search, update velocities
  kernUpdateVelNeighborSearchScattered << <blocksParticles, blockSize >> > (numObjects, gridSideCount,
    gridMinimum, gridInverseCellWidth,
    gridCellWidth, dev_gridCellStartIndices,
    dev_gridCellEndIndices, dev_particleArrayIndices,
    dev_pos, dev_vel1, dev_vel2
    );
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered error");
  
  //Ping pong buffers to put new vels in vel1
  glm::vec3* temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;

  //Update positions
  kernUpdatePos << <blocksParticles, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
  checkCUDAErrorWithLine("kernUpdatePos error");
  cudaDeviceSynchronize();
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
  dim3 blocksParticles((numObjects + blockSize - 1) / blockSize);
  dim3 blocksCells((gridCellCount + blockSize - 1) / blockSize);

  // Label boids' array and grid indices
  kernComputeIndices << <blocksParticles, blockSize >> > (numObjects, gridSideCount, gridMinimum,
    gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices error");

  //Sort by key using thrust
  thrust::sort_by_key(dev_thrust_particleGridIndices,
    dev_thrust_particleGridIndices + numObjects, dev_particleArrayIndices);
  checkCUDAErrorWithLine("Thrust sort by key failed");

  //Reset start and end buffers to -1
  kernResetIntBuffer << <blocksCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer << <blocksCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer error");

  //Identify cell ranges
  kernIdentifyCellStartEnd << <blocksParticles, blockSize >> > (numObjects, dev_particleGridIndices,
    dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd error");

  //Reorder pos and vel1 to coherent arrays
  kernReorderCoherent << <blocksParticles, blockSize >> > (
    numObjects, dev_particleArrayIndices, dev_pos, dev_vel1,
    dev_posCoherent, dev_vel1Coherent);
  checkCUDAErrorWithLine("reorder coherent");

  //Neighbour search, update velocities
  kernUpdateVelNeighborSearchCoherent << <blocksParticles, blockSize >> > (
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
    dev_gridCellStartIndices, dev_gridCellEndIndices,
    dev_posCoherent, dev_vel1Coherent, dev_vel2Coherent);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent error");

  //Ping pong buffers (coherent)
  glm::vec3* temp = dev_vel1Coherent;
	dev_vel1Coherent = dev_vel2Coherent;
	dev_vel2Coherent = temp;

  //Update positions
  kernUpdatePos << <blocksParticles, blockSize >> > (numObjects, dt, dev_posCoherent, dev_vel1Coherent);
  checkCUDAErrorWithLine("kernUpdatePosCoherent error");

  //Publish coherent arrays as the live arrays for next frame
  glm::vec3* tmp;

  tmp = dev_pos;
	dev_pos = dev_posCoherent;
	dev_posCoherent = tmp;

  tmp = dev_vel1;
	dev_vel1 = dev_vel1Coherent;
	dev_vel1Coherent = tmp;

  tmp = dev_vel2;
	dev_vel2 = dev_vel2Coherent;
	dev_vel2Coherent = tmp;

  cudaDeviceSynchronize();
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_posCoherent);
  cudaFree(dev_vel1Coherent);
  cudaFree(dev_vel2Coherent);
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
