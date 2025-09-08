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
// start by initializing to 0-#boids
// sort it by array below after calculating grid cell # of each boid
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

glm::vec3* dev_pos_coherent;
glm::vec3* dev_vel_coherent;

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
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));

  cudaMalloc((void**)&dev_pos_coherent, N * sizeof(glm::vec3));
  cudaMalloc((void**)&dev_vel_coherent, N * sizeof(glm::vec3));
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

  kernCopyPositionsToVBO <<<fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO <<<fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

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
    glm::vec3 percieved_center = glm::vec3(0.);
    glm::vec3 c = glm::vec3(0.);
    glm::vec3 percieved_velocity = glm::vec3(0.);
    int count1 = 0, count3 = 0;

    for (int i = 0;i < N;i++) {
        //Dist between current boid and original boid
        float dist = glm::distance(pos[i], pos[iSelf]);

        //Rule 1
        if (i != iSelf && dist<rule1Distance) {
            percieved_center += pos[i];
            count1++;
        }

        //Rule 2
        if (i != iSelf && dist < rule2Distance) {
            c -= (pos[i] - pos[iSelf]);
        }

        //Rule 3
        if (i != iSelf && dist < rule3Distance) {
            percieved_velocity += vel[i];
            count3++;
        }
    }

    //Divide by number of neighbors for pos and vel
    if (count1 > 0) percieved_center /= (float) count1;
    if (count3 > 0) percieved_velocity /= (float) count3;


    //Scale each computed vel
    glm::vec3 v1 = count1 > 0 ? (percieved_center - pos[iSelf]) * rule1Scale : glm::vec3(0.);
    glm::vec3 v2 = c * rule2Scale;
    glm::vec3 v3 = count3 > 0 ? (percieved_velocity ) * rule3Scale : glm::vec3(0.);

    //return accumulated dv
    return v1 + v2 + v3;
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
    int iSelf = threadIdx.x + blockIdx.x * blockDim.x;
    if (iSelf >= N) return;
    glm::vec3 dv = computeVelocityChange(N, iSelf, pos, vel1);
    glm::vec3 newVel = vel1[iSelf] + dv;
    if (glm::length(newVel) > maxSpeed) {
        newVel = glm::normalize(newVel) * maxSpeed;
    }

    vel2[iSelf] = newVel;
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
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    indices[index] = index;

    glm::vec3 rel = (pos[index] - gridMin) * inverseCellWidth;

    int ix = static_cast<int>(floor(rel.x));
    int iy = static_cast<int>(floor(rel.y));
    int iz = static_cast<int>(floor(rel.z));

    gridIndices[index] = gridIndex3Dto1D(ix, iy, iz, gridResolution);

}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int* particleGridIndices,
    int* gridCellStartIndices, int* gridCellEndIndices) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) return;

    int currCell = particleGridIndices[index];
    int prevCell = (index == 0) ? -1 : particleGridIndices[index - 1];
    int nextCell = (index == N - 1) ? -1 : particleGridIndices[index + 1];

    if (currCell != prevCell) {
        gridCellStartIndices[currCell] = index; 
    }
    if (currCell != nextCell) {
        gridCellEndIndices[currCell] = index;  
    }
}

void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize >>>(
        numObjects, dev_pos, dev_vel1, dev_vel2);

    glm::vec3* temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;
    kernUpdatePos<<<fullBlocksPerGrid, blockSize >>>(numObjects, dt, dev_pos, dev_vel1);
}

__global__ void kernUpdateVelNeighborSearchScattered(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices,
    int* particleArrayIndices,
    glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    glm::vec3 thisPos = pos[index];
    glm::vec3 thisVel = vel1[index];

    glm::vec3 perceived_center(0.0f);   // rule 1
    glm::vec3 c(0.0f);                  // rule 2
    glm::vec3 perceived_velocity(0.0f); // rule 3
    int count1 = 0, count3 = 0;

    // get this thread's boid grid cell
    glm::vec3 rel = (thisPos - gridMin) * inverseCellWidth;
    int ix = static_cast<int>(floor(rel.x));
    int iy = static_cast<int>(floor(rel.y));
    int iz = static_cast<int>(floor(rel.z));

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int nx = ix + dx;
                int ny = iy + dy;
                int nz = iz + dz;

                // bounds check
                if (nx < 0 || ny < 0 || nz < 0 || nx >= gridResolution || ny >= gridResolution || nz >= gridResolution) {
                    continue;
                }

                int neighborCell = gridIndex3Dto1D(nx, ny, nz, gridResolution);

                int startIdx = gridCellStartIndices[neighborCell];
                int endIdx = gridCellEndIndices[neighborCell];

                if (startIdx == -1 || endIdx == -1) continue;

                // loop through boids in this cell
                for (int j = startIdx; j <= endIdx; j++) {
                    int boidIndex = particleArrayIndices[j];
                    if (boidIndex == index) continue;

                    float dist = glm::distance(pos[boidIndex], thisPos);

                    // RULE 1
                    if (dist < rule1Distance) {
                        perceived_center += pos[boidIndex];
                        count1++;
                    }
                    // RULE 2
                    if (dist < rule2Distance) {
                        c -= (pos[boidIndex] - thisPos);
                    }

                    // RULE 3
                    if (dist < rule3Distance) {
                        perceived_velocity += vel1[boidIndex];
                        count3++;
                    }
                }
            }
        }
    }

    // --- finalize rule contributions ---
    if (count1 > 0) perceived_center /= (float)count1;
    if (count3 > 0) perceived_velocity /= (float)count3;

    glm::vec3 v1 = count1 > 0 ? (perceived_center - thisPos) * rule1Scale : glm::vec3(0.0f);
    glm::vec3 v2 = c * rule2Scale;
    glm::vec3 v3 = count3 > 0 ? (perceived_velocity ) * rule3Scale : glm::vec3(0.0f);

    glm::vec3 dv = v1 + v2 + v3;

    glm::vec3 newVel = thisVel + dv;
    if (glm::length(newVel) > maxSpeed) {
        newVel = glm::normalize(newVel) * maxSpeed;
    }

    vel2[index] = newVel;
}

__global__ void kernReorderData(int N,
    int* particleArrayIndices,
    glm::vec3* pos,
    glm::vec3* vel,
    glm::vec3* pos_coherent,
    glm::vec3* vel_coherent)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) return;

    int sortedIndex = particleArrayIndices[index];
    pos_coherent[index] = pos[sortedIndex];
    vel_coherent[index] = vel[sortedIndex];
}


#define USE_8_NEIGHBORS 0  // Set 0 for 27-cell, 1 for 8-cell

__global__ void kernUpdateVelNeighborSearchCoherent(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices,
    glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) return;

    glm::vec3 thisPos = pos[index];
    glm::vec3 thisVel = vel1[index];

    glm::vec3 perceived_center(0.0f);
    glm::vec3 c(0.0f);
    glm::vec3 perceived_velocity(0.0f);
    int count1 = 0, count3 = 0;

    // Compute which cell this boid is in
    glm::vec3 rel = (thisPos - gridMin) * inverseCellWidth;
    int ix = static_cast<int>(floor(rel.x));
    int iy = static_cast<int>(floor(rel.y));
    int iz = static_cast<int>(floor(rel.z));

#if USE_8_NEIGHBORS
    // 8 Cell
    int ix0 = (rel.x - ix < 0.5f) ? ix - 1 : ix;
    int iy0 = (rel.y - iy < 0.5f) ? iy - 1 : iy;
    int iz0 = (rel.z - iz < 0.5f) ? iz - 1 : iz;

    for (int dx = 0; dx <= 1; dx++) {
        for (int dy = 0; dy <= 1; dy++) {
            for (int dz = 0; dz <= 1; dz++) {
                int nx = ix0 + dx;
                int ny = iy0 + dy;
                int nz = iz0 + dz;

                if (nx < 0 || ny < 0 || nz < 0 || nx >= gridResolution || ny >= gridResolution || nz >= gridResolution)
                    continue;

                int neighborCell = gridIndex3Dto1D(nx, ny, nz, gridResolution);
                int startIdx = gridCellStartIndices[neighborCell];
                int endIdx = gridCellEndIndices[neighborCell];
                if (startIdx == -1 || endIdx == -1) continue;

                for (int j = startIdx; j <= endIdx; j++) {
                    if (j == index) continue;
                    float dist = glm::distance(pos[j], thisPos);

                    if (dist < rule1Distance) {
                        perceived_center += pos[j];
                        count1++;
                    }
                    if (dist < rule2Distance) {
                        c -= (pos[j] - thisPos);
                    }
                    if (dist < rule3Distance) {
                        perceived_velocity += vel1[j];
                        count3++;
                    }
                }
            }
        }
    }
#else
    // 27 Cell
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int nx = ix + dx;
                int ny = iy + dy;
                int nz = iz + dz;

                if (nx < 0 || ny < 0 || nz < 0 || nx >= gridResolution || ny >= gridResolution || nz >= gridResolution)
                    continue;

                int neighborCell = gridIndex3Dto1D(nx, ny, nz, gridResolution);
                int startIdx = gridCellStartIndices[neighborCell];
                int endIdx = gridCellEndIndices[neighborCell];
                if (startIdx == -1 || endIdx == -1) continue;

                for (int j = startIdx; j <= endIdx; j++) {
                    if (j == index) continue;
                    float dist = glm::distance(pos[j], thisPos);

                    if (dist < rule1Distance) {
                        perceived_center += pos[j];
                        count1++;
                    }
                    if (dist < rule2Distance) {
                        c -= (pos[j] - thisPos);
                    }
                    if (dist < rule3Distance) {
                        perceived_velocity += vel1[j];
                        count3++;
                    }
                }
            }
        }
    }
#endif

    if (count1 > 0) perceived_center /= (float)count1;
    if (count3 > 0) perceived_velocity /= (float)count3;

    glm::vec3 v1 = count1 > 0 ? (perceived_center - thisPos) * rule1Scale : glm::vec3(0.0f);
    glm::vec3 v2 = c * rule2Scale;
    glm::vec3 v3 = count3 > 0 ? (perceived_velocity)*rule3Scale : glm::vec3(0.0f);

    glm::vec3 newVel = thisVel + v1 + v2 + v3;
    if (glm::length(newVel) > maxSpeed)
        newVel = glm::normalize(newVel) * maxSpeed;

    vel2[index] = newVel;
}




/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationScatteredGrid(float dt) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);

    // TODO-2.1
    // Uniform Grid Neighbor search using Thrust sort.
    // In Parallel:
    // - label each particle with its array index as well as its grid index.
    //   Use 2x width grids.
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        dev_pos,
        dev_particleArrayIndices,
        dev_particleGridIndices
        );
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    //   are welcome to do a performance comparison.
    thrust::device_ptr<int> dev_thrust_particleGridIndices(dev_particleGridIndices);
    thrust::device_ptr<int> dev_thrust_particleArrayIndices(dev_particleArrayIndices);
    thrust::sort_by_key(
        dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices
    );

    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (
        gridCellCount, dev_gridCellStartIndices, -1
        );
    kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (
        gridCellCount, dev_gridCellEndIndices, -1
        );
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");

    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        dev_particleGridIndices,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices
        );
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        gridCellWidth,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        dev_particleArrayIndices,
        dev_pos,
        dev_vel1,
        dev_vel2
        );
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

    // - Update positions
    // - Ping-pong buffers as needed
    glm::vec3* temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;

    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        dt,
        dev_pos,
        dev_vel1
        );
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    cudaDeviceSynchronize();
}


void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);

    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        dev_pos,
        dev_particleArrayIndices,
        dev_particleGridIndices
        );
    checkCUDAErrorWithLine("kernComputeIndices failed!");
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    thrust::device_ptr<int> dev_thrust_particleGridIndices(dev_particleGridIndices);
    thrust::device_ptr<int> dev_thrust_particleArrayIndices(dev_particleArrayIndices);
    thrust::sort_by_key(
        dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices
    );

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
    kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >>> (
        numObjects,
        dev_particleGridIndices,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices
        );
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
    kernReorderData<<<fullBlocksPerGrid, blockSize>>> (
        numObjects,
        dev_particleArrayIndices,
        dev_pos,
        dev_vel1,
        dev_pos_coherent,
        dev_vel_coherent
        );

  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

    std::swap(dev_pos, dev_pos_coherent);
    std::swap(dev_vel1, dev_vel_coherent);

    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        gridCellWidth,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        dev_pos,
        dev_vel1,
        dev_vel2
        );
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

    std::swap(dev_vel1, dev_vel2);

    kernUpdatePos <<<fullBlocksPerGrid, blockSize >> > (
        numObjects,
        dt,
        dev_pos,
        dev_vel1
        );
    checkCUDAErrorWithLine("kernUpdatePos failed!");

}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);

  cudaFree(dev_pos_coherent);
  cudaFree(dev_vel_coherent);

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
