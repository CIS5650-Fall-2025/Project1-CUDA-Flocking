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

#define ed 1.0f

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
// reshuffled pos
glm::vec3* dev_rPos;
// reshuffled vel1
glm::vec3* dev_rVel1;
// reshuffled vel2
glm::vec3* dev_rVel2;

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

    // buffer containing a pointer for each boid to its data in dev_pos and dev_vel1 and dev_vel2
    cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");
    dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);

    // buffer containing the grid index of each boid
    cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");
    dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);

    // buffer containing a pointer for each cell to the beginning of its data in dev_particleArrayIndices
    cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

    // buffer containing a pointer for each cell to the end of its data in dev_particleArrayIndices
    cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

    cudaMalloc((void**)&dev_rPos, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_rPos failed!");

    cudaMalloc((void**)&dev_rVel1, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_rVel1 failed!");

    cudaMalloc((void**)&dev_rVel2, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_rVel2 failed!");
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
    ///////// parameters
    glm::vec3 perceived_center = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 perceived_velocity = glm::vec3(0.f, 0.f, 0.f);

    // counter for number of neighbors 
    int numNeighbors1 = 0;
    int numNeighbors3 = 0;

    // position of this boid
    glm::vec3 selfPos = pos[iSelf];

    glm::vec3 c = glm::vec3(0.f, 0.f, 0.f);

    ///////// loop through boids
    for (int bIdx = 0; bIdx < N; bIdx++) {
        if (bIdx != iSelf) {
            // distance from boid iSelf to the current boid we've looped to
            glm::vec3 boidPos = pos[bIdx];
            float dist = glm::length(selfPos - boidPos);

            //////// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
            if (dist < rule1Distance) {
                perceived_center += boidPos;
                numNeighbors1++;
            }

            ///////// Rule 2: boids try to stay a distance d away from each other
            if (dist < rule2Distance) {
                c -= (boidPos - selfPos);
            }

            //////// Rule 3: boids try to match the speed of surrounding boids
            if (dist < rule3Distance) {
                perceived_velocity += vel[bIdx];
                numNeighbors3++;
            }
        }
    }

    //////// compute result
    glm::vec3 resultVel = glm::vec3(0.f, 0.f, 0.f);

    // Rule 1
    if (numNeighbors1 > 0) {
        perceived_center /= numNeighbors1;
        resultVel += (perceived_center - selfPos) * rule1Scale;
    }
    
    // Rule 2
    resultVel += c * rule2Scale;

    // Rule 3
    if (numNeighbors3 > 0) {
        perceived_velocity /= numNeighbors3;
        resultVel += perceived_velocity * rule3Scale;
    }

    return resultVel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
    // compute thread
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // Compute a new velocity based on pos and vel1
    glm::vec3 velChange = computeVelocityChange(N, index, pos, vel1);
    glm::vec3 thisVel = vel1[index] + velChange;
  
    // Clamp the speed
    float thisSpeed = glm::length(thisVel);
    if (thisSpeed > maxSpeed) {
        thisVel = glm::normalize(thisVel) * maxSpeed;
    }

    // Record the new velocity into vel2. Question: why NOT vel1? 
    // Answer: calculations for one boid depend on calculations for
    // surrounding boids. If you change their state values before all 
    // other boids have finished computations, the result will be incorrect.
    vel2[index] = thisVel;
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
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // - Label each boid with the index of its grid cell.
    glm::vec3 gridIdx3D = glm::floor((pos[index] - gridMin) * inverseCellWidth);
    gridIdx3D = glm::max(glm::vec3(0), glm::min(gridIdx3D, glm::vec3(gridResolution - 1)));
    gridIndices[index] = gridIndex3Dto1D(gridIdx3D.x, gridIdx3D.y, gridIdx3D.z, gridResolution);

    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    indices[index] = index;
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

    // boid data idx
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // grid idx
    int currGridIdx = particleGridIndices[index];

    //////// identify start point of each cell
    // initialize boid 0
    if (index == 0) {
        gridCellStartIndices[currGridIdx] = 0;
    } else {
        int prevGridIdx = particleGridIndices[index - 1];
        if (currGridIdx != prevGridIdx) {
            // beginning cell of data for curr boid in dev_particleArrayIndices
            gridCellStartIndices[currGridIdx] = index;

            // end cell of data for prev boid in dev_particleArrayIndices
            gridCellEndIndices[prevGridIdx] = index - 1;
        }
    }
    if (index == N - 1) {
        gridCellEndIndices[currGridIdx] = N - 1;
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
    // boid data idx
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }


    ///////// parameters
    glm::vec3 perceived_center = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 perceived_velocity = glm::vec3(0.f, 0.f, 0.f);

    // counter for number of neighbors 
    int numNeighbors1 = 0;
    int numNeighbors3 = 0;

    // get pos and vel of this boid
    glm::vec3 thisPos = pos[index];
    glm::vec3 thisVel = vel1[index];

    glm::vec3 c = glm::vec3(0.f, 0.f, 0.f);



    // identify min and max grid cell coords to search through
    float maxDist = glm::max(glm::max(rule1Distance, rule2Distance), rule3Distance);
    glm::vec3 minGridIdx3D = glm::floor((thisPos - gridMin - glm::vec3(maxDist)) * inverseCellWidth);
    glm::vec3 maxGridIdx3D = glm::floor((thisPos - gridMin + glm::vec3(maxDist)) * inverseCellWidth);

    // - Identify which cells may contain neighbors. This isn't always 8.
    // check that indices are w/in bounds
    if (minGridIdx3D.x >= 0 && minGridIdx3D.y >= 0 && minGridIdx3D.z >= 0) {
        for (int x = minGridIdx3D.x; x <= maxGridIdx3D.x; x++) {
            for (int y = minGridIdx3D.y; y <= maxGridIdx3D.y; y++) {
                for (int z = minGridIdx3D.z; z <= maxGridIdx3D.z; z++) {

                    // check that indices are w/in bounds
                    if (x < gridResolution && y < gridResolution && z < gridResolution) {
                        
                        // - For each cell, read the start/end indices in the boid pointer array.
                        int cGridIdx = gridIndex3Dto1D(x, y, z, gridResolution);
                        int cGridStart = gridCellStartIndices[cGridIdx];
                        int cGridEnd = gridCellEndIndices[cGridIdx];

                        // check that start has been initialized
                        if (cGridStart == -1) {
                            continue;
                        }

                        // - Access each boid in the cell and compute velocity change from
                        //   the boids rules, if this boid is within the neighborhood distance.
                        for (int nbDataIdx = cGridStart; nbDataIdx <= cGridEnd; nbDataIdx++) {
                            // index of boid's data in dev_pos, dev_vel1, and dev_vel2
                            int nbIdx = particleArrayIndices[nbDataIdx];

                            // don't compare this boid to itself
                            if (index != nbIdx) {
                                // get pos and vel of neighbor boid
                                glm::vec3 nbPos = pos[nbIdx];
                                glm::vec3 nbVel = vel1[nbIdx];

                                // distance btwn this boid and the neighbor boid
                                float dist = glm::length(nbPos - thisPos);



                                //////// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
                                if (dist < rule1Distance) {
                                    perceived_center += nbPos;
                                    numNeighbors1++;
                                }

                                ///////// Rule 2: boids try to stay a distance d away from each other
                                if (dist < rule2Distance) {
                                    c -= (nbPos - thisPos);
                                }

                                //////// Rule 3: boids try to match the speed of surrounding boids
                                if (dist < rule3Distance) {
                                    perceived_velocity += nbVel;
                                    numNeighbors3++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    // - Clamp the speed change before putting the new speed in vel2
    //////// compute result
    glm::vec3 resultVel = glm::vec3(0.f, 0.f, 0.f);

    // Rule 1
    if (numNeighbors1 > 0) {
        perceived_center /= numNeighbors1;
        resultVel += (perceived_center - thisPos) * rule1Scale;
    }

    // Rule 2
    resultVel += c * rule2Scale;

    // Rule 3
    if (numNeighbors3 > 0) {
        perceived_velocity /= numNeighbors3;
        resultVel += perceived_velocity * rule3Scale;
    }

    resultVel += thisVel;

    // Clamp the speed
    float resultSpeed = glm::length(resultVel);
    if (resultSpeed > maxSpeed) {
        resultVel = glm::normalize(resultVel) * maxSpeed;
    }

    // Record the new velocity into vel2. 
    vel2[index] = resultVel;
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

    // - Identify the grid cell that this particle is in
    // boid data idx
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }


    ///////// parameters
    glm::vec3 perceived_center = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 perceived_velocity = glm::vec3(0.f, 0.f, 0.f);

    // counter for number of neighbors 
    int numNeighbors1 = 0;
    int numNeighbors3 = 0;

    // get pos and vel of this boid
    glm::vec3 thisPos = pos[index];
    glm::vec3 thisVel = vel1[index];

    glm::vec3 c = glm::vec3(0.f, 0.f, 0.f);

    // get grid cell of this boid
    glm::vec3 gridCell = glm::floor((thisPos - gridMin) * inverseCellWidth);

    // identify min and max grid cell coords to search through
    float maxDist = glm::max(glm::max(rule1Distance, rule2Distance), rule3Distance);
    glm::vec3 minGridIdx3D = glm::floor((thisPos - gridMin - glm::vec3(maxDist)) * inverseCellWidth);
    minGridIdx3D = glm::max(glm::vec3(0), minGridIdx3D);
    glm::vec3 maxGridIdx3D = glm::floor((thisPos - gridMin + glm::vec3(maxDist)) * inverseCellWidth);
    maxGridIdx3D = glm::min(glm::vec3(gridResolution - 1), maxGridIdx3D);

    // - Identify which cells may contain neighbors. This isn't always 8.
    // check that indices are w/in bounds
    if (minGridIdx3D.x >= 0 && minGridIdx3D.y >= 0 && minGridIdx3D.z >= 0) {
        for (int x = (int)minGridIdx3D.x; x <= (int)maxGridIdx3D.x; x++) {
            for (int y = (int)minGridIdx3D.y; y <= (int)maxGridIdx3D.y; y++) {
                for (int z = (int)minGridIdx3D.z; z <= (int)maxGridIdx3D.z; z++) {

                    // check that indices are w/in bounds
                    if (x < gridResolution && y < gridResolution && z < gridResolution) {

                        // - For each cell, read the start/end indices in the boid pointer array.
                        int cGridIdx = gridIndex3Dto1D(x, y, z, gridResolution);
                        int cGridStart = gridCellStartIndices[cGridIdx];
                        int cGridEnd = gridCellEndIndices[cGridIdx];

                        // check that start has been initialized
                        if (cGridStart == -1) {
                            continue;
                        }

                        // - Access each boid in the cell and compute velocity change from
                        //   the boids rules, if this boid is within the neighborhood distance.
                        for (int nbDataIdx = cGridStart; nbDataIdx <= cGridEnd; nbDataIdx++) {

                            // don't compare this boid to itself
                            if (index != nbDataIdx) {
                                // get pos and vel of neighbor boid
                                glm::vec3 nbPos = pos[nbDataIdx];
                                glm::vec3 nbVel = vel1[nbDataIdx];

                                // distance btwn this boid and the neighbor boid
                                float dist = glm::length(nbPos - thisPos);

                                //////// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
                                if (dist < rule1Distance) {
                                    perceived_center += nbPos;
                                    numNeighbors1++;
                                }

                                ///////// Rule 2: boids try to stay a distance d away from each other
                                if (dist < rule2Distance) {
                                    c -= (nbPos - thisPos);
                                }

                                //////// Rule 3: boids try to match the speed of surrounding boids
                                if (dist < rule3Distance) {
                                    perceived_velocity += nbVel;
                                    numNeighbors3++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    // - Clamp the speed change before putting the new speed in vel2
    //////// compute result
    glm::vec3 resultVel = glm::vec3(0.f, 0.f, 0.f);

    // Rule 1
    if (numNeighbors1 > 0) {
        perceived_center /= numNeighbors1;
        resultVel += (perceived_center - thisPos) * rule1Scale;
    }

    // Rule 2
    resultVel += c * rule2Scale;

    // Rule 3
    if (numNeighbors3 > 0) {
        perceived_velocity /= numNeighbors3;
        resultVel += perceived_velocity * rule3Scale;
    }

    resultVel += thisVel;

    // Clamp the speed
    float resultSpeed = glm::length(resultVel);
    if (resultSpeed > maxSpeed) {
        resultVel = glm::normalize(resultVel) * maxSpeed;
    }

    // Record the new velocity into vel2. 
    vel2[index] = resultVel;
}

// reorder particle data into array
// sortedParticleArrayIndices is after sorting by key
// rPos and rVel are reshuffled values
__global__ void kernReshuffleParticleData(int N, int* sortedParticleArrayIndices, glm::vec3* pos, glm::vec3* vel,
    glm::vec3* rPos, glm::vec3* rVel) {

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // original idx
    int oIdx = sortedParticleArrayIndices[index];
    rPos[index] = pos[oIdx];
    rVel[index] = vel[oIdx];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
    // timer
    float ms;
    cudaEvent_t startNaive, stopNaive, startNaiveSearch, stopNaiveSearch;
    cudaEventCreate(&startNaive);
    cudaEventCreate(&stopNaive);
    cudaEventCreate(&startNaiveSearch);
    cudaEventCreate(&stopNaiveSearch);
    cudaEventRecord(startNaive);

    // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    cudaEventRecord(startNaiveSearch);

    kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize>> > (numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

    cudaEventRecord(stopNaiveSearch);
    cudaEventSynchronize(stopNaiveSearch);
    cudaEventElapsedTime(&ms, startNaiveSearch, stopNaiveSearch);
    printf("naive search took %.3f ms\n", ms);
    cudaEventDestroy(startNaiveSearch);
    cudaEventDestroy(stopNaiveSearch);
    
    // TODO-1.2 ping-pong the velocity buffers
    std::swap(dev_vel1, dev_vel2);

    cudaDeviceSynchronize();

    cudaEventRecord(stopNaive);
    cudaEventSynchronize(stopNaive);
    cudaEventElapsedTime(&ms, startNaive, stopNaive);
    printf("naive simulation took %.3f ms\n", ms);
    cudaEventDestroy(startNaive);
    cudaEventDestroy(stopNaive);
}

void Boids::stepSimulationScatteredGrid(float dt) {
    // timer
    float ms;
    cudaEvent_t startScattered, stopScattered, startScatteredSearch, stopScatteredSearch;    
    cudaEventCreate(&startScattered);
    cudaEventCreate(&stopScattered);
    cudaEventCreate(&startScatteredSearch);
    cudaEventCreate(&stopScatteredSearch);
    cudaEventRecord(startScattered);

  // TODO-2.1
    // Uniform Grid Neighbor search using Thrust sort.
    // In Parallel:
    // - label each particle with its array index as well as its grid index.
    //   Use 2x width grids.
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 fullCellsPerGrid((gridCellCount + blockSize - 1) / blockSize);

    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, 
        gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    //   are welcome to do a performance comparison.
    thrust::device_ptr<int> key(dev_particleGridIndices);
    thrust::device_ptr<int> val(dev_particleArrayIndices);
    thrust::sort_by_key(key, key + numObjects, val);

    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    kernResetIntBuffer << <fullCellsPerGrid, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <fullCellsPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, 
        dev_gridCellStartIndices, dev_gridCellEndIndices);

    // - Perform velocity updates using neighbor search
    cudaEventRecord(startScatteredSearch);

    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount,
        gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

    cudaEventRecord(stopScatteredSearch);
    cudaEventSynchronize(stopScatteredSearch);
    cudaEventElapsedTime(&ms, startScatteredSearch, stopScatteredSearch);
    printf("scattered search took %.3f ms\n", ms);
    cudaEventDestroy(startScatteredSearch);
    cudaEventDestroy(stopScatteredSearch);

    // - Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

    // - Ping-pong buffers as needed
    std::swap(dev_vel1, dev_vel2);

    cudaEventRecord(stopScattered);
    cudaEventSynchronize(stopScattered);
    cudaEventElapsedTime(&ms, startScattered, stopScattered);
    printf("scattered simulation took %.3f ms\n", ms);
    cudaEventDestroy(startScattered);
    cudaEventDestroy(stopScattered);
}



void Boids::stepSimulationCoherentGrid(float dt) {
    // timer
    float ms;
    cudaEvent_t startCoherent, stopCoherent, startCoherentSearch, stopCoherentSearch;
    cudaEventCreate(&startCoherent);
    cudaEventCreate(&stopCoherent);
    cudaEventCreate(&startCoherentSearch);
    cudaEventCreate(&stopCoherentSearch);
    cudaEventRecord(startCoherent);

    // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
    // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
    // In Parallel:
    // - Label each particle with its array index as well as its grid index.
    //   Use 2x width grids
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 fullCellsPerGrid((gridCellCount + blockSize - 1) / blockSize);

    // compute grid idx
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount,
        gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices,
        dev_particleGridIndices);

    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    //   are welcome to do a performance comparison.
    thrust::device_ptr<int> key(dev_particleGridIndices);
    thrust::device_ptr<int> val(dev_particleArrayIndices);
    thrust::sort_by_key(key, key + numObjects, val);

    // reset grid cell start/end buffers
    kernResetIntBuffer << <fullCellsPerGrid, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <fullCellsPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, 
        dev_gridCellStartIndices, dev_gridCellEndIndices);

    // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
    //   the particle data in the simulation array.
    //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
    kernReshuffleParticleData << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleArrayIndices, 
        dev_pos, dev_vel1, dev_rPos, dev_rVel1);
    
    // - Perform velocity updates using neighbor search
    cudaEventRecord(startCoherentSearch);

    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount,
        gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_rPos, dev_rVel1, dev_rVel2);

    cudaEventRecord(stopCoherentSearch);
    cudaEventSynchronize(stopCoherentSearch);
    cudaEventElapsedTime(&ms, startCoherentSearch, stopCoherentSearch);
    printf("coherent search took %.3f ms\n", ms);
    cudaEventDestroy(startCoherentSearch);
    cudaEventDestroy(stopCoherentSearch);
    
    // - Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_rPos, dev_rVel2);

    // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
    std::swap(dev_rVel1, dev_rVel2);

    // sync coherent buffers back into main pointers
    std::swap(dev_pos, dev_rPos);
    std::swap(dev_vel1, dev_rVel1);
    std::swap(dev_vel2, dev_rVel2);

    cudaEventRecord(stopCoherent);
    cudaEventSynchronize(stopCoherent);
    cudaEventElapsedTime(&ms, startCoherent, stopCoherent);
    printf("coherent simulation took %.3f ms\n", ms);
    cudaEventDestroy(startCoherent);
    cudaEventDestroy(stopCoherent);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  cudaFree(dev_rVel1);
  cudaFree(dev_rVel2);
  cudaFree(dev_rPos);
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
