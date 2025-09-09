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
void checkCUDAError(const char* msg, int line = -1) {
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
#define blockSize 128  //Originally 128

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
glm::vec3* dev_pos;
glm::vec3* dev_vel1;
glm::vec3* dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int* dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int* dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int* dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int* dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_pos_coherent;    // positions reordered to match sorted-by-cell order
glm::vec3* dev_vel1_coherent;   // "current" velocities in coherent order (input to neighbor search)
glm::vec3* dev_vel2_coherent;   // "next" velocities in coherent order (output of neighbor search)

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
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3* arr, float scale) {
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
    kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> > (1, numObjects,
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


    ///////////////////////////////////////////////////////////////////////////
      //TODO-2.1 - Allocate memory for uniform grid data structures

    // Allocate memory for the particle array indices and grid indices
    cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

    // Allocate memory for the particle grid indices
    cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

    // Set up the thrust pointers
    cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

    // Allocate memory for the grid cell end indices
    cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

    ///////////////////////////////////////////////////////////////////////////
    // TODO-2.3 - Allocate additional buffers here.
    // Allocate memory for the coherent position and velocity buffers
    cudaMalloc((void**)&dev_pos_coherent, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_pos_coherent failed!");

    cudaMalloc((void**)&dev_vel1_coherent, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_vel1_coherent failed!");

    cudaMalloc((void**)&dev_vel2_coherent, N * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_vel2_coherent failed!");


    cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3* pos, float* vbo, float s_scale) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale = -1.0f / s_scale;

    if (index < N) {
        vbo[4 * index + 0] = pos[index].x * c_scale;
        vbo[4 * index + 1] = pos[index].y * c_scale;
        vbo[4 * index + 2] = pos[index].z * c_scale;
        vbo[4 * index + 3] = 1.0f;
    }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3* vel, float* vbo, float s_scale) {
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
void Boids::copyBoidsToVBO(float* vbodptr_positions, float* vbodptr_velocities) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, vbodptr_positions, scene_scale);
    kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_vel1, vbodptr_velocities, scene_scale);

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
    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    glm::vec3 perceived_center(0.0f, 0.0f, 0.0f);
    int rule1_neighbors = 0; //number of neighbors within a certain distance for rule 1

    // Rule 2: boids try to stay a distance d away from each other
    glm::vec3 c(0.0f, 0.0f, 0.0f); // the center of mass of the local neighborhood

    // Rule 3: boids try to match the speed of surrounding boids
    glm::vec3 perceived_velocity(0.0f, 0.0f, 0.0f);
    int rule3_neighbors = 0; //number of neighbors within a certain distance for rule 3

    // Implementation of the 3 boids rules

    // Loop over all boids
    for (int i = 0; i < N; i++) {
        if (i != iSelf) {
            float distance = glm::distance(pos[i], pos[iSelf]); //distance between the two boids

            // Rule1: Cohesion - Move towards perceived center
            if (distance < rule1Distance) {
                perceived_center += pos[i];
                rule1_neighbors++;
            }

            // Rule2: Separation - Avoid crowding neighbors
            if (distance < rule2Distance) {
                c -= (pos[i] - pos[iSelf]);
            }

            // Rule 3: Alignment - Match velocity with nearby boids
            if (distance < rule3Distance) {
                perceived_velocity += vel[i];
                rule3_neighbors++;
            }
        }
    }

    glm::vec3 result(0.0f, 0.0f, 0.0f); // The change in velocity

    // Apply Rule 1
    if (rule1_neighbors > 0) {
        perceived_center /= rule1_neighbors;  // Average position of neighbors => perceived_center /= number_of_neighbors
        result += (perceived_center - pos[iSelf]) * rule1Scale; // Move towards the perceived center => (perceived_center - boid.position) * rule1Scale
    }

    // Apply Rule 2
    result += c * rule2Scale; // Move away from neighbors => c * rule2Scale

    // Apply Rule 3
    if (rule3_neighbors > 0) {
        perceived_velocity /= rule3_neighbors; // Average velocity of neighbors => perceived_velocity /= number_of_neighbors
        result += perceived_velocity * rule3Scale; // Match the perceived velocity => perceived_velocity * rule3Scale
    }

    return result; // Return the total change in velocity
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
    /*
    We’re doing a read‑old and write‑new update
    Every thread must see the same old velocities (vel1) while computing
    if we write back into vel1 we will read partially‑updated data from other threads in the same step causing race behavior
    */

    int index = threadIdx.x + (blockIdx.x * blockDim.x); // Current boid index

    // Ensure we don't go out of bounds
    if (index >= N) {
        return;
    }

    // Compute velocity change based on three rules
    glm::vec3 velocityChange = computeVelocityChange(N, index, pos, vel1);

    // Update velocity and clamp to max speed
    glm::vec3 newVel = vel1[index] + velocityChange; // New velocity after applying rules
    float speed = glm::length(newVel); // Calculate the speed (magnitude of velocity)

    if (speed > maxSpeed) {
        newVel = (newVel / speed) * maxSpeed; // Clamp to max speed
    }

    vel2[index] = newVel; // Store new velocity in vel2 for ping-ponging
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3* pos, glm::vec3* vel) {
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
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

    // Compute grid indices for each particle
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Ensure we don't go out of bounds
    if (index >= N) {
        return;
    }

    // Compute 3D grid index for this particle
    glm::vec3 gridPos = (pos[index] - gridMin) * inverseCellWidth; // Convert position to grid coordinates
    int gridX = (int)floor(gridPos.x);
    int gridY = (int)floor(gridPos.y);
    int gridZ = (int)floor(gridPos.z);

    // Clamp to grid bounds
    gridX = imax(0, imin(gridX, gridResolution - 1));
    gridY = imax(0, imin(gridY, gridResolution - 1));
    gridZ = imax(0, imin(gridZ, gridResolution - 1));

    // Store particle array index and grid index
    indices[index] = index;
    gridIndices[index] = gridIndex3Dto1D(gridX, gridY, gridZ, gridResolution);
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < N) {
        intBuffer[index] = value;
    }
}

__global__ void kernIdentifyCellStartEnd(int N, int* particleGridIndices,
    int* gridCellStartIndices, int* gridCellEndIndices) {
    // TODO-2.1
    // Identify the start point of each cell in the gridIndices array.
    // This is basically a parallel unrolling of a loop that goes
    // "this index doesn't match the one before it, must be a new cell!"

    int index = (blockIdx.x * blockDim.x) + threadIdx.x; // Current particle index

    // Ensure we don't go out of bounds
    if (index >= N) {
        return;
    }

    int currentGridIndex = particleGridIndices[index]; // Grid index of the current particle

    // If this is the first particle, it marks the start of its grid cell
    if (index == 0) {
        gridCellStartIndices[currentGridIndex] = index; // Start of current cell

    }
    else {
        int previousGridIndex = particleGridIndices[index - 1]; // Grid index of the previous particle

        // If the grid index changes, mark the end of the previous cell and start of the new cell
        if (currentGridIndex != previousGridIndex) {
            gridCellEndIndices[previousGridIndex] = index; // End of previous cell
            gridCellStartIndices[currentGridIndex] = index; // Start of current cell
        }
    }

    // If this is the last particle, it marks the end of its grid cell
    if (index == N - 1) {
        gridCellEndIndices[currentGridIndex] = index + 1; // End is exclusive
    }
}

////////////////////////////////////////////////////////////////////////
// Helper functions

// Reorder position/velocity into coherent arrays based on the sorted index mapping.
// sortedIdx => originalIdx mapping lives in particleArrayIndices[sortedIdx].
__global__ void kernReorderDataToCoherent(int N,
    const int* __restrict__ particleArrayIndices,
    const glm::vec3* __restrict__ pos_in,
    const glm::vec3* __restrict__ vel1_in,
    glm::vec3* __restrict__ pos_out,
    glm::vec3* __restrict__ vel1_out) {

    int sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sortedIdx >= N) return;

    const int originalIdx = particleArrayIndices[sortedIdx];
    pos_out[sortedIdx] = pos_in[originalIdx];
    vel1_out[sortedIdx] = vel1_in[originalIdx];
}

// Scatter newly computed velocities from coherent order back to original order.
// particleArrayIndices[sortedIdx] tells us where that coherent item came from.
__global__ void kernScatterCoherentVelToUnsorted(int N,
    const int* __restrict__ particleArrayIndices,
    const glm::vec3* __restrict__ vel2_coherent,
    glm::vec3* __restrict__ vel2_unsorted) {

    int sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sortedIdx >= N) return;

    const int originalIdx = particleArrayIndices[sortedIdx];
    vel2_unsorted[originalIdx] = vel2_coherent[sortedIdx];
}

////////////////////////////////////////////////////////////////////////

__global__ void kernUpdateVelNeighborSearchScattered(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices,
    int* particleArrayIndices,
    glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
    // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
    // the number of boids that need to be checked.
    // - Identify the grid cell that this particle is in
    // - Identify which cells may contain neighbors. This isn't always 8.
    // - For each cell, read the start/end indices in the boid pointer array.
    // - Access each boid in the cell and compute velocity change from
    //   the boids rules, if this boid is within the neighborhood distance.
    // - Clamp the speed change before putting the new speed in vel2

      // Grid-based neighbor search with scattered memory access
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Ensure we don't go out of bounds
    if (index >= N) {
        return;
    }

    glm::vec3 thisPos = pos[index]; // Current particle position
    glm::vec3 thisVel = vel1[index]; // Current particle velocity

    // Get grid position
    glm::vec3 gridPos = (thisPos - gridMin) * inverseCellWidth;
    int gridX = (int)floor(gridPos.x);
    int gridY = (int)floor(gridPos.y);
    int gridZ = (int)floor(gridPos.z);

    // Rule accumulators
    glm::vec3 perceived_center(0.0f); // Rule 1
    int rule1_neighbors = 0; //number of neighbors within a certain distance for rule 1

    glm::vec3 c(0.0f); // Rule 2

    glm::vec3 perceived_velocity(0.0f); // Rule 3
    int rule3_neighbors = 0; //number of neighbors within a certain distance for rule 3

    // Check neighboring cells (8 cells for cellWidth = 2 * searchRadius)

    // Iterate over neighboring cells in a 3x3x3 cube around the current cell
    for (int offsetX = -1; offsetX <= 1; offsetX++) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {

                // Compute neighbor cell coordinates
                int neighborX = gridX + offsetX;
                int neighborY = gridY + offsetY;
                int neighborZ = gridZ + offsetZ;

                // Skip if out of bounds
                if (neighborX < 0 || neighborX >= gridResolution ||
                    neighborY < 0 || neighborY >= gridResolution ||
                    neighborZ < 0 || neighborZ >= gridResolution) {
                    continue;
                }

                // Get start/end indices of boids in this neighbor cell
                int neighborGridIndex = gridIndex3Dto1D(neighborX, neighborY, neighborZ, gridResolution);
                int startIndex = gridCellStartIndices[neighborGridIndex];
                int endIndex = gridCellEndIndices[neighborGridIndex];

                // Skip empty cells
                if (startIndex == -1) {
                    continue;
                }

                // Check all particles in this grid cell
                for (int i = startIndex; i <= endIndex; i++) {
                    int neighborIndex = particleArrayIndices[i];

                    // Skip self
                    if (neighborIndex != index) {
                        glm::vec3 neighborPos = pos[neighborIndex];
                        float distance = glm::distance(neighborPos, thisPos);

                        // Rule 1: Cohesion - Move towards perceived center
                        if (distance < rule1Distance) {
                            perceived_center += neighborPos;
                            rule1_neighbors++;
                        }

                        // Rule 2: Separation - Avoid crowding neighbors
                        if (distance < rule2Distance) {
                            c -= (neighborPos - thisPos);
                        }

                        // Rule 3: Alignment - Match velocity with nearby boids
                        if (distance < rule3Distance) {
                            perceived_velocity += vel1[neighborIndex];
                            rule3_neighbors++;
                        }
                    }
                }
            }
        }
    }

    // Compute final velocity change
    glm::vec3 result(0.0f);

    // Apply Rule 1
    if (rule1_neighbors > 0) {
        perceived_center /= rule1_neighbors;
        result += (perceived_center - thisPos) * rule1Scale;
    }

    // Apply Rule 2
    result += c * rule2Scale;

    // Apply Rule 3
    if (rule3_neighbors > 0) {
        perceived_velocity /= rule3_neighbors;
        result += perceived_velocity * rule3Scale;
    }

    // Update velocity and clamp
    glm::vec3 newVel = thisVel + result; // New velocity after applying rules
    float speed = glm::length(newVel); // Calculate the speed (magnitude of velocity)
    if (speed > maxSpeed) {
        newVel = (newVel / speed) * maxSpeed; // Clamp to max speed
    }

    // Write new velocity to vel2
    vel2[index] = newVel;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices,
    glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
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


    const int index = blockIdx.x * blockDim.x + threadIdx.x; // Current boid index

    if (index >= N) return; // Ensure we don't go out of bounds

    // These arrays are already in cell-sorted (coherent) order.
    const glm::vec3 currentPos = pos[index]; // Current boid position
    const glm::vec3 currentVel = vel1[index]; // Current boid velocity

    // Figure out which grid cell this boid lives in
    const glm::vec3 gridPos = (currentPos - gridMin) * inverseCellWidth; // Convert position to grid coordinates
    const int gridX = (int)floor(gridPos.x);
    const int gridY = (int)floor(gridPos.y);
    const int gridZ = (int)floor(gridPos.z);

    // Accumulators for Boids rules

    glm::vec3 perceived_center(0.0f);    int rule1_neighbors = 0;  // Rule 1 (cohesion)
    glm::vec3 c(0.0f);                         // Rule 2 (separation)
    glm::vec3 perceived_velocity(0.0f);    int rule3_neighbors = 0;  // Rule 3 (alignment)

    // Visit neighboring cells in a 3x3x3 block
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetY = -1; offsetY <= 1; ++offsetY) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {

                // Compute neighbor cell coordinates
                int neighborX = gridX + offsetX;
                int neighborY = gridY + offsetY;
                int neighborZ = gridZ + offsetZ;

                // Skip if out of bounds
                if (neighborX < 0 || neighborX >= gridResolution ||
                    neighborY < 0 || neighborY >= gridResolution ||
                    neighborZ < 0 || neighborZ >= gridResolution) {
                    continue;
                }

                // Compute the 1D index of the neighbor cell
                const int neighborCell1D = gridIndex3Dto1D(neighborX, neighborY, neighborZ, gridResolution);

                // Look up the contiguous run [start, end) of boids in this cell
                const int startIndex = gridCellStartIndices[neighborCell1D]; // INCLUSIVE
                const int endIndex = gridCellEndIndices[neighborCell1D];  // EXCLUSIVE

                if (startIndex == -1) continue; // empty cell

                // Sequentially scan neighbors in this cell (coalesced reads)
                for (int j = startIndex; j < endIndex; ++j) {

                    if (j == index) continue;  // skip self (same coherent index)

                    const glm::vec3 neighborPos = pos[j];
                    const float distanceBoid = glm::distance(neighborPos, currentPos);

                    // Rule 1: Cohesion – move toward center of mass of neighbors within rule1Distance
                    if (distanceBoid < rule1Distance) {
                        perceived_center += neighborPos;
                        ++rule1_neighbors;
                    }

                    // Rule 2: Separation – avoid crowding neighbors within rule2Distance
                    if (distanceBoid < rule2Distance) {
                        c -= (neighborPos - currentPos);
                    }

                    // Rule 3: Alignment – align velocity with neighbors within rule3Distance
                    if (distanceBoid < rule3Distance) {
                        perceived_velocity += vel1[j];
                        ++rule3_neighbors;
                    }
                }
            }
        }
    }

    // Combine rule contributions
    glm::vec3 delta_v(0.0f); // change in velocity (delta v)

    //rule 1
    if (rule1_neighbors > 0) {
        perceived_center /= rule1_neighbors; // Average position of neighbors
        delta_v += (perceived_center - currentPos) * rule1Scale; // Move towards the perceived center
    }

    //rule 2
    delta_v += c * rule2Scale; // Move away from neighbors

    //rule 3
    if (rule3_neighbors > 0) {
        perceived_velocity /= rule3_neighbors; // Average velocity of neighbors
        delta_v += perceived_velocity * rule3Scale; // Match the perceived velocity
    }

    // Clamp speed
    glm::vec3 outVel = currentVel + delta_v; // New velocity after applying rules
    const float speed = glm::length(outVel); // Calculate the speed (magnitude of velocity)

    if (speed > maxSpeed) {
        outVel = (outVel / speed) * maxSpeed; // Clamp to max speeds
    }

    // Write new velocity in coherent order
    vel2[index] = outVel;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Extra credit - grid loop optimization

// For this kernel, the uniform grid is constructed such that the cell width is
__global__ void kernUpdateVelNeighborSearchCoherentOptimized(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices,
    glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2)
{
 
	int index = (blockIdx.x * blockDim.x) + threadIdx.x; // Current boid index

	if (index >= N) return; // Ensure we don't go out of bounds

    // Current boid’s position and velocity (coherent arrays)
	glm::vec3 currentPos = pos[index]; // Current boid position
	glm::vec3 currentVel = vel1[index]; // Current boid velocity

    // Determine this boid’s grid cell coordinates
	glm::vec3 gridPos = (currentPos - gridMin) * inverseCellWidth; // Convert position to grid coordinates
    const int gridX = (int)floor(gridPos.x);
    const int gridY = (int)floor(gridPos.y);
    const int gridZ = (int)floor(gridPos.z);

    // Compute search radius (max neighbor influence distance)
    float maxDistance = fmaxf(rule1Distance, fmaxf(rule2Distance, rule3Distance));

    // Determine grid cell index range in each dimension that lies within `maxDistance`
    int minCellX = static_cast<int>(floorf((currentPos.x - gridMin.x - maxDistance) * inverseCellWidth));
    int maxCellX = static_cast<int>(floorf((currentPos.x - gridMin.x + maxDistance) * inverseCellWidth));
    int minCellY = static_cast<int>(floorf((currentPos.y - gridMin.y - maxDistance) * inverseCellWidth));
    int maxCellY = static_cast<int>(floorf((currentPos.y - gridMin.y + maxDistance) * inverseCellWidth));
    int minCellZ = static_cast<int>(floorf((currentPos.z - gridMin.z - maxDistance) * inverseCellWidth));
    int maxCellZ = static_cast<int>(floorf((currentPos.z - gridMin.z + maxDistance) * inverseCellWidth));

    // Clamp the cell index ranges to the grid bounds
    if (minCellX < 0)              minCellX = 0;
    if (minCellY < 0)              minCellY = 0;
    if (minCellZ < 0)              minCellZ = 0;
    if (maxCellX >= gridResolution) maxCellX = gridResolution - 1;
    if (maxCellY >= gridResolution) maxCellY = gridResolution - 1;
    if (maxCellZ >= gridResolution) maxCellZ = gridResolution - 1;

    // Accumulators for the three Boids rules
    glm::vec3 perceived_center(0.0f);  int rule1_neighbors = 0;   // Cohesion
    glm::vec3 c(0.0f);                                           // Separation
    glm::vec3 perceived_velocity(0.0f); int rule3_neighbors = 0;  // Alignment

    // Loop over all candidate neighbor cells in the computed range
    for (int z = minCellZ; z <= maxCellZ; ++z) {
        for (int y = minCellY; y <= maxCellY; ++y) {
            for (int x = minCellX; x <= maxCellX; ++x) {

				
                int cellIndex = gridIndex3Dto1D(x, y, z, gridResolution); // Get the 1D index of this neighbor cell
				int startIndex = gridCellStartIndices[cellIndex]; // INCLUSIVE
				int endIndex = gridCellEndIndices[cellIndex];// EXCLUSIVE

                if (startIndex == -1) continue;  // skip empty cells

                // Check all boids in this cell
                for (int j = startIndex; j < endIndex; ++j) {
                    if (j == index) continue;  // skip itself

                    // Compute distance to this neighbor
					glm::vec3 neighborPos = pos[j]; // Neighbor boid position
					float dist = glm::distance(neighborPos, currentPos); // Distance to neighbor


                    if (dist < rule1Distance) {  //rules1 - Cohesion
                        perceived_center += neighborPos;
                        rule1_neighbors++;
                    }

                    if (dist < rule2Distance) {  // rule2 - Separation
                        c -= (neighborPos - currentPos);
                    }

                    if (dist < rule3Distance) {  // rule3 -Alignment
                        perceived_velocity += vel1[j];
                        rule3_neighbors++;
                    }
                }
            }
        }
    }

    // Apply the three rules to compute velocity change
	glm::vec3 deltaV(0.0f); // Change in velocity (delta v)

	// Rule 1: Cohesion
    if (rule1_neighbors > 0) {
        perceived_center /= rule1_neighbors;
        deltaV += (perceived_center - currentPos) * rule1Scale;
    }

	// Rule 2: Separation
    deltaV += c * rule2Scale;

	// Rule 3: Alignment
    if (rule3_neighbors > 0) {
        perceived_velocity /= rule3_neighbors;
        deltaV += perceived_velocity * rule3Scale;
    }

    // Clamp the new speed to maxSpeed
    glm::vec3 newVel = currentVel + deltaV;
    float speed = glm::length(newVel);
    if (speed > maxSpeed) {
        newVel = (newVel / speed) * maxSpeed;
    }

    // Write back the new velocity (coherent order)
    vel2[index] = newVel;
}


//////////////////////////////////////////////////////////////////////////////////////////

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
    // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
    // TODO-1.2 ping-pong the velocity buffers
      //Naive method

    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize); // Number of blocks needed

    // Update velocity (naive O(N^2) neighbor search)
    kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

    // Integrate positions using the NEW velocities in dev_vel2s
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    // Swap velocity buffers
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

    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize); // Number of blocks needed
    dim3 gridBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize); // Number of blocks for grid cells

    // Reset grid indices
    kernResetIntBuffer << <gridBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);  // Reset start indices to -1
    kernResetIntBuffer << <gridBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1); // Reset end indices to -1
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");

    // Compute grid indices for each particle
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount,
        gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices); // Label each particle with its array index and grid index
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    // Sort particles by grid index
    dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices); // Set up the thrust pointers
    dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices); // Set up the thrust pointers
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices); // Sort based on grid indices

    // Identify cell start and end
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects,
        dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices); // Identify start/end of each cell
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    // Update velocities using grid
    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
        dev_pos, dev_vel1, dev_vel2); // Update velocities based on neighbor search
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

    // Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2); // Update positions
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    // Swap velocity buffers
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

    // Blocks for boids vs grid
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 gridBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize);

    // 1) Clear cell start/end to empty (-1)
    kernResetIntBuffer << <gridBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <gridBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");

    // 2) Label each boid with (a) its original index and (b) its grid cell id
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    // 3) Sort by grid cell (keys: cell ids, values: boid indices)
    dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices);

    // 4) Build cell start/end tables over the sorted cell-id array
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (
        numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    // 5) Reorder position/velocity into coherent arrays (contiguous per cell)
    kernReorderDataToCoherent << <fullBlocksPerGrid, blockSize >> > (
        numObjects, dev_particleArrayIndices,
        dev_pos, dev_vel1,
        dev_pos_coherent, dev_vel1_coherent);
    checkCUDAErrorWithLine("kernReorderDataToCoherent failed!");

    // 6) Neighbor search directly over coherent arrays, write vel2_coherent
    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_pos_coherent, dev_vel1_coherent, dev_vel2_coherent);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

    // 7) Scatter new velocities back to original order for integration
    kernScatterCoherentVelToUnsorted << <fullBlocksPerGrid, blockSize >> > (
        numObjects, dev_particleArrayIndices, dev_vel2_coherent, dev_vel2);
    checkCUDAErrorWithLine("kernScatterCoherentVelToUnsorted failed!");

    // 8) Integrate positions with the new velocities (unsorted arrays)
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    // 9) Ping-pong velocity buffers (dev_vel1 holds newest after swap)
    std::swap(dev_vel1, dev_vel2);
}

///////////////////////////////////////////////////////////////////////
// Extra Credit - grid loop optimization


void Boids::stepSimulationGridLoopOptimized(float dt) {

    // Launch configurations
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 gridBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize);

    // 1) Reset grid cell start and end indices to -1 (empty)
    kernResetIntBuffer << <gridBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <gridBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");

    // 2) Compute grid indices for each boid (map each particle to its grid cell)
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    // 3) Sort boids by grid index using Thrust (cell indices as keys, boid indices as values)
    dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices);

    // 4) Identify the start and end index of each cell’s boid list
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (
        numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    // 5) Reorder boid data into contiguous memory by cell (coherent arrays)
    kernReorderDataToCoherent << <fullBlocksPerGrid, blockSize >> > (
        numObjects, dev_particleArrayIndices,
        dev_pos, dev_vel1,
        dev_pos_coherent, dev_vel1_coherent);
    checkCUDAErrorWithLine("kernReorderDataToCoherent failed!");

    // 6) Update velocities using dynamic grid-loop neighbor search (optimized kernel)
    kernUpdateVelNeighborSearchCoherentOptimized << <fullBlocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_pos_coherent, dev_vel1_coherent, dev_vel2_coherent);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherentOptimized failed!");

    // 7) Scatter the new velocities from coherent order back to original order
    kernScatterCoherentVelToUnsorted << <fullBlocksPerGrid, blockSize >> > (
        numObjects, dev_particleArrayIndices, dev_vel2_coherent, dev_vel2);
    checkCUDAErrorWithLine("kernScatterCoherentVelToUnsorted failed!");

    // 8) Integrate positions using the updated velocities (dev_vel2)
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    // 9) Ping-pong the velocity buffers for next iteration (make dev_vel1 current)
    std::swap(dev_vel1, dev_vel2);
}


////////////////////////////////////////////////////////////////////////////

void Boids::endSimulation() {
    cudaFree(dev_vel1);
    cudaFree(dev_vel2);
    cudaFree(dev_pos);

    // TODO-2.1 TODO-2.3 - Free any additional buffers here.

    //TODO-2.1 - Free the memory allocated for the uniform grid data structures
    cudaFree(dev_particleArrayIndices); // Free particle array indices
    cudaFree(dev_particleGridIndices); // Free particle grid indices
    cudaFree(dev_gridCellStartIndices); // Free grid cell start indices
    cudaFree(dev_gridCellEndIndices); // Free grid cell end indices

    //TODO-2.3 - Free the additional buffers for coherent grid
    cudaFree(dev_pos_coherent); // Free coherent position buffer
    cudaFree(dev_vel1_coherent); // Free coherent velocity buffer
    cudaFree(dev_vel2_coherent); // Free coherent velocity buffer
}

void Boids::unitTest() {
    // LOOK-1.2 Feel free to write additional tests here.

    // test unstable sort
    int* dev_intKeys;
    int* dev_intValues;
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