#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "boidImpl.h"
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
glm::vec3* dev_col;

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
#ifdef HALF_GRID_WIDTH
    gridCellWidth = 5.f;
#else
  gridCellWidth = 2.0f * 5.f;
#endif
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  std::cout << "[gridCellCount] " << gridCellCount << std::endl;
  std::cout << "[particleSpawn] " << numObjects << std::endl;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.


  cudaMalloc((void**)&dev_col, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_col failed!");


  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");


  std::vector<int> particleArrayIndex;
  particleArrayIndex.resize(numObjects);
  for (int i = 0; i < numObjects; i++)
  {
      particleArrayIndex[i] = i;
  }
  cudaMemcpy(dev_particleArrayIndices, particleArrayIndex.data(), sizeof(int) * numObjects, cudaMemcpyHostToDevice);

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
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_col, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
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


/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
    glm::vec3* temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce <<<fullBlocksPerGrid, blockSize >>> (numObjects, dev_pos, dev_vel1, dev_vel2);
	kernUpdatePos <<<fullBlocksPerGrid, blockSize >>> (numObjects, dt, dev_pos, dev_vel2);

    cudaMemcpy(dev_col, dev_vel1, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
    checkCUDAErrorWithLine("memcpy back failed!");
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.

    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernComputeIndices <<<fullBlocksPerGrid, blockSize >>> (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, 
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices, dev_col);

    checkCUDAErrorWithLine("kernComputeIndices failed!");
    thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);
    thrust::device_ptr<int> dev_thrust_values(dev_particleArrayIndices);
    thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_values);
    checkCUDAErrorWithLine("thrust::sort_by_key failed!");

    cudaMemset(dev_gridCellStartIndices, 0, sizeof(int) * gridCellCount);
    cudaMemset(dev_gridCellEndIndices, 0, sizeof(int) * gridCellCount);

  // - Ping-pong buffers as needed

    glm::vec3* temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;
    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>> (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered <<<fullBlocksPerGrid, blockSize >>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
		dev_pos, dev_vel1, dev_vel2, dev_col);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");
    // - Update positions
    kernUpdatePos <<<fullBlocksPerGrid, blockSize >>> (numObjects, dt, dev_pos, dev_vel1);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

	cudaMemcpy(dev_col, dev_vel1, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
    checkCUDAErrorWithLine("cudaMemcpy dev_col failed!");


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
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices, dev_col);

    glm::vec3* temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;
    checkCUDAErrorWithLine("kernComputeIndices failed!");
    thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);
    thrust::device_ptr<glm::vec3> dev_thrust_vel1(dev_vel1);
    thrust::device_ptr<glm::vec3> dev_thrust_pos(dev_pos);
    thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, thrust::make_zip_iterator(dev_thrust_vel1, dev_thrust_pos));
    checkCUDAErrorWithLine("thrust::sort_by_key failed!");

    cudaMemset(dev_gridCellStartIndices, 0, sizeof(int) * gridCellCount);
    cudaMemset(dev_gridCellEndIndices, 0, sizeof(int) * gridCellCount);

    // - Ping-pong buffers as needed

    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

    // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchCoherent <<<fullBlocksPerGrid, blockSize >>> (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");
    // - Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    cudaMemcpy(dev_col, dev_vel1, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
    checkCUDAErrorWithLine("cudaMemcpy dev_col failed!");

}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);
  cudaFree(dev_col);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
}

void Boids::unitTest() {
  //// LOOK-1.2 Feel free to write additional tests here.

  //// test unstable sort
  //int *dev_intKeys;
  //int *dev_intValues;
  //int N = 10;

  //std::unique_ptr<int[]>intKeys{ new int[N] };
  //std::unique_ptr<int[]>intValues{ new int[N] };

  //intKeys[0] = 0; intValues[0] = 0;
  //intKeys[1] = 1; intValues[1] = 1;
  //intKeys[2] = 0; intValues[2] = 2;
  //intKeys[3] = 3; intValues[3] = 3;
  //intKeys[4] = 0; intValues[4] = 4;
  //intKeys[5] = 2; intValues[5] = 5;
  //intKeys[6] = 2; intValues[6] = 6;
  //intKeys[7] = 0; intValues[7] = 7;
  //intKeys[8] = 5; intValues[8] = 8;
  //intKeys[9] = 6; intValues[9] = 9;

  //cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  //checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  //cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  //checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  //dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  //std::cout << "before unstable sort: " << std::endl;
  //for (int i = 0; i < N; i++) {
  //  std::cout << "  key: " << intKeys[i];
  //  std::cout << " value: " << intValues[i] << std::endl;
  //}

  //// How to copy data to the GPU
  //cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  //cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  //// Wrap device vectors in thrust iterators for use with thrust.
  //thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  //thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  //// LOOK-2.1 Example for using thrust::sort_by_key
  //thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  //// How to copy data back to the CPU side from the GPU
  //cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  //cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  //checkCUDAErrorWithLine("memcpy back failed!");

  //std::cout << "after unstable sort: " << std::endl;
  //for (int i = 0; i < N; i++) {
  //  std::cout << "  key: " << intKeys[i];
  //  std::cout << " value: " << intValues[i] << std::endl;
  //}

  //// cleanup
  //cudaFree(dev_intKeys);
  //cudaFree(dev_intValues);
  //checkCUDAErrorWithLine("cudaFree failed!");
  //return;
}
