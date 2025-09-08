#include <cuda.h>
#include <../device_launch_parameters.h>

#define HALF_GRID_WIDTH

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 1024

#define test_scale 2.f
// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance (5.0f * test_scale)
#define rule2Distance (3.0f * test_scale)
#define rule3Distance (5.0f * test_scale)

#define rule1Scale 0.01f
#define rule2Scale 0.1f 
#define rule3Scale 0.1f 

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale (100.0f * test_scale)

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3* pos,
	glm::vec3* vel1, glm::vec3* vel2);

__global__ void kernComputeIndices(int N, int gridResolution,
	glm::vec3 gridMin, float inverseCellWidth,
	glm::vec3* pos, int* particlePropertyIndex, int* particleGridIndex, glm::vec3* col);
__global__ void kernIdentifyCellStartEnd(int N, int* particleGridIndices,
	int* gridCellStartIndices, int* gridCellEndIndices);
__global__ void kernUpdateVelNeighborSearchScattered(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int* gridCellStartIndices, int* gridCellEndIndices,
	int* particleArrayIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2, glm::vec3* col);
__global__ void kernUpdateVelNeighborSearchCoherent(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int* gridCellStartIndices, int* gridCellEndIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2);
