#include "boidImpl.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>



// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
	return x + y * gridResolution + z * gridResolution * gridResolution;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
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
	// Rule 2: boids try to stay a distance d away from each other
	// Rule 3: boids try to match the speed of surrounding boids
	glm::vec3 BoidPos = pos[iSelf];
	glm::vec3 AccumulatedPos = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 Propulsion = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 AccumulatedVelo = glm::vec3(0.0f, 0.0f, 0.0f);
	int Counts1 = 0;
	int Counts3 = 0;
	for (int i = 0; i < N; i++)
	{
		glm::vec3 CurPos = pos[i];
		float Dist = glm::length(BoidPos - CurPos);
		if (i != iSelf && Dist < rule1Distance)
		{
			AccumulatedPos += CurPos;
			Counts1++;
		}
		if (i != iSelf && Dist < rule2Distance)
		{
			Propulsion += (BoidPos - CurPos);
		}
		if (i != iSelf && Dist < rule3Distance)
		{
			AccumulatedVelo += vel[i];
			Counts3++;
		}
	}

	glm::vec3 dvel;
	if (Counts1>0)
	{
		dvel += rule1Scale * (AccumulatedPos / static_cast<float>(Counts1) - BoidPos);
	}
	if (Counts3>0)
	{
		dvel += rule3Scale * AccumulatedVelo / static_cast<float>(Counts3);
	}
	dvel += rule2Scale * Propulsion;
	return dvel;
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

	  // Update position by velocity
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	vel2[index] = vel1[index] + computeVelocityChange(N, index, pos, vel1);
	if (glm::length(vel2[index]) > maxSpeed)
	{
		vel2[index] = glm::normalize(vel2[index]) * maxSpeed;
	}
}

__global__ void kernIdentifyCellStartEnd(int N, int* particleGridIndices,
	int* gridCellStartIndices, int* gridCellEndIndices) {
	// TODO-2.1
	// Identify the start point of each cell in the gridIndices array.
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= N) {
		return;
	}
	if (tid==0)
	{
		gridCellStartIndices[0] = 0;
	}
	if(tid == N - 1)
	{
		gridCellEndIndices[particleGridIndices[tid]] = tid + 1;
	}
	if (tid > 0 && particleGridIndices[tid] != particleGridIndices[tid - 1])
	{
		gridCellStartIndices[particleGridIndices[tid]] = tid;
		gridCellEndIndices[particleGridIndices[tid - 1]] = tid;
	}
}

__device__ void accumulateOneCell(
	int tid, int iX, int iY, int iZ,
	int startIdx, int endIdx,
	int* particleArrayIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* col, 
	glm::vec3& InOutAccuVelo1, glm::vec3& InOutAccuVelo2, glm::vec3& InOutAccuVelo3,
	int& InOutCount1, int& InOutCount2, int& InOutCount3)
{
	glm::vec3 BoidPos = pos[tid];
	for (int i = startIdx; i < endIdx; i++)
	{
		int bufferIdx = particleArrayIndices[i];
		glm::vec3 CurPos = pos[bufferIdx];
		float Dist = glm::length(BoidPos - CurPos);
		if (tid != bufferIdx && Dist < rule1Distance)
		{
			InOutAccuVelo1 += CurPos;
			InOutCount1++;
		}
		if (tid != bufferIdx && Dist < rule2Distance)
		{
			InOutAccuVelo2 += (BoidPos - CurPos);
			InOutCount2++;
		}
		if (tid != bufferIdx && Dist < rule3Distance)
		{
			InOutAccuVelo3 += vel1[i];
			InOutCount3++;
		}
		
		col[bufferIdx].x = iX / 10 % 2;
		col[bufferIdx].y = 0;
		col[bufferIdx].z = 0;
	}
}

__global__ void kernUpdateVelNeighborSearchScattered(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int* gridCellStartIndices, int* gridCellEndIndices,
	int* particleArrayIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2, glm::vec3* col) {
	// TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
	// the number of boids that need to be checked.

	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= N) {
		return;
	}
	glm::vec3 BoidPos = pos[tid];
	glm::vec3 AccumulatedPos = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 Propulsion = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 AccumulatedVelo = glm::vec3(0.0f, 0.0f, 0.0f);
	int Counts1 = 0;
	int Counts2 = 0;
	int Counts3 = 0;
	int iX = floor((BoidPos.x - gridMin.x) * inverseCellWidth);
	int iY = floor((BoidPos.y - gridMin.y) * inverseCellWidth);
	int iZ = floor((BoidPos.z - gridMin.z) * inverseCellWidth);

	// - Identify the grid cell that this particle is in
	int maxGridIndex = gridIndex3Dto1D(gridResolution, gridResolution, gridResolution, gridResolution) - 1;
#ifndef HALF_GRID_WIDTH
	glm::vec3 posInCell;
	posInCell.x = BoidPos.x - (iX * cellWidth + gridMin.x);
	posInCell.y = BoidPos.y - (iY * cellWidth + gridMin.y);
	posInCell.z = BoidPos.z - (iZ * cellWidth + gridMin.z);
	int dX = (posInCell.x < cellWidth * 0.5f) ? -1 : 1;
	int dY = (posInCell.y < cellWidth * 0.5f) ? -1 : 1;
	int dZ = (posInCell.z < cellWidth * 0.5f) ? -1 : 1;

	int cellIndices[] = {

		gridIndex3Dto1D(iX,    iY,     iZ, gridResolution),
		gridIndex3Dto1D(iX+dX, iY,     iZ, gridResolution),
		gridIndex3Dto1D(iX,    iY+ dY, iZ, gridResolution),
		gridIndex3Dto1D(iX+ dX,iY+ dY, iZ, gridResolution),
		gridIndex3Dto1D(iX,    iY,     iZ+ dZ, gridResolution),
		gridIndex3Dto1D(iX+ dX,iY,     iZ+ dZ, gridResolution),
		gridIndex3Dto1D(iX,    iY+ dY, iZ+ dZ, gridResolution),
		gridIndex3Dto1D(iX+ dX,iY+ dY, iZ+ dZ, gridResolution),
	};
#else
	int cellIndices[27];
	int cellId = 0;
	for (int dX : {-1, 0, 1})
	{
		for (int dY : {-1, 0, 1})
		{
			for (int dZ : {-1, 0, 1})
			{
				cellIndices[cellId++] = gridIndex3Dto1D(iX + dX, iY + dY, iZ + dZ, gridResolution);
			}
		}
	}
#endif
	for (int i = 0;i<8;i++)
	{
		if (cellIndices[i] < 0 || cellIndices[i]>maxGridIndex)
		{
			continue;
		}
		accumulateOneCell(tid, iX, iY, iZ,
			gridCellStartIndices[cellIndices[i]],
			gridCellEndIndices[cellIndices[i]], particleArrayIndices, pos, vel1, col,
			AccumulatedPos, Propulsion, AccumulatedVelo,
			Counts1, Counts2, Counts3);
	}

	glm::vec3 newVel = vel1[tid];
	if (Counts1>0)
	{
		newVel +=  rule1Scale * (AccumulatedPos / static_cast<float>(Counts1) - BoidPos);
	}
	if (Counts2>0)
	{
		newVel += rule2Scale * Propulsion;
	}
	if (Counts3 > 0)
	{
		newVel += rule3Scale * AccumulatedVelo / static_cast<float>(Counts3);
	}
	if (glm::length(newVel) > maxSpeed)
	{
		newVel = glm::normalize(newVel) * maxSpeed;
	}
	vel2[tid] = newVel;
	// - Identify which cells may contain neighbors. This isn't always 8.
	// - For each cell, read the start/end indices in the boid pointer array.
	// - Access each boid in the cell and compute velocity change from
	//   the boids rules, if this boid is within the neighborhood distance.
	// - Clamp the speed change before putting the new speed in vel2
}

__device__ void accumulateOneCellCoherent(
	int tid, int iX, int iY, int iZ,
	int startIdx, int endIdx,
	glm::vec3* pos, glm::vec3* vel1, 
	glm::vec3& InOutAccuVelo1, glm::vec3& InOutAccuVelo2, glm::vec3& InOutAccuVelo3,
	int& InOutCount1, int& InOutCount2, int& InOutCount3)
{
	glm::vec3 BoidPos = pos[tid];
	for (int i = startIdx; i < endIdx; i++)
	{
		glm::vec3 CurPos = pos[i];
		float Dist = glm::length(BoidPos - CurPos);
		if (tid != i && Dist < rule1Distance)
		{
			InOutAccuVelo1 += CurPos;
			InOutCount1++;
		}
		if (tid != i && Dist < rule2Distance)
		{
			InOutAccuVelo2 -= (CurPos -BoidPos);
			InOutCount2++;
		}
		if (tid != i && Dist < rule3Distance)
		{
			InOutAccuVelo3 += vel1[i];
			InOutCount3++;
		}
	}
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

	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= N) {
		return;
	}
	glm::vec3 BoidPos = pos[tid];
	glm::vec3 AccumulatedPos = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 Propulsion = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 AccumulatedVelo = glm::vec3(0.0f, 0.0f, 0.0f);
	int Counts1 = 0;
	int Counts2 = 0;
	int Counts3 = 0;
	int iX = floor((BoidPos.x - gridMin.x) * inverseCellWidth);
	int iY = floor((BoidPos.y - gridMin.y) * inverseCellWidth);
	int iZ = floor((BoidPos.z - gridMin.z) * inverseCellWidth);

	// - Identify the grid cell that this particle is in
	// current cell 

	glm::vec3 posInCell;
	posInCell.x = BoidPos.x - (iX * cellWidth + gridMin.x);
	posInCell.y = BoidPos.y - (iY * cellWidth + gridMin.y);
	posInCell.z = BoidPos.z - (iZ * cellWidth + gridMin.z);

	int dX = (posInCell.x < cellWidth * 0.5f) ? -1 : 1;
	int dY = (posInCell.y < cellWidth * 0.5f) ? -1 : 1;
	int dZ = (posInCell.z < cellWidth * 0.5f) ? -1 : 1;

	int maxGridIndex = gridIndex3Dto1D(gridResolution, gridResolution, gridResolution, gridResolution) - 1;

	int cellIndices[] = {
		gridIndex3Dto1D(iX,      iY,      iZ, gridResolution),
		gridIndex3Dto1D(iX + dX, iY,      iZ, gridResolution),
		gridIndex3Dto1D(iX,      iY + dY, iZ, gridResolution),
		gridIndex3Dto1D(iX + dX, iY + dY, iZ, gridResolution),
		gridIndex3Dto1D(iX,      iY,      iZ + dZ, gridResolution),
		gridIndex3Dto1D(iX + dX, iY,      iZ + dZ, gridResolution),
		gridIndex3Dto1D(iX,      iY + dY, iZ + dZ, gridResolution),
		gridIndex3Dto1D(iX + dX, iY + dY, iZ + dZ, gridResolution),
	};

	for (int i = 0; i < 8; i++)
	{
		if (cellIndices[i] < 0 || cellIndices[i]>maxGridIndex)
		{
			continue;
		}
		accumulateOneCellCoherent(tid, iX, iY, iZ,
			gridCellStartIndices[cellIndices[i]],
			gridCellEndIndices[cellIndices[i]], pos, vel1,
			AccumulatedPos, Propulsion, AccumulatedVelo,
			Counts1, Counts2, Counts3);
	}
	glm::vec3 newVel;
	newVel += vel1[tid];
	if (Counts1 > 0)
	{
		newVel += rule1Scale * (AccumulatedPos / static_cast<float>(Counts1) - BoidPos);
	}
	newVel += rule2Scale * Propulsion;
	if (Counts3 > 0)
	{
		newVel += rule3Scale * AccumulatedVelo / static_cast<float>(Counts3);
	}
	if (glm::length(newVel) > maxSpeed)
	{
		newVel = glm::normalize(newVel) * maxSpeed;
	}
	vel2[tid] = newVel;

}
__global__ void kernComputeIndices(int N, int gridResolution,
	glm::vec3 gridMin, float inverseCellWidth,
	glm::vec3* pos, int* particlePropertyIndex, int* particleGridIndex, glm::vec3* col) {
	// TODO-2.
	// - Label each boid with the index of its grid cell.
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= N) {
		return;
	}
	int bufferIndex = particlePropertyIndex[tid];
	int iX = floor((pos[bufferIndex].x - gridMin.x) * inverseCellWidth);
	int iY = floor((pos[bufferIndex].y - gridMin.y) * inverseCellWidth);
	int iZ = floor((pos[bufferIndex].z - gridMin.z) * inverseCellWidth);
	// - Set up a parallel array of integer indices as pointers to the actual
	//   boid data in pos and vel1/vel2
	particleGridIndex[tid] = gridIndex3Dto1D(iX, iY, iZ, gridResolution);
}