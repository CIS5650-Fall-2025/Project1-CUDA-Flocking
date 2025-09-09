#pragma once

namespace Boids {
    void initSimulation(int N);
    void stepSimulationNaive(float dt);
    void stepSimulationScatteredGrid(float dt);
    void stepSimulationCoherentGrid(float dt);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);
    int getBlockSize();
    int getGridLoopingOptimization();
    float getGridWidthScale();

    void endSimulation();
    void unitTest();
}
