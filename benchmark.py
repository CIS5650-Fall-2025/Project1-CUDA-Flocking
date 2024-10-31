import subprocess
import itertools

executable = './build/bin/cis5650_boids'

simulations = [2]
boidcounts = [100000]
blocksizes = [1, 2, 8, 16, 32, 64, 128, 256, 512, 1024]

for (simulation, boidcount, blocksize) in itertools.product(simulations, boidcounts, blocksizes):
    command = [executable, str(simulation), str(boidcount), str(blocksize)]
    process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    time = float(process.stderr.decode('utf-8').strip())

    print(simulation, boidcount, blocksize, time)