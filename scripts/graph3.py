import matplotlib.pyplot as plt
import numpy as np

# Data
block_sizes = np.array([16, 32, 64, 128, 256, 512])
framerate = np.array([879.867, 1151.34, 1279.67, 1850.24, 1899.78, 1990.74])

# Plot
plt.figure(figsize=(10,6))
line, = plt.plot(block_sizes, framerate, 'o--', label="")

# Annotate points with same color as line
for x, y in zip(block_sizes, framerate):
    plt.text(x, y+40, f"{y:.2f}", fontsize=10, ha='center', color="black", fontweight="bold")

plt.xlabel("CUDA Block Size")
plt.ylabel("Framerate (FPS)")
plt.title("Framerate vs CUDA Block Size (100000 Boids, No Visualization)")
plt.ylim(750, 2100)  # keep scale consistent with earlier plots
plt.xticks(block_sizes)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()
