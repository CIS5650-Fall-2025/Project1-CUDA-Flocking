import matplotlib.pyplot as plt
import numpy as np

# Data
boids = np.array([500, 5000, 100000, 10000000])

coherent = np.array([416.21, 900.728, 1850.24, 1299.89])
scattered = np.array([315.12, 658.871, 1365.973, 1097.21])
naive_branch = np.array([200.21, 250.17, 125.19, 10.2168])
naive_mask = np.array([5.9, 5.1, 16.5, 0.2])


# Plot
plt.figure(figsize=(12,7))
line1, = plt.plot(boids, coherent, 'x--', label="Coherent Grid")
line2, = plt.plot(boids, scattered, 'd--', label="Scattered Grid")
line3, = plt.plot(boids, naive_branch, 's--', label="Naive (Branching)")
line4, = plt.plot(boids, naive_mask, 'o--', label="Naive (Mask-based)")

# Annotation with offsets to reduce overlap

lines = {
    "Coherent Grid": (line1, coherent, (0, 60)),
    "Scattered Grid": (line2, scattered, (0, 30)),
    "Naive (Branching)": (line3, naive_branch, (0, 45)),
    "Naive (Mask-based)": (line4, naive_mask, (0, -60)),
}

# Annotate with same color as line
for label, (line, values, (dx, dy)) in lines.items():
    color = line.get_color()
    for x, y in zip(boids, values):
        plt.text(x+dx, y+dy, f"{y:.2f}", fontsize=10, ha='center', color=color, fontweight='bold')

plt.xscale("log")
plt.xticks(boids, [str(b) for b in boids])

plt.ylim(-100, 2000)

plt.xlabel("Number of Boids (log-based x-scale)")
plt.ylabel("Framerate (FPS)")
plt.title("Framerate vs Number of Boids (No Visualization)", fontsize=18, fontweight='bold')
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()
