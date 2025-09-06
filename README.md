**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Crystal Jin
  * [LinkedIn](www.linkedin.com/in/xiaoyue-jin), [personal website](xiaoyuejin.com)
* Tested on: Windows 11, i7-14700K @ 3.40GHz, 64GB RAM, NVIDIA GeForce RTX 4080 SUPER

# Performance Analysis Report

## Methodology
- **Runtime environment:** CUDA implementation compiled in Release mode, VSync disabled.  
- **Measurement:** Framerates (FPS) recorded after ~20 s runtime. Values are estimates due to natural runtime variation.  
- **Modes tested:**
  - **Naive:** O(N²) neighbor checks.  
  - **Scattered uniform grid:** O(N log N + N·k) using Thrust sort + indirect indexing.  
  - **Coherent uniform grid:** O(N log N + N·k), but with reordered positions/velocities for better memory locality.  
- **Parameters:**
  - Default block size = 128 (unless otherwise specified).  
  - Visualized and non-visualized runs were both measured.  
  - Block size sweeps (32–1024) performed at N=5000 without visualization.  

---

## Results

### Framerate vs. Number of Boids

**With visualization (block size = 128):**

| # of boids     | Naive | Scattered uniform grid | Coherent uniform grid |
|----------------|-------|-------------------------|------------------------|
| 2000           | ~1550 | ~1700                  | ~1800                 |
| 5000 (default) | ~900  | ~1600                  | ~1700                 |
| 10000          | ~600  | ~1500                  | ~1650                 |
| 50000          | ~150  | ~850                   | ~1000                 |

<img width="2400" height="1500" alt="fps_vs_boids_with_vis" src="https://github.com/user-attachments/assets/e7421c53-ec28-4bcb-b73b-42e0893bc3a3" />

**Without visualization (block size = 128):**

| # of boids     | Naive | Scattered uniform grid | Coherent uniform grid |
|----------------|-------|-------------------------|------------------------|
| 2000           | ~1750 | ~2050                  | ~2150                 |
| 5000 (default) | ~1100 | ~1800                  | ~1950                 |
| 10000          | ~700  | ~1700                  | ~1950                 |
| 50000          | ~160  | ~1150                  | ~1300                 |

<img width="2400" height="1500" alt="fps_vs_boids_no_vis" src="https://github.com/user-attachments/assets/1ede1b26-19c4-4110-bc1d-62caba957b82" />


---

### Framerate vs. Block Size (N=5000, without visualization)

| Block size     | Naive | Scattered uniform grid | Coherent uniform grid |
|----------------|-------|-------------------------|------------------------|
| 32             | ~1050 | ~1650                  | ~1850                 |
| 64             | ~1100 | ~1750                  | ~1900                 |
| 128 (default)  | ~1100 | ~1800                  | ~1950                 |
| 256            | ~1130 | ~1800                  | ~1900                 |
| 512            | ~1050 | ~1750                  | ~1850                 |
| 1024 (maximum) | ~900  | ~1750                  | ~1850                 |

<img width="2400" height="1500" alt="fps_vs_block_size" src="https://github.com/user-attachments/assets/b2bcf60c-7119-467d-8b68-8583833d9326" />


---

## Related Questions & Insights

### 1) Effect of number of boids on performance

**Naive (O(N²))**  
- **Trend:** FPS falls steeply as N grows.  
- **Data (no-viz):** 1100→700→160 FPS at 5k→10k→50k.  
- **Reason:** Each boid checks every other boid, so neighbor work explodes with N; memory traffic and ALU usage both balloon.

**Scattered uniform grid (O(N log N + N·k))**  
- **Trend:** Much flatter scaling; high FPS even at large N.  
- **Data (no-viz):** 1800→1700→1150 FPS at 5k→10k→50k (≈ linear-ish).  
- **Reason:** Sorting is N log N, then each boid only inspects nearby candidates (k ≪ N), drastically reducing distance checks.

**Coherent uniform grid (same asymptotics, better locality)**  
- **Trend:** Consistently faster than scattered; the gap widens with N.  
- **Data (no-viz):** 1950→1950→1300 FPS at 5k→10k→50k.  
- **Reason:** Reordering positions/velocities into cell-contiguous arrays gives coalesced loads and better cache hit rates during neighbor passes.  

*At 5k boids (no-viz), coherent is ~8% faster than scattered; at 50k, ~13% faster. Versus naive, grid methods are 6–8× faster at 50k.*

---

### 2) Effect of block size / block count on performance

**Block size sweep (N=5000, no-viz):**  
- **Naive:** 1050–1130 FPS peaks around 256, drops at 1024 (900).  
- **Scattered:** ~1650→1800 at 128–256, slight dip outside that range.  
- **Coherent:** ~1850→1950 at 128, slight dip at ≥256.  

**Reason for shape (shallow U):**  
- **Too small (32–64):** insufficient resident warps per SM → weaker latency hiding.  
- **Sweet spot (128–256):** good occupancy, enough warps to hide global-memory latency in neighbor kernels.  
- **Too large (512–1024):** register/shared-mem pressure reduces active blocks/SM; fewer scheduling opportunities; tail block under-fills.  

*Best performance appears with 128–256 threads per block.*

---

### 3) Coherent grid improvements

- **Observation:** In every tested N, coherent > scattered (e.g., 1950 vs 1800 FPS at 5k; 1300 vs 1150 at 50k, no-viz).  
- **Expectation:** Improvement was anticipated, since coherent layout removes an indirection and makes neighbor reads contiguous, improving coalescing and cache locality.  
- **Result:** The benefit grows with N, as the neighbor pass dominates more of the frame time.  

---

### 4) Cell width and neighboring cells

Although 27 cells are checked when cell width = R, the important factor is **candidate boids**, not the number of cell headers accessed.  

Back-of-envelope for uniform density ρ:  
- `W = R, 27 cells` → candidates ∝ 27 · ρ · R³  
- `W = 2R, 8 cells` → candidates ∝ 64 · ρ · R³  

Thus, 2R+8 cells produces ~2.37× more candidates than R+27 cells. Distance checks dominate the cost, so R+27 is often faster, especially at moderate or high densities. Coherent layout benefits both, but the faster configuration is the one that minimizes candidate neighbors.  

**Edge cases:** In sparse scenes, the difference is small. If the implementation checks 27 cells for both widths, then R is strictly better due to lower per-cell occupancy.  
