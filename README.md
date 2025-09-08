![Untitled video - Made with Clipchamp (4)](https://github.com/user-attachments/assets/ef38b626-2857-4cae-83aa-4d56ad9089a1)

# Boids CUDA Implementation

*  Eli Asimow
* [LinkedIn](https://www.linkedin.com/in/eli-asimow/), [personal website](https://easimow.com)
* Tested on: Windows 11, AMD Ryzen 7 7435HS @ 2.08GHz 16GB, Nvidia GeForce RTX 4060 GPU

## **Overview**

This project is a complete implementation of the boids algorithm utilizing Cuda for performance optimization. Three methods for that Cuda utilization, Naive, Scattered and Coherent Grid, are included. They operate at different performances levels in different contexts, but with Coherent Grid the simulation can manage well over 1,000,000 boids in a real time environment (1,000 FPS). Additional rules are included for user enjoyement, including a revolving planetary effect and bouncing walls. 

## **Performance**
 <img width="600" height="371" alt="Naive, Scattered and Coherent Performance (FPS) (1)" src="https://github.com/user-attachments/assets/1ebd2ec4-0cda-4eac-9cde-f0e0c23863f6" />

As the boid count increases, performance deteriorates across all three implementations. However, the rate of change varies. The ultimate collection of optimizations of the coherent grid implementation slowly show their worth as the boid count grows. By 50,000, coherent was the clear best performer. By 1,000,000, coherent was the only viable real time algorithm left.

It’s important to note that both scattered and naive had moments of superiority as well. After metrics analysis, we can conclude that the new sort algorithms and memory overhead are the primary factors here. At exceedingly small boid counts, running additional kernels to sort our boid indices has a greater performance cost than the performance saved by scattered and coherent. So, at 1,000 boids, naive can actually be the best option. Likewise, at 10,000, although the grid cell sort of coherent and scattered make them superior to naive, coherent’s memory adjacency is still relatively less useful than the sort algorithms and additional buffer that scattered can skip. 

 <img width="600" height="371" alt="Naive, Scattered and Coherent Performance (FPS)" src="https://github.com/user-attachments/assets/d67a684d-45a2-4e83-9073-c3951e13e41a" />

I was surprised to see negligible effects on performance when modifying block size. Perhaps the block size ultimately doesn’t matter outside of a hard ceiling and floor. 
More expected is the difference between coherent and scattered grid performance. It makes sense that, as the number of boids in neighboring cells increases, the importance of memory adjacency for the speed of accessing them increases accordingly.

Lastly, my favorite optimization was realizing that 27 neighboring grid cells actually outperforms 8 neighbor cells! Grid cell size causes this counterintuitive result. In order to have only 8 cells checked, we must have a grid cell length of 2 * the maximum distance threshold. This results in a total neighbor check side length of 2 * 2d = 4d. With 27 neighboring cells, however, we can have each grid cell sized at the smaller 1 * maximum distance threshold. This results in a side length of just 3 * 1d = 3d, smaller than in the 8 cells implementation. Because 27’s radius check is ultimately smaller, that means more boids are filtered out of this step and the kernel can compute the velocity step relatively quicker. 

## **Bonus Work!**

![Untitled video - Made with Clipchamp](https://github.com/user-attachments/assets/2a2b7050-d5ea-4e33-bce9-63458173de8a)

I really fell in love with adding new rules to the system and experimented quite a bit over the week. The above gif is my favorite, and was accomplished with three changes:

1) Walls which push the particles away from the boundary
2) A universal gravity effect that pulls the particles down
3) A sinusoidal time gravity affected, modulated by particle distance, drawing particles towards the grid center and then dropping them down.

I'll leave you with some other results I found while messing around.

![Untitled video - Made with Clipchamp (7)](https://github.com/user-attachments/assets/866125de-dc03-43d8-9590-d182f2cb3b22)
![Untitled video - Made with Clipchamp (6)](https://github.com/user-attachments/assets/1cc25ca3-bd39-40ab-98da-bcb465445baf)
![Untitled video - Made with Clipchamp (5)](https://github.com/user-attachments/assets/d622082a-2a45-41f9-8f20-8c1f4d69b21b)
