---
layout: single
author_profile: false
---

Some guidlines on selecting the number of MPI processes per GPU
- When using the GPU package, you cannot assign more than one GPU to a single MPI task
- Multiple MPI tasks can share the same GPU, and in many cases it will be more efficient to run this way
- You should experiment with how many MPI tasks per GPU to use to give the best performance for your problem and machine. This is also a function of the problem size and the pair style being using. 
- You should also experiment with the precision setting for the GPU library to see if single or mixed precision will give accurate results, since they will typically be faster.
- MPI parallelism typically outperforms OpenMP parallelism, but in some cases using fewer MPI tasks and multiple OpenMP threads with the GPU package can give better performance. 
- If the number of particles per MPI task is small (e.g. 100s of particles), it can be more efficient to run with fewer MPI tasks per GPU, even if you do not use all the cores on the compute node.
