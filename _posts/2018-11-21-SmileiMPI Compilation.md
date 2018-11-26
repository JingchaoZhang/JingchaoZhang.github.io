---
layout: single
author_profile: false
---

1. Compile OpenMPI with MPI_THREAD_MULTIPLE support. ./configure \-\-enable-mpi-thread-multiple
1. ml load compiler/gcc/4.9 openmpi/1.10 phdf5/1.8 python/3.6
1. export LIBRARY_PATH=$LIBRARY_PATH:/util/opt/anaconda/deployed-conda-envs/packages/python/envs/python-3.6/lib  
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/util/opt/anaconda/deployed-conda-envs/packages/python/envs/python-3.6/lib
1. wget https://github.com/SmileiPIC/Smilei/archive/v4.0.tar.gz
1. tar zxvf v4.0.tar.gz
1. cd Smilei-4.0/
1. make
