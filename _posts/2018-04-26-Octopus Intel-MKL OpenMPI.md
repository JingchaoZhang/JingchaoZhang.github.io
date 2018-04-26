---
layout: single
author_profile: false
---

[Octopus](http://octopus-code.org/wiki/Octopus_7)
FFTW3
OpenMPI
Intel-MKL
GSL

```bash
$ module load compiler/intel/13 intel-mkl/13 openmpi/1.10 fftw3/3.3 GSL/1.6
$ make clean && make distclean
$ export FC=mpif90
$ export CC=mpicc
$ export FCFLAGS="-O3"
$ export CFLAGS="-O3"
$ ./configure \
--with-gsl-prefix=/util/opt/GSL/1.16/intel/13 \
--with-libxc-prefix=/util/opt/libxc/2.2/intel/13 \
--with-libxc-include=/util/opt/libxc/2.2/intel/13/include \
--disable-zdotc-test --enable-mpi \
--with-fftw-prefix=/util/opt/fftw3/3.3/intel/13 \
--prefix=/home/zeng/xuwenwu/octopus-install
$ make -j 12 install 
```
