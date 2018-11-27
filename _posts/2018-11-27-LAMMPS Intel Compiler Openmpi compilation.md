---
layout: single
author_profile: false
---

```bash
1. wget https://github.com/lammps/lammps/archive/stable_22Aug2018.tar.gz
1. tar zxvf stable_22Aug2018.tar.gz
1. cd lammps-stable_22Aug2018/src
1. for i in BODY CLASS2 COLLOID DIPOLE GRANULAR KSPACE MANYBODY MISC MOLECULE MPIIO RIGID USER-COLVARS USER-MISC USER-PHONON USER-REAXC; do make yes-$i; done
1. ml load compiler/intel/16 openmpi/3.1
1. cd ../lib/colvars/
1. make -j 8 -f Makefile.mpi
1. cd ../../src
1. make -j 8 icc_openmpi
```
