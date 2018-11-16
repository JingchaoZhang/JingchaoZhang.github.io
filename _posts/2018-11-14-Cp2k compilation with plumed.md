---
layout: single
author_profile: false
---


Building Your Own FFTW3 Interface Wrapper Library
```bash
source /util/opt/lmod/lmod/init/profile
ml load compiler/intel/16
cd /util/comp/intel/2016.3/mkl/interfaces/fftw3xf
make libintel64 compiler=intel
cd /util/comp/intel/2016.3/mkl/interfaces/fftw3xc
make libintel64 compiler=intel
cp /util/comp/intel/2016.3/mkl/interfaces/fftw3xc/libfftw3xc_intel.a /util/comp/intel/2016.3/mkl/lib/intel64/
cp /util/comp/intel/2016.3/mkl/interfaces/fftw3xf/libfftw3xf_intel.a /util/comp/intel/2016.3/mkl/lib/intel64/
```
https://software.intel.com/en-us/mkl-developer-reference-c-building-your-own-fftw3-interface-wrapper-library

OpenMPI configurations
```bash
Open MPI configuration:
-----------------------
Version: 3.1.3
Build MPI C bindings: yes
Build MPI C++ bindings (deprecated): no
Build MPI Fortran bindings: mpif.h, use mpi, use mpi_f08
MPI Build Java bindings (experimental): no
Build Open SHMEM support: yes
Debug build: no
Platform file: (none)

Miscellaneous
-----------------------
CUDA support: no
PMIx support: internal

Transports
-----------------------
Cisco usNIC: no
Cray uGNI (Gemini/Aries): no
Intel Omnipath (PSM2): yes
Intel SCIF: no
Intel TrueScale (PSM): yes
Mellanox MXM: no
Open UCX: no
OpenFabrics Libfabric: no
OpenFabrics Verbs: yes
Portals4: no
Shared memory/copy in+copy out: yes
Shared memory/Linux CMA: yes
Shared memory/Linux KNEM: yes
Shared memory/XPMEM: no
TCP: yes

Resource Managers
-----------------------
Cray Alps: no
Grid Engine: no
LSF: no
Moab: no
Slurm: yes
ssh/rsh: yes
Torque: no

OMPIO File Systems
-----------------------
Generic Unix FS: yes
Lustre: yes
PVFS2/OrangeFS: no
```

dbcsr_tensor_test.F90 error
```bash
/tmp/ifortFeHxAb.i90: catastrophic error: **Internal compiler error: segmentation violation signal raised** Please report this error along with the circumstances in which it occurred in a Software Problem Report.  Note: File and line given may not be explicit cause of this error.
compilation aborted for dbcsr_tensor_test.F90 (code 1)
make[3]: *** [dbcsr_tensor_test.o] Error 1
make[2]: *** [all] Error 2
make[1]: *** [popt] Error 2
make: *** [all] Error 2
```
Fix
```
As a quick and dirty solution, you could remove the two files
dbcsr_tensor/dbcsr_tensor_test.F
dbcsr_tensor/dbcsr_tensor_unittest.F
and CP2K will probably compile fine even with the 2015 version of the intel compiler, since these 2 files are compiled into a separate executable and not part of CP2K executable.
```
