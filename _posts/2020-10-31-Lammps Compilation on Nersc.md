---
layout: single
author_profile: false
---

**NOTE: NERSC provides compiled LAMMPS binaries as modules. This tutorial is for users with modified source code.**

**NERSC**: National Energy Research Scientific Computing Center
**Mission**: Accelerate scientific discovery at the DOE Office of Science through High-Performance Computing and Extreme Data Analysis

**Node type**:
- Haswell nodes: ● For throughput ● Queues allow single-core jobs ● Longer walltime limits for smaller jobs ● Long queues
- KNL nodes: ● For performance ● Codes should exploit many-core architecture ● Large jobs encouraged; discount for jobs using ≥1024 nodes ● 4x larger than Haswell - - partition ● Shorter queues ● Flex queue increases throughput & offers substantial discount

**Storage options**:
- Home: ● Permanent, relatively small storage ● Mounted on all platforms ● NOT tuned to perform well for parallel jobs ● Quota cannot be changed ● Snapshot backups (7-day history) ● Perfect for storing data such as source code, shell scripts
- Community File System (CFS): ● Permanent, larger storage ● Mounted on all platforms ● Medium performance for parallel jobs ● Quota can be changed ● Snapshot backups (7-day history) ● Perfect for sharing data within research group
- Scratch: ● Large, temporary storage ● Optimized for read/write operations, NOT storage ● Not backed up ● Purge policy (12 weeks) ● Perfect for staging data and performing computations
- Burst Buffer: ● Temporary per-job storage ● High-performance SSD file system ● Available on Cori only ● Perfect for getting good performance in I/O-constrained codes


#The following steps are for "Haswell" nodes, which is also the type of the login node
```bash
cd ~/lammps-stable_29Oct2020/src
module load intel/19.0.3.199    
module load openmpi/4.0.2 #impi/2020
#If not first time compilation, run make clean-all first
make clean-all
#The for command only need to be run once to enable optional packages
for i in BODY CLASS2 COLLOID DIPOLE GRANULAR KSPACE MANYBODY MISC MOLECULE MPIIO RIGID USER-MISC USER-PHONON USER-REAXC; do make yes-$i; done
make -j 12 icc_openmpi
```

#TODO: KNL COMPILATION NOT WORKING. 
#The following steps are for "KNL" nodes
```bash
#If compiling on a KNL node is needed, do the following “sacct” to get onto a compute node
#salloc –N 1 –q interactive –C knl –t 4:00:00
module load impi/2020
module swap craype-haswell craype-mic-knl
make -j 12 knl
```
#TODO: KNL COMPILATION NOT WORKING

#Place the binary in PATH. Only need to do this once
```bash
mkdir bin
vim ~/.bash_profile
  #Put this line in ~/.bash_profile
  export PATH=$HOME/bin:$PATH
source ~/.bash_profile
cd #HOME/bin
ln -s ~/lammps-stable_29Oct2020/src/ ./
```

To submit a job, switch to $SCATCH first
```bash
cd $SCRATCH
```
**Sample SLURM File**
```bash
#!/bin/bash
#SBATCH -J test_lammps
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 01:00:00
#SBATCH -e job.%j.err
#SBATCH -o job.%j.out

module load intel/19.0.3.199 openmpi/4.0.2

srun -n 64 -c 2 --cpu-bind=cores lmp_icc_openmpi < in.snr
```
Submit the job using sbatch
```bash
sbatch SLURMFILE
```
