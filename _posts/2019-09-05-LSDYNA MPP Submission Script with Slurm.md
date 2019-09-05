---
layout: single
author_profile: false
---

```bash
========================================================
#!/bin/sh
# Change the job name, if you want to
#SBATCH --job-name=pendulum  

# Change the number of cpu's, if you want to
#SBATCH --ntasks=4

# Change the cpu requested time (walltime), if you need to
#SBATCH --time=00:10:00


# ##################################################################
#  system memory - this is a lot and probably doesn't need changing
#    What you might have to change is the memory command on the
#    LS-Dyna "mpirun" command near the bottom.
# ##################################################################

#SBATCH --mem-per-cpu=2048

# #########################
#    Do not change these
# #########################

#SBATCH --error=std-err.%J
#SBATCH --output=std-out.%J

# #SBATCH --qos=short

# LOAD MPI AND DYNA ENVIRONMENT
module load lsdyna/10.1

# GLOBAL/LOCAL for pfile (MPP DYNA) - automatic
mkdir d.results
mkdir d.dump-files
echo "directory { global ./d.results local ./d.dump-files }" > pfile

# ##############
#    LS-Dyna
# ##############
# keep track of LS-Dyna runtime
/bin/date > time-start

# Change dyna deck file name
# Change memory requirement, if you need to

mpirun mpp971_s i=pendulum.k p=pfile memory=32m 

/bin/date > time-end
```

The submit script uses single precision version of MPP DYNA (mpp971_s), double precision is also available (mpp971_d).
