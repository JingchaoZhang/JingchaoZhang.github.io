---
layout: single
author_profile: false
---

```
module load compiler/gcc openmpi R
export R_LIBS=$HOME/R/x86_64-pc-linux-gnu-library/3.3
mkdir -p $HOME/R/x86_64-pc-linux-gnu-library/3.3
wget https://cran.r-project.org/src/contrib/Rmpi_0.6-6.tar.gz
R CMD INSTALL Rmpi_0.6-6.tar.gz --configure-args="--with-Rmpi-include=/util/opt/openmpi/1.10/gcc/4.9/include --with-Rmpi-libpath=/util/opt/openmpi/1.10/gcc/4.9/lib --with-Rmpi-type=OPENMPI"
```

With QLogic infiniband, add  
```
export OMPI_MCA_mtl=^psm
```  
before  
```
R CMD INSTALL Rmpi_0.6-6.tar.gz --configure-args="--with-Rmpi-include=/util/opt/openmpi/1.10/gcc/4.9/include --with-Rmpi-libpath=/util/opt/openmpi/1.10/gcc/4.9/lib --with-Rmpi-type=OPENMPI"
```

Example

- Rmpi-test.R  
```
library("datasets")
library("snow")
library("Rmpi")

mydata = iris[,-5] #dataset used to test
self.num = c(3,5,7,9,10) #centers tested
nboot.d=5

parallel.function <- function(i,data,centers) {
            kmeans(data, centers, nstart=i )
        }
cl <- makeCluster( mpi.universe.size()-1, type="MPI" )
clusterExport(cl, c('data'))

for(round.j in c(1:length(self.num))){
  para.result <- parLapply( cl, rep(1,nboot.d), fun=parallel.function, data=mydata, centers=self.num[round.j])
  print(para.result)
}

stopCluster(cl)
mpi.exit()
```  

- submit.slurm  
```
#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=1024
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module load compiler/gcc/4.9 openmpi/1.10 R/3.3
mpirun -n 1 R CMD BATCH Rmpi-test.R
```  

- $sbatch submit.slurm
