---
layout: single
author_profile: false
---

```
module load compiler/gcc openmpi R
export R_LIBS=$HOME/R/x86_64-pc-linux-gnu-library/3.3
mkdir -p $HOME/R/x86_64-pc-linux-gnu-library/3.3
wget https://cran.r-project.org/src/contrib/Rmpi_0.6-6.tar.gz
R CMD INSTALL Rmpi_0.6-6.tar.gz --configure-args="--with-Rmpi-type=OPENMPI"
```

With QLogic infiniband, add  
```
export OMPI_MCA_mtl=^psm
```  
before  
```
R CMD INSTALL Rmpi_0.6-6.tar.gz --configure-args="--with-Rmpi-type=OPENMPI"
```
