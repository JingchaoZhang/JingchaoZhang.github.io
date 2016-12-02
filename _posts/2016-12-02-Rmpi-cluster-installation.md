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
- $cat Rmpi-test.Rout
```
R version 3.3.1 (2016-06-21) -- "Bug in Your Hair"
Copyright (C) 2016 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

[Previously saved workspace restored]

> library("datasets")
> library("snow")
> library("Rmpi")
>
> mydata = iris[,-5] #dataset used to test
> self.num = c(3,5,7,9,10) #centers tested
> nboot.d=5
>
> parallel.function <- function(i,data,centers) {
+             kmeans(data, centers, nstart=i )
+         }
> cl <- makeCluster( mpi.universe.size()-1, type="MPI" )
        7 slaves are spawned successfully. 0 failed.
> clusterExport(cl, c('data'))
>
> for(round.j in c(1:length(self.num))){
+   para.result <- parLapply( cl, rep(1,nboot.d), fun=parallel.function, data=mydata, centers=self.num[round.j])
+   print(para.result)
+ }
[[1]]
K-means clustering with 3 clusters of sizes 62, 38, 50

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.901613    2.748387     4.393548    1.433871
2     6.850000    3.073684     5.742105    2.071053
3     5.006000    3.428000     1.462000    0.246000

Clustering vector:
  [1] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 [38] 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [75] 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2
[112] 2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2
[149] 2 1

Within cluster sum of squares by cluster:
[1] 39.82097 23.87947 15.15100
 (between_SS / total_SS =  88.4 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[2]]
K-means clustering with 3 clusters of sizes 62, 50, 38

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.901613    2.748387     4.393548    1.433871
2     5.006000    3.428000     1.462000    0.246000
3     6.850000    3.073684     5.742105    2.071053

Clustering vector:
  [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 [38] 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [75] 1 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 1 3 3 3 3 1 3 3 3 3
[112] 3 3 1 1 3 3 3 3 1 3 1 3 1 3 3 1 1 3 3 3 3 3 1 3 3 3 3 1 3 3 3 1 3 3 3 1 3
[149] 3 1

Within cluster sum of squares by cluster:
[1] 39.82097 15.15100 23.87947
 (between_SS / total_SS =  88.4 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[3]]
K-means clustering with 3 clusters of sizes 62, 38, 50

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.901613    2.748387     4.393548    1.433871
2     6.850000    3.073684     5.742105    2.071053
3     5.006000    3.428000     1.462000    0.246000

Clustering vector:
  [1] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 [38] 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [75] 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2
[112] 2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2
[149] 2 1

Within cluster sum of squares by cluster:
[1] 39.82097 23.87947 15.15100
 (between_SS / total_SS =  88.4 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[4]]
K-means clustering with 3 clusters of sizes 50, 38, 62

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.006000    3.428000     1.462000    0.246000
2     6.850000    3.073684     5.742105    2.071053
3     5.901613    2.748387     4.393548    1.433871

Clustering vector:
  [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [38] 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 [75] 3 3 3 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 3 2 2 2 2 3 2 2 2 2
[112] 2 2 3 3 2 2 2 2 3 2 3 2 3 2 2 3 3 2 2 2 2 2 3 2 2 2 2 3 2 2 2 3 2 2 2 3 2
[149] 2 3

Within cluster sum of squares by cluster:
[1] 15.15100 23.87947 39.82097
 (between_SS / total_SS =  88.4 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[5]]
K-means clustering with 3 clusters of sizes 62, 50, 38

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.901613    2.748387     4.393548    1.433871
2     5.006000    3.428000     1.462000    0.246000
3     6.850000    3.073684     5.742105    2.071053

Clustering vector:
  [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 [38] 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [75] 1 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 1 3 3 3 3 1 3 3 3 3
[112] 3 3 1 1 3 3 3 3 1 3 1 3 1 3 3 1 1 3 3 3 3 3 1 3 3 3 3 1 3 3 3 1 3 3 3 1 3
[149] 3 1

Within cluster sum of squares by cluster:
[1] 39.82097 15.15100 23.87947
 (between_SS / total_SS =  88.4 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[1]]
K-means clustering with 5 clusters of sizes 24, 30, 22, 50, 24

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.483333    2.591667     3.904167    1.200000
2     6.213333    2.886667     5.176667    1.926667
3     7.122727    3.113636     6.031818    2.131818
4     5.006000    3.428000     1.462000    0.246000
5     6.312500    2.912500     4.537500    1.420833

Clustering vector:
  [1] 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
 [38] 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 1 5 5 5 1 5 1 1 5 1 5 1 5 5 1 5 1 2 5 5 5
 [75] 5 5 5 2 5 1 1 1 1 2 1 5 5 5 1 1 1 5 1 1 1 1 1 5 1 1 3 2 3 2 3 3 1 3 3 3 2
[112] 2 3 2 2 2 2 3 3 2 3 2 3 2 3 3 2 2 2 3 3 3 2 2 2 3 2 2 2 3 3 2 2 3 3 2 2 2
[149] 2 2

Within cluster sum of squares by cluster:
[1]  7.941250  9.081667 11.540000 15.151000  6.148333
 (between_SS / total_SS =  92.7 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[2]]
K-means clustering with 5 clusters of sizes 16, 26, 36, 50, 22

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.412500    2.468750     3.743750    1.168750
2     6.019231    2.869231     4.384615    1.357692
3     6.288889    2.905556     5.111111    1.852778
4     5.006000    3.428000     1.462000    0.246000
5     7.122727    3.113636     6.031818    2.131818

Clustering vector:
  [1] 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
 [38] 4 4 4 4 4 4 4 4 4 4 4 4 4 3 2 3 1 2 2 3 1 2 1 1 2 1 2 1 2 2 2 2 1 3 2 3 2
 [75] 2 2 3 3 2 1 1 1 1 3 2 2 3 2 2 1 2 2 1 1 2 2 2 2 1 2 5 3 5 3 5 5 1 5 5 5 3
[112] 3 5 3 3 3 3 5 5 3 5 3 5 3 5 5 3 3 3 5 5 5 3 3 3 5 3 3 3 5 5 3 3 5 5 3 3 3
[149] 3 3

Within cluster sum of squares by cluster:
[1]  5.105625  6.393077 12.899722 15.151000 11.540000
 (between_SS / total_SS =  92.5 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[3]]
K-means clustering with 5 clusters of sizes 22, 28, 27, 45, 28

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     4.704545    3.122727     1.413636   0.2000000
2     5.242857    3.667857     1.500000   0.2821429
3     7.014815    3.096296     5.918519   2.1555556
4     6.264444    2.884444     4.886667   1.6666667
5     5.532143    2.635714     3.960714   1.2285714

Clustering vector:
  [1] 2 1 1 1 2 2 1 2 1 1 2 1 1 1 2 2 2 2 2 2 2 2 1 2 1 1 2 2 2 1 1 2 2 2 1 1 2
 [38] 2 1 2 2 1 1 2 2 1 2 1 2 1 4 4 4 5 4 5 4 5 4 5 5 5 5 4 5 4 5 5 4 5 4 5 4 4
 [75] 4 4 4 4 4 5 5 5 5 4 5 4 4 4 5 5 5 4 5 5 5 5 5 4 5 5 3 4 3 4 3 3 5 3 3 3 4
[112] 4 3 4 4 4 4 3 3 4 3 4 3 4 3 3 4 4 3 3 3 3 3 4 4 3 3 4 4 3 3 3 4 3 3 3 4 4
[149] 4 4

Within cluster sum of squares by cluster:
[1]  3.114091  4.630714 15.351111 17.014222  9.749286
 (between_SS / total_SS =  92.7 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[4]]
K-means clustering with 5 clusters of sizes 7, 50, 39, 32, 22

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.242857    2.371429     3.442857    1.028571
2     5.006000    3.428000     1.462000    0.246000
3     6.253846    2.853846     4.828205    1.633333
4     6.912500    3.100000     5.846875    2.131250
5     5.654545    2.731818     4.140909    1.295455

Clustering vector:
  [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 [38] 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 5 3 5 3 1 3 5 1 5 5 3 5 3 5 5 3 5 3 5 3 3
 [75] 3 3 3 3 3 1 1 1 5 3 5 3 3 3 5 5 5 3 5 1 5 5 5 5 1 5 4 3 4 4 4 4 5 4 4 4 3
[112] 3 4 3 3 4 4 4 4 3 4 3 4 3 4 4 3 3 4 4 4 4 4 3 3 4 4 4 3 4 4 4 3 4 4 4 3 3
[149] 4 3

Within cluster sum of squares by cluster:
[1]  1.262857 15.151000 13.239487 18.703437  4.545000
 (between_SS / total_SS =  92.2 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[5]]
K-means clustering with 5 clusters of sizes 32, 22, 28, 28, 40

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     6.912500    3.100000     5.846875   2.1312500
2     4.704545    3.122727     1.413636   0.2000000
3     5.532143    2.635714     3.960714   1.2285714
4     5.242857    3.667857     1.500000   0.2821429
5     6.252500    2.855000     4.815000   1.6250000

Clustering vector:
  [1] 4 2 2 2 4 4 2 4 2 2 4 2 2 2 4 4 4 4 4 4 4 4 2 4 2 2 4 4 4 2 2 4 4 4 2 2 4
 [38] 4 2 4 4 2 2 4 4 2 4 2 4 2 5 5 5 3 5 3 5 3 5 3 3 3 3 5 3 5 3 3 5 3 5 3 5 5
 [75] 5 5 5 5 5 3 3 3 3 5 3 5 5 5 3 3 3 5 3 3 3 3 3 5 3 3 1 5 1 1 1 1 3 1 1 1 5
[112] 5 1 5 5 1 1 1 1 5 1 5 1 5 1 1 5 5 1 1 1 1 1 5 5 1 1 1 5 1 1 1 5 1 1 1 5 5
[149] 1 5

Within cluster sum of squares by cluster:
[1] 18.703437  3.114091  9.749286  4.630714 13.624750
 (between_SS / total_SS =  92.7 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[1]]
K-means clustering with 7 clusters of sizes 40, 7, 32, 5, 28, 21, 17

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     6.252500    2.855000     4.815000   1.6250000
2     5.528571    4.042857     1.471429   0.2857143
3     6.912500    3.100000     5.846875   2.1312500
4     4.400000    2.880000     1.280000   0.2000000
5     5.532143    2.635714     3.960714   1.2285714
6     5.147619    3.542857     1.509524   0.2809524
7     4.794118    3.194118     1.452941   0.2000000

Clustering vector:
  [1] 6 7 7 7 6 2 7 6 4 7 6 7 7 4 2 2 2 6 2 6 6 6 7 6 7 7 6 6 6 7 7 6 2 2 7 7 6
 [38] 6 4 6 6 4 4 6 6 7 6 7 6 7 1 1 1 5 1 5 1 5 1 5 5 5 5 1 5 1 5 5 1 5 1 5 1 1
 [75] 1 1 1 1 1 5 5 5 5 1 5 1 1 1 5 5 5 1 5 5 5 5 5 1 5 5 3 1 3 3 3 3 5 3 3 3 1
[112] 1 3 1 1 3 3 3 3 1 3 1 3 1 3 3 1 1 3 3 3 3 3 1 1 3 3 3 1 3 3 3 1 3 3 3 1 1
[149] 3 1

Within cluster sum of squares by cluster:
[1] 13.6247500  0.8342857 18.7034375  0.5560000  9.7492857  1.7142857  1.4611765
 (between_SS / total_SS =  93.2 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[2]]
K-means clustering with 7 clusters of sizes 21, 50, 19, 22, 19, 12, 7

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.628571    2.723810     4.133333    1.295238
2     5.006000    3.428000     1.462000    0.246000
3     6.036842    2.705263     5.000000    1.778947
4     6.568182    3.086364     5.536364    2.163636
5     6.442105    2.978947     4.594737    1.431579
6     7.475000    3.125000     6.300000    2.050000
7     5.242857    2.371429     3.442857    1.028571

Clustering vector:
  [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 [38] 2 2 2 2 2 2 2 2 2 2 2 2 2 5 5 5 1 5 1 5 7 5 1 7 1 1 5 1 5 1 1 3 1 3 1 3 5
 [75] 5 5 5 5 5 7 7 7 1 3 1 5 5 5 1 1 1 5 1 7 1 1 1 5 7 1 4 3 6 4 4 6 1 6 4 6 4
[112] 3 4 3 3 4 4 6 6 3 4 3 6 3 4 6 3 3 4 6 6 6 4 3 3 6 4 4 3 4 4 4 3 4 4 4 3 4
[149] 4 3

Within cluster sum of squares by cluster:
[1]  4.177143 15.151000  4.125263  4.315455  3.708421  4.655000  1.262857
 (between_SS / total_SS =  94.5 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[3]]
K-means clustering with 7 clusters of sizes 24, 19, 12, 27, 23, 8, 37

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     6.529167    3.058333     5.508333    2.162500
2     4.678947    3.084211     1.378947    0.200000
3     7.475000    3.125000     6.300000    2.050000
4     5.529630    2.622222     3.940741    1.218519
5     5.100000    3.513043     1.526087    0.273913
6     5.512500    4.000000     1.475000    0.275000
7     6.229730    2.851351     4.767568    1.572973

Clustering vector:
  [1] 5 2 2 2 5 6 2 5 2 2 6 5 2 2 6 6 6 5 6 5 5 5 2 5 5 2 5 5 5 2 2 5 6 6 2 2 5
 [38] 5 2 5 5 2 2 5 5 2 5 2 5 5 7 7 7 4 7 4 7 4 7 4 4 4 4 7 4 7 7 4 7 4 7 4 7 7
 [75] 7 7 7 7 7 4 4 4 4 7 4 7 7 7 4 4 4 7 4 4 4 4 4 7 4 4 1 7 3 1 1 3 4 3 1 3 1
[112] 1 1 7 1 1 1 3 3 7 1 7 3 7 1 3 7 7 1 3 3 3 1 7 7 3 1 1 7 1 1 1 7 1 1 1 7 1
[149] 1 7

Within cluster sum of squares by cluster:
[1]  5.462500  2.488421  4.655000  9.228889  2.094783  0.958750 11.963784
 (between_SS / total_SS =  94.6 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[4]]
K-means clustering with 7 clusters of sizes 8, 62, 5, 16, 16, 38, 5

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.212500    3.812500     1.587500    0.275000
2     5.901613    2.748387     4.393548    1.433871
3     5.620000    4.060000     1.420000    0.300000
4     5.125000    3.450000     1.475000    0.275000
5     4.781250    3.187500     1.456250    0.200000
6     6.850000    3.073684     5.742105    2.071053
7     4.400000    2.880000     1.280000    0.200000

Clustering vector:
  [1] 4 5 5 5 4 1 5 4 7 5 1 5 5 7 3 3 3 4 3 1 4 1 5 4 5 5 4 4 4 5 5 4 1 3 5 5 4
 [38] 4 7 4 4 7 7 4 1 5 1 5 1 4 2 2 6 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 [75] 2 2 2 6 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 6 2 6 6 6 6 2 6 6 6 6
[112] 6 6 2 2 6 6 6 6 2 6 2 6 2 6 6 2 2 6 6 6 6 6 2 6 6 6 6 2 6 6 6 2 6 6 6 2 6
[149] 6 2

Within cluster sum of squares by cluster:
[1]  0.50125 39.82097  0.52800  1.07000  1.40125 23.87947  0.55600
 (between_SS / total_SS =  90.1 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[5]]
K-means clustering with 7 clusters of sizes 23, 24, 19, 25, 39, 8, 12

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.100000    3.513043     1.526087    0.273913
2     6.529167    3.058333     5.508333    2.162500
3     4.678947    3.084211     1.378947    0.200000
4     5.508000    2.600000     3.908000    1.204000
5     6.207692    2.853846     4.746154    1.564103
6     5.512500    4.000000     1.475000    0.275000
7     7.475000    3.125000     6.300000    2.050000

Clustering vector:
  [1] 1 3 3 3 1 6 3 1 3 3 6 1 3 3 6 6 6 1 6 1 1 1 3 1 1 3 1 1 1 3 3 1 6 6 3 3 1
 [38] 1 3 1 1 3 3 1 1 3 1 3 1 1 5 5 5 4 5 5 5 4 5 4 4 5 4 5 4 5 5 4 5 4 5 4 5 5
 [75] 5 5 5 5 5 4 4 4 4 5 4 5 5 5 4 4 4 5 4 4 4 4 4 5 4 4 2 5 7 2 2 7 4 7 2 7 2
[112] 2 2 5 2 2 2 7 7 5 2 5 7 5 2 7 5 5 2 7 7 7 2 5 5 7 2 2 5 2 2 2 5 2 2 2 5 2
[149] 2 5

Within cluster sum of squares by cluster:
[1]  2.094783  5.462500  2.488421  8.366400 12.811282  0.958750  4.655000
 (between_SS / total_SS =  94.6 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[1]]
K-means clustering with 9 clusters of sizes 28, 16, 12, 22, 7, 19, 10, 17, 19

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.532143    2.635714     3.960714   1.2285714
2     4.668750    3.025000     1.412500   0.1937500
3     7.475000    3.125000     6.300000   2.0500000
4     6.568182    3.086364     5.536364   2.1636364
5     5.528571    4.042857     1.471429   0.2857143
6     6.442105    2.978947     4.594737   1.4315789
7     5.260000    3.630000     1.550000   0.2700000
8     4.958824    3.435294     1.452941   0.2647059
9     6.036842    2.705263     5.000000   1.7789474

Clustering vector:
  [1] 8 2 2 2 8 5 8 8 2 2 7 8 2 2 5 5 5 8 5 7 7 7 8 8 8 2 8 7 8 2 2 7 5 5 2 8 7
 [38] 8 2 8 8 2 2 8 7 2 7 2 7 8 6 6 6 1 6 1 6 1 6 1 1 1 1 6 1 6 1 1 9 1 9 1 9 6
 [75] 6 6 6 6 6 1 1 1 1 9 1 6 6 6 1 1 1 6 1 1 1 1 1 6 1 1 4 9 3 4 4 3 1 3 4 3 4
[112] 9 4 9 9 4 4 3 3 9 4 9 3 9 4 3 9 9 4 3 3 3 4 9 9 3 4 4 9 4 4 4 9 4 4 4 9 4
[149] 4 9

Within cluster sum of squares by cluster:
[1] 9.7492857 1.7312500 4.6550000 4.3154545 0.8342857 3.7084211 0.7710000
[8] 1.5611765 4.1252632
 (between_SS / total_SS =  95.4 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[2]]
K-means clustering with 9 clusters of sizes 21, 18, 7, 17, 16, 10, 10, 18, 33

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.628571    2.723810     4.133333   1.2952381
2     6.427778    2.977778     4.572222   1.4166667
3     5.242857    2.371429     3.442857   1.0285714
4     5.370588    3.800000     1.517647   0.2764706
5     6.543750    2.987500     5.381250   2.0312500
6     6.720000    3.240000     5.800000   2.3300000
7     7.540000    3.090000     6.360000   2.0000000
8     6.016667    2.705556     4.983333   1.7722222
9     4.818182    3.236364     1.433333   0.2303030

Clustering vector:
  [1] 9 9 9 9 9 4 9 9 9 9 4 9 9 9 4 4 4 9 4 4 4 4 9 9 9 9 9 4 9 9 9 4 4 4 9 9 4
 [38] 9 9 9 9 9 9 9 4 9 4 9 4 9 2 2 2 1 2 1 2 3 2 1 3 1 1 2 1 2 1 1 8 1 8 1 8 2
 [75] 2 2 2 5 2 3 3 3 1 8 1 2 2 2 1 1 1 2 1 3 1 1 1 2 3 1 6 8 6 5 6 7 1 7 5 6 5
[112] 5 5 8 8 5 5 7 7 8 6 8 7 8 6 7 8 8 5 7 7 7 5 8 8 7 6 5 8 5 6 5 8 6 6 5 8 5
[149] 5 8

Within cluster sum of squares by cluster:
[1] 4.177143 3.388333 1.262857 2.630588 2.795625 1.601000 3.677000 3.875556
[9] 5.428485
 (between_SS / total_SS =  95.8 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[3]]
K-means clustering with 9 clusters of sizes 6, 7, 19, 21, 22, 7, 21, 28, 19

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     7.150000    2.900000     5.983333   1.8333333
2     5.242857    2.371429     3.442857   1.0285714
3     6.036842    2.705263     5.000000   1.7789474
4     5.628571    2.723810     4.133333   1.2952381
5     4.704545    3.122727     1.413636   0.2000000
6     7.642857    3.228571     6.500000   2.2000000
7     6.561905    3.114286     5.523810   2.1809524
8     5.242857    3.667857     1.500000   0.2821429
9     6.442105    2.978947     4.594737   1.4315789

Clustering vector:
  [1] 8 5 5 5 8 8 5 8 5 5 8 5 5 5 8 8 8 8 8 8 8 8 5 8 5 5 8 8 8 5 5 8 8 8 5 5 8
 [38] 8 5 8 8 5 5 8 8 5 8 5 8 5 9 9 9 4 9 4 9 2 9 4 2 4 4 9 4 9 4 4 3 4 3 4 3 9
 [75] 9 9 9 9 9 2 2 2 4 3 4 9 9 9 4 4 4 9 4 2 4 4 4 9 2 4 7 3 1 7 7 6 4 1 1 6 7
[112] 3 7 3 3 7 7 6 6 3 7 3 6 3 7 1 3 3 7 1 1 6 7 3 3 6 7 7 3 7 7 7 3 7 7 7 3 7
[149] 7 3

Within cluster sum of squares by cluster:
[1] 0.8966667 1.2628571 4.1252632 4.1771429 3.1140909 2.5314286 3.7257143
[8] 4.6307143 3.7084211
 (between_SS / total_SS =  95.9 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[4]]
K-means clustering with 9 clusters of sizes 9, 21, 19, 22, 7, 12, 17, 28, 15

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     6.411111    2.822222     5.522222   1.8666667
2     5.628571    2.723810     4.133333   1.2952381
3     6.442105    2.978947     4.594737   1.4315789
4     4.704545    3.122727     1.413636   0.2000000
5     5.242857    2.371429     3.442857   1.0285714
6     7.475000    3.125000     6.300000   2.0500000
7     6.011765    2.711765     4.947059   1.7941176
8     5.242857    3.667857     1.500000   0.2821429
9     6.620000    3.186667     5.533333   2.2733333

Clustering vector:
  [1] 8 4 4 4 8 8 4 8 4 4 8 4 4 4 8 8 8 8 8 8 8 8 4 8 4 4 8 8 8 4 4 8 8 8 4 4 8
 [38] 8 4 8 8 4 4 8 8 4 8 4 8 4 3 3 3 2 3 2 3 5 3 2 5 2 2 3 2 3 2 2 7 2 7 2 7 3
 [75] 3 3 3 3 3 5 5 5 2 7 2 3 3 3 2 2 2 3 2 5 2 2 2 3 5 2 9 7 6 1 9 6 2 6 1 6 9
[112] 1 9 7 7 9 1 6 6 7 9 7 6 7 9 6 7 7 1 6 6 6 1 7 1 6 9 1 7 9 9 9 7 9 9 9 7 1
[149] 9 7

Within cluster sum of squares by cluster:
[1] 1.200000 4.177143 3.708421 3.114091 1.262857 4.655000 3.307059 4.630714
[9] 2.444000
 (between_SS / total_SS =  95.8 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[5]]
K-means clustering with 9 clusters of sizes 12, 21, 18, 22, 12, 9, 15, 28, 13

Cluster means:
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     6.633333    3.033333     4.633333   1.4583333
2     5.785714    2.828571     4.304762   1.3380952
3     6.016667    2.722222     4.933333   1.7722222
4     4.704545    3.122727     1.413636   0.2000000
5     7.475000    3.125000     6.300000   2.0500000
6     6.411111    2.822222     5.522222   1.8666667
7     6.620000    3.186667     5.533333   2.2733333
8     5.242857    3.667857     1.500000   0.2821429
9     5.392308    2.438462     3.653846   1.1230769

Clustering vector:
  [1] 8 4 4 4 8 8 4 8 4 4 8 4 4 4 8 8 8 8 8 8 8 8 4 8 4 4 8 8 8 4 4 8 8 8 4 4 8
 [38] 8 4 8 8 4 4 8 8 4 8 4 8 4 1 1 1 9 1 2 1 9 1 9 9 2 9 3 9 1 2 2 3 9 3 2 3 2
 [75] 1 1 1 1 2 9 9 9 2 3 2 2 1 2 2 9 2 2 2 9 2 2 2 2 9 2 7 3 5 6 7 5 2 5 6 5 7
[112] 6 7 3 3 7 6 5 5 3 7 3 5 3 7 5 3 3 6 5 5 5 6 3 6 5 7 6 3 7 7 7 3 7 7 7 3 6
[149] 7 3

Within cluster sum of squares by cluster:
[1] 1.409167 4.587619 3.552222 3.114091 4.655000 1.200000 2.444000 4.630714
[9] 3.375385
 (between_SS / total_SS =  95.7 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[1]]
K-means clustering with 10 clusters of sizes 11, 12, 12, 50, 9, 23, 6, 11, 4, 12

Cluster means:
   Sepal.Length Sepal.Width Petal.Length Petal.Width
1      6.090909    3.018182     4.663636    1.563636
2      5.541667    2.833333     4.275000    1.375000
3      7.475000    3.125000     6.300000    2.050000
4      5.006000    3.428000     1.462000    0.246000
5      6.200000    2.500000     4.944444    1.555556
6      6.560870    3.069565     5.526087    2.152174
7      5.766667    2.750000     5.050000    2.000000
8      6.663636    3.009091     4.627273    1.445455
9      5.000000    2.300000     3.275000    1.025000
10     5.700000    2.550000     3.875000    1.150000

Clustering vector:
  [1]  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4
 [26]  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4
 [51]  8  8  8 10  8  2  1  9  8  2  9  2 10  1 10  8  2 10  5 10  1 10  5  1  8
 [76]  8  8  8  1 10 10 10 10  5  2  1  8  5  2 10  2  1 10  9  2  2  2  1  9  2
[101]  6  7  3  6  6  3  2  3  6  3  6  6  6  7  7  6  6  3  3  5  6  7  3  5  6
[126]  3  1  1  6  3  3  3  6  5  5  3  6  6  1  6  6  6  7  6  6  6  5  6  6  7

Within cluster sum of squares by cluster:
 [1]  1.3163636  1.8208333  4.6550000 15.1510000  1.8044444  4.6052174
 [7]  0.4433333  1.1836364  0.2950000  1.5025000
 (between_SS / total_SS =  95.2 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[2]]
K-means clustering with 10 clusters of sizes 24, 21, 7, 15, 12, 23, 5, 12, 14, 17

Cluster means:
   Sepal.Length Sepal.Width Petal.Length Petal.Width
1      5.554167    2.604167     3.870833   1.1833333
2      5.147619    3.542857     1.509524   0.2809524
3      5.528571    4.042857     1.471429   0.2857143
4      6.133333    2.646667     4.820000   1.5066667
5      5.708333    2.850000     4.858333   1.8416667
6      6.560870    3.069565     5.526087   2.1521739
7      4.400000    2.880000     1.280000   0.2000000
8      7.475000    3.125000     6.300000   2.0500000
9      6.557143    3.050000     4.600000   1.4571429
10     4.794118    3.194118     1.452941   0.2000000

Clustering vector:
  [1]  2 10 10 10  2  3 10  2  7 10  2 10 10  7  3  3  3  2  3  2  2  2 10  2 10
 [26] 10  2  2  2 10 10  2  3  3 10 10  2  2  7  2  2  7  7  2  2 10  2 10  2 10
 [51]  9  9  9  1  9  4  9  1  9  1  1  1  1  4  1  9  5  1  4  1  5  1  4  4  9
 [76]  9  9  9  4  1  1  1  1  4  5  9  9  4  1  1  1  4  1  1  1  1  1  9  1  1
[101]  6  5  8  6  6  8  5  8  6  8  6  6  6  5  5  6  6  8  8  4  6  5  8  4  6
[126]  8  4  5  6  8  8  8  6  4  4  8  6  6  5  6  6  6  5  6  6  6  4  6  6  5

Within cluster sum of squares by cluster:
 [1] 7.1720833 1.7142857 0.8342857 3.2640000 2.9175000 4.6052174 0.5560000
 [8] 4.6550000 2.2235714 1.4611765
 (between_SS / total_SS =  95.7 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[3]]
K-means clustering with 10 clusters of sizes 21, 19, 12, 8, 18, 7, 11, 23, 9, 22

Cluster means:
   Sepal.Length Sepal.Width Petal.Length Petal.Width
1      5.628571    2.723810     4.133333    1.295238
2      4.678947    3.084211     1.378947    0.200000
3      7.475000    3.125000     6.300000    2.050000
4      5.512500    4.000000     1.475000    0.275000
5      6.450000    3.016667     4.605556    1.438889
6      5.242857    2.371429     3.442857    1.028571
7      6.218182    2.545455     4.963636    1.609091
8      5.100000    3.513043     1.526087    0.273913
9      5.844444    2.855556     4.977778    1.933333
10     6.568182    3.086364     5.536364    2.163636

Clustering vector:
  [1]  8  2  2  2  8  4  2  8  2  2  4  8  2  2  4  4  4  8  4  8  8  8  2  8  8
 [26]  2  8  8  8  2  2  8  4  4  2  2  8  8  2  8  8  2  2  8  8  2  8  2  8  8
 [51]  5  5  5  1  5  1  5  6  5  1  6  1  1  5  1  5  1  1  7  1  9  1  7  5  5
 [76]  5  5  5  5  6  6  6  1  7  1  5  5  7  1  1  1  5  1  6  1  1  1  5  6  1
[101] 10  9  3 10 10  3  1  3 10  3 10  7 10  9  9 10 10  3  3  7 10  9  3  7 10
[126]  3  7  9 10  3  3  3 10  7  7  3 10 10  9 10 10 10  9 10 10 10  7 10 10  9

Within cluster sum of squares by cluster:
 [1] 4.177143 2.488421 4.655000 0.958750 3.142222 1.262857 2.238182 2.094783
 [9] 0.980000 4.315455
 (between_SS / total_SS =  96.1 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[4]]
K-means clustering with 10 clusters of sizes 9, 20, 6, 33, 12, 22, 11, 17, 9, 11

Cluster means:
   Sepal.Length Sepal.Width Petal.Length Petal.Width
1      5.844444    2.855556     4.977778   1.9333333
2      5.610000    2.680000     4.095000   1.2650000
3      5.200000    2.366667     3.383333   1.0166667
4      4.818182    3.236364     1.433333   0.2303030
5      7.475000    3.125000     6.300000   2.0500000
6      6.568182    3.086364     5.536364   2.1636364
7      6.100000    3.027273     4.500000   1.4363636
8      5.370588    3.800000     1.517647   0.2764706
9      6.722222    3.000000     4.677778   1.4555556
10     6.218182    2.545455     4.963636   1.6090909

Clustering vector:
  [1]  4  4  4  4  4  8  4  4  4  4  8  4  4  4  8  8  8  4  8  8  8  8  4  4  4
 [26]  4  4  8  4  4  4  8  8  8  4  4  8  4  4  4  4  4  4  4  8  4  8  4  8  4
 [51]  9  7  9  2  9  2  7  3  9  2  3  7  2  7  2  9  7  2 10  2  1  2 10  7  7
 [76]  9  9  9  7  3  2  3  2 10  2  7  9 10  2  2  2  7  2  3  2  2  2  7  3  2
[101]  6  1  5  6  6  5  2  5  6  5  6 10  6  1  1  6  6  5  5 10  6  1  5 10  6
[126]  5 10  1  6  5  5  5  6 10 10  5  6  6  1  6  6  6  1  6  6  6 10  6  6  1

Within cluster sum of squares by cluster:
 [1] 0.9800000 3.9050000 1.0300000 5.4284848 4.6550000 4.3154545 1.3672727
 [8] 2.6305882 0.7933333 2.2381818
 (between_SS / total_SS =  96.0 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

[[5]]
K-means clustering with 10 clusters of sizes 18, 19, 7, 18, 22, 1, 12, 19, 28, 6

Cluster means:
   Sepal.Length Sepal.Width Petal.Length Petal.Width
1      4.688889    3.127778     1.383333   0.1944444
2      6.442105    2.978947     4.594737   1.4315789
3      5.528571    4.042857     1.471429   0.2857143
4      5.161111    3.538889     1.461111   0.2333333
5      6.568182    3.086364     5.536364   2.1636364
6      4.500000    2.300000     1.300000   0.3000000
7      7.475000    3.125000     6.300000   2.0500000
8      6.036842    2.705263     5.000000   1.7789474
9      5.532143    2.635714     3.960714   1.2285714
10     4.966667    3.466667     1.716667   0.3833333

Clustering vector:
  [1]  4  1  1  1  4  3  1  4  1  1  4 10  1  1  3  3  3  4  3  4  4  4  1 10 10
 [26]  1 10  4  4  1  1  4  3  3  1  1  4  4  1  4  4  6  1 10 10  1  4  1  4  4
 [51]  2  2  2  9  2  9  2  9  2  9  9  9  9  2  9  2  9  9  8  9  8  9  8  2  2
 [76]  2  2  2  2  9  9  9  9  8  9  2  2  2  9  9  9  2  9  9  9  9  9  2  9  9
[101]  5  8  7  5  5  7  9  7  5  7  5  8  5  8  8  5  5  7  7  8  5  8  7  8  5
[126]  7  8  8  5  7  7  7  5  8  8  7  5  5  8  5  5  5  8  5  5  5  8  5  5  8

Within cluster sum of squares by cluster:
 [1] 1.7883333 3.7084211 0.8342857 1.1483333 4.3154545 0.0000000 4.6550000
 [8] 4.1252632 9.7492857 0.4833333
 (between_SS / total_SS =  95.5 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"

>
> stopCluster(cl)
[1] 1
> mpi.exit()
[1] "Detaching Rmpi. Rmpi cannot be used unless relaunching R."
>
> proc.time()
   user  system elapsed
  1.937   0.219   2.665
```
