---
layout: single
author_profile: false
---

You can setup a SLURM cluster on Azure using AZHOP. This [blog](https://techcommunity.microsoft.com/t5/azure-high-performance-computing/az-hop-in-the-azure-marketplace/ba-p/3829838) has details on how to deploy AZHOP.

### Cluster information
```bash
$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
execute      up   infinite    256  idle~ execute-[1-256]
nc48v4       up   infinite      1  idle~ nc48v4-pg0-2
nc48v4       up   infinite      1    mix nc48v4-pg0-1
nc96v4       up   infinite      1  idle~ nc96v4-pg0-1
nc96v4       up   infinite      1   idle nc96v4-pg0-2
ncrv3        up   infinite      2  idle~ ncrv3-[1-2]
ncv3         up   infinite      1  idle~ ncv3-2
ncv3         up   infinite      1   idle ncv3-1
ndv4*        up   infinite      1  comp% ndv4-pg0-1
ndv4*        up   infinite      1  idle% ndv4-pg0-2
```
The image used in all N-series VMs is `microsoft-dsvm:ubuntu-hpc:2004:20.04.2023031501`.

### This post will compare nc48v4 and nc96v4, which have 2 and 4 80G A100 GPUs, respectively. 
**NC48v4**
```bash
$ scontrol show node nc48v4-pg0-1
NodeName=nc48v4-pg0-1 Arch=x86_64 CoresPerSocket=1 
   CPUAlloc=0 CPUTot=48 CPULoad=0.00
   AvailableFeatures=cloud
   ActiveFeatures=cloud
   Gres=gpu:2
   NodeAddr=nc48v4-pg0-1 NodeHostName=nc48v4-pg0-1 Version=20.11.9
   OS=Linux 5.15.0-1034-azure #41~20.04.1-Ubuntu SMP Sat Feb 11 17:02:42 UTC 2023
   RealMemory=414515 AllocMem=0 FreeMem=438726 Sockets=48 Boards=1
   State=IDLE+CLOUD ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=nc48v4
   BootTime=2023-07-14T18:41:10 SlurmdStartTime=2023-07-14T18:41:11
   CfgTRES=cpu=48,mem=414515M,billing=48
   AllocTRES=
   CapWatts=n/a
   CurrentWatts=0 AveWatts=0
   ExtSensorsJoules=n/s ExtSensorsWatts=0 ExtSensorsTemp=n/s
   Comment=(null)
```
```bash
clusteradmin@nc48v4-pg0-1:~/NCCL_test$ nvidia-smi 
Fri Jul 14 20:10:11 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.85.02    Driver Version: 510.85.02    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100 80G...  Off  | 00000001:00:00.0 Off |                    0 |
| N/A   37C    P0    53W / 300W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100 80G...  Off  | 00000002:00:00.0 Off |                    0 |
| N/A   38C    P0    54W / 300W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
```bash
clusteradmin@nc48v4-pg0-1:~/NCCL_test$ nvidia-smi topo -m
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV12    0-1     0-1
GPU1    NV12     X      0-1     0-1

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

**NC96v4**
```bash
$ scontrol show node nc96v4-pg0-2
NodeName=nc96v4-pg0-2 Arch=x86_64 CoresPerSocket=1 
   CPUAlloc=0 CPUTot=96 CPULoad=0.00
   AvailableFeatures=cloud
   ActiveFeatures=cloud
   Gres=gpu:4
   NodeAddr=nc96v4-pg0-2 NodeHostName=nc96v4-pg0-2 Version=20.11.9
   OS=Linux 5.15.0-1034-azure #41~20.04.1-Ubuntu SMP Sat Feb 11 17:02:42 UTC 2023
   RealMemory=829030 AllocMem=0 FreeMem=879666 Sockets=96 Boards=1
   State=IDLE+CLOUD ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=nc96v4
   BootTime=2023-07-14T04:57:25 SlurmdStartTime=2023-07-14T04:57:28
   CfgTRES=cpu=96,mem=829030M,billing=96
   AllocTRES=
   CapWatts=n/a
   CurrentWatts=0 AveWatts=0
   ExtSensorsJoules=n/s ExtSensorsWatts=0 ExtSensorsTemp=n/s
   Comment=(null)
```
```bash
clusteradmin@nc96v4-pg0-2:~/NCCL_test$ nvidia-smi 
Fri Jul 14 20:13:54 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.85.02    Driver Version: 510.85.02    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100 80G...  Off  | 00000001:00:00.0 Off |                    0 |
| N/A   38C    P0    54W / 300W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100 80G...  Off  | 00000002:00:00.0 Off |                    0 |
| N/A   38C    P0    58W / 300W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A100 80G...  Off  | 00000003:00:00.0 Off |                    0 |
| N/A   37C    P0    52W / 300W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A100 80G...  Off  | 00000004:00:00.0 Off |                    0 |
| N/A   39C    P0    55W / 300W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
```bash
clusteradmin@nc96v4-pg0-2:~/NCCL_test$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV12    SYS     SYS     0       0-3
GPU1    NV12     X      SYS     SYS     0       0-3
GPU2    SYS     SYS      X      NV12    0       0-3
GPU3    SYS     SYS     NV12     X      0       0-3

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

## NCCL benchmark
There are two preset NCCL environment variables, `NCCL_TOPO_FILE` and `NCCL_GRAPH_FILE`, in the `/etc/nccl.conf` file on the compute VM.
```bash
$ cat /etc/nccl.conf 
NCCL_TOPO_FILE=/opt/microsoft/ncv4/topo.xml
NCCL_GRAPH_FILE=/opt/microsoft/ncv4/graph.xml
```
**In order to run NCCL test with SLURM, you need to install pmix following the instructions [here](https://github.com/Azure/azurehpc/blob/df46027e0380aee06a292b54ed6a1d90a6f5a1db/experimental/deploy_cycle_slurm_ndv4/scripts/install-pmix.sh) on the compute node.**
### NC96v4
#### Test with both `NCCL_TOPO_FILE` and `NCCL_GRAPH_FILE` being set
SLURM script
```bash
#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH -p nc96v4
#SBATCH -w nc96v4-pg0-2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH -o job.%J.out
#SBATCH --error=job.%J.err

BASE_DIR=/opt
NCCL_TESTS_EXE=all_reduce_perf

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

source /etc/profile.d/modules.sh
module load mpi/openmpi

PIN_MASK='0xffffff,0xffffff000000,0xffffff000000000000,0xffffff000000000000000000'

srun --mpi=pmix --cpu-bind=mask_cpu:$PIN_MASK --gpus-per-node=4 \
--ntasks-per-node=4 \
${BASE_DIR}/nccl-tests/build/$NCCL_TESTS_EXE -b8 -f 2 -g 1 -e 8G -c 1
```
NCCL results
```bash
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
nc96v4-pg0-2:89444:89486 [1] NCCL INFO comm 0x55a7dcf2de60 rank 1 nranks 4 cudaDev 1 busId 200000 - Init COMPLETE
nc96v4-pg0-2:89445:89488 [2] NCCL INFO comm 0x559348ec5c50 rank 2 nranks 4 cudaDev 2 busId 300000 - Init COMPLETE
           8             2     float     sum      -1    13.14    0.00    0.00      0    13.17    0.00    0.00      0
          16             4     float     sum      -1    13.23    0.00    0.00      0    13.43    0.00    0.00      0
          32             8     float     sum      -1    13.29    0.00    0.00      0    13.07    0.00    0.00      0
          64            16     float     sum      -1    13.37    0.00    0.01      0    13.15    0.00    0.01      0
         128            32     float     sum      -1    13.51    0.01    0.01      0    13.31    0.01    0.01      0
         256            64     float     sum      -1    13.64    0.02    0.03      0    13.81    0.02    0.03      0
         512           128     float     sum      -1    13.72    0.04    0.06      0    13.82    0.04    0.06      0
        1024           256     float     sum      -1    15.17    0.07    0.10      0    14.78    0.07    0.10      0
        2048           512     float     sum      -1    16.16    0.13    0.19      0    16.05    0.13    0.19      0
        4096          1024     float     sum      -1    17.09    0.24    0.36      0    16.90    0.24    0.36      0
        8192          2048     float     sum      -1    17.92    0.46    0.69      0    17.50    0.47    0.70      0
       16384          4096     float     sum      -1    19.73    0.83    1.25      0    18.89    0.87    1.30      0
       32768          8192     float     sum      -1    20.93    1.57    2.35      0    20.83    1.57    2.36      0
       65536         16384     float     sum      -1    21.88    3.00    4.49      0    21.50    3.05    4.57      0
      131072         32768     float     sum      -1    31.54    4.16    6.23      0    31.33    4.18    6.28      0
      262144         65536     float     sum      -1    64.60    4.06    6.09      0    64.06    4.09    6.14      0
      524288        131072     float     sum      -1    73.72    7.11   10.67      0    73.69    7.11   10.67      0
     1048576        262144     float     sum      -1    93.18   11.25   16.88      0    92.97   11.28   16.92      0
     2097152        524288     float     sum      -1    136.8   15.33   23.00      0    136.2   15.40   23.10      0
     4194304       1048576     float     sum      -1    225.1   18.63   27.94      0    227.1   18.47   27.71      0
     8388608       2097152     float     sum      -1    437.5   19.17   28.76      0    435.1   19.28   28.92      0
    16777216       4194304     float     sum      -1    865.0   19.39   29.09      0    872.7   19.22   28.84      0
    33554432       8388608     float     sum      -1   1761.5   19.05   28.57      0   1747.0   19.21   28.81      0
    67108864      16777216     float     sum      -1   3362.9   19.96   29.93      0   3374.3   19.89   29.83      0
   134217728      33554432     float     sum      -1   6646.9   20.19   30.29      0   6668.2   20.13   30.19      0
   268435456      67108864     float     sum      -1    13144   20.42   30.63      0    13206   20.33   30.49      0
   536870912     134217728     float     sum      -1    26266   20.44   30.66      0    26160   20.52   30.78      0
  1073741824     268435456     float     sum      -1    52288   20.53   30.80      0    52474   20.46   30.69      0
  2147483648     536870912     float     sum      -1   105840   20.29   30.43      0   104302   20.59   30.88      0
  4294967296    1073741824     float     sum      -1   216222   19.86   29.80      0   215370   19.94   29.91      0
  8589934592    2147483648     float     sum      -1   459314   18.70   28.05      0   459949   18.68   28.01      0
nc96v4-pg0-2:89444:89444 [1] NCCL INFO comm 0x55a7dcf2de60 rank 1 nranks 4 cudaDev 1 busId 200000 - Destroy COMPLETE
nc96v4-pg0-2:89446:89446 [3] NCCL INFO comm 0x556a7b4d7bf0 rank 3 nranks 4 cudaDev 3 busId 400000 - Destroy COMPLETE
nc96v4-pg0-2:89443:89443 [0] NCCL INFO comm 0x56323adcad80 rank 0 nranks 4 cudaDev 0 busId 100000 - Destroy COMPLETE
nc96v4-pg0-2:89445:89445 [2] NCCL INFO comm 0x559348ec5c50 rank 2 nranks 4 cudaDev 2 busId 300000 - Destroy COMPLETE
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 13.7946
```
#### Test with only `NCCL_TOPO_FILE`. Comment out `NCCL_GRAPH_FILE=/opt/microsoft/ncv4/graph.xml` in `/etc/nccl.conf`.
NCCL results
```bash
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
nc96v4-pg0-2:89786:89819 [2] NCCL INFO comm 0x55fb36d03b10 rank 2 nranks 4 cudaDev 2 busId 300000 - Init COMPLETE
           8             2     float     sum      -1    13.38    0.00    0.00      0    13.11    0.00    0.00      0
          16             4     float     sum      -1    13.26    0.00    0.00      0    13.02    0.00    0.00      0
          32             8     float     sum      -1    13.30    0.00    0.00      0    13.18    0.00    0.00      0
          64            16     float     sum      -1    13.37    0.00    0.01      0    13.46    0.00    0.01      0
         128            32     float     sum      -1    13.43    0.01    0.01      0    13.37    0.01    0.01      0
         256            64     float     sum      -1    13.65    0.02    0.03      0    13.36    0.02    0.03      0
         512           128     float     sum      -1    13.69    0.04    0.06      0    13.71    0.04    0.06      0
        1024           256     float     sum      -1    15.31    0.07    0.10      0    15.12    0.07    0.10      0
        2048           512     float     sum      -1    16.19    0.13    0.19      0    15.73    0.13    0.20      0
        4096          1024     float     sum      -1    17.17    0.24    0.36      0    16.72    0.25    0.37      0
        8192          2048     float     sum      -1    18.11    0.45    0.68      0    17.35    0.47    0.71      0
       16384          4096     float     sum      -1    19.63    0.83    1.25      0    19.23    0.85    1.28      0
       32768          8192     float     sum      -1    21.48    1.53    2.29      0    20.84    1.57    2.36      0
       65536         16384     float     sum      -1    21.87    3.00    4.49      0    21.64    3.03    4.54      0
      131072         32768     float     sum      -1    31.87    4.11    6.17      0    31.61    4.15    6.22      0
      262144         65536     float     sum      -1    64.36    4.07    6.11      0    64.34    4.07    6.11      0
      524288        131072     float     sum      -1    74.00    7.09   10.63      0    73.69    7.11   10.67      0
     1048576        262144     float     sum      -1    93.75   11.19   16.78      0    93.54   11.21   16.82      0
     2097152        524288     float     sum      -1    137.2   15.29   22.93      0    137.1   15.30   22.95      0
     4194304       1048576     float     sum      -1    228.5   18.36   27.54      0    228.3   18.37   27.56      0
     8388608       2097152     float     sum      -1    436.9   19.20   28.80      0    435.6   19.26   28.89      0
    16777216       4194304     float     sum      -1    866.6   19.36   29.04      0    870.8   19.27   28.90      0
    33554432       8388608     float     sum      -1   1731.9   19.37   29.06      0   1736.4   19.32   28.99      0
    67108864      16777216     float     sum      -1   3360.6   19.97   29.95      0   3330.4   20.15   30.23      0
   134217728      33554432     float     sum      -1   6599.3   20.34   30.51      0   6616.6   20.28   30.43      0
   268435456      67108864     float     sum      -1    13043   20.58   30.87      0    13134   20.44   30.66      0
   536870912     134217728     float     sum      -1    26168   20.52   30.77      0    26043   20.61   30.92      0
  1073741824     268435456     float     sum      -1    51970   20.66   30.99      0    51754   20.75   31.12      0
  2147483648     536870912     float     sum      -1   104730   20.50   30.76      0   103974   20.65   30.98      0
  4294967296    1073741824     float     sum      -1   214739   20.00   30.00      0   214882   19.99   29.98      0
  8589934592    2147483648     float     sum      -1   456716   18.81   28.21      0   457441   18.78   28.17      0
nc96v4-pg0-2:89784:89784 [0] NCCL INFO comm 0x55ed0c7d34a0 rank 0 nranks 4 cudaDev 0 busId 100000 - Destroy COMPLETE
nc96v4-pg0-2:89785:89785 [1] NCCL INFO comm 0x55919eac19b0 rank 1 nranks 4 cudaDev 1 busId 200000 - Destroy COMPLETE
nc96v4-pg0-2:89787:89787 [3] NCCL INFO comm 0x556a167eb1d0 rank 3 nranks 4 cudaDev 3 busId 400000 - Destroy COMPLETE
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 13.8362
#
nc96v4-pg0-2:89786:89786 [2] NCCL INFO comm 0x55fb36d03b10 rank 2 nranks 4 cudaDev 2 busId 300000 - Destroy COMPLETE
```
### NC96v4
#### Test with both `NCCL_TOPO_FILE` and `NCCL_GRAPH_FILE` being set
SLURM script
```bash
#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH -p nc48v4
#SBATCH -w nc48v4-pg0-1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=2
#SBATCH --mem=0
#SBATCH -o job.%J.out
#SBATCH --error=job.%J.err

BASE_DIR=/opt
NCCL_TESTS_EXE=all_reduce_perf

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

source /etc/profile.d/modules.sh
module load mpi/openmpi

PIN_MASK='0xffffff,0xffffff000000'

srun --mpi=pmix --cpu-bind=mask_cpu:$PIN_MASK --gpus-per-node=2 \
--ntasks-per-node=2 \
${BASE_DIR}/nccl-tests/build/$NCCL_TESTS_EXE -b8 -f 2 -g 1 -e 8G -c 1
```
NCCL results
```bash
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1     8.71    0.00    0.00      0     8.66    0.00    0.00      0
          16             4     float     sum      -1     8.76    0.00    0.00      0     8.71    0.00    0.00      0
          32             8     float     sum      -1     8.77    0.00    0.00      0     8.77    0.00    0.00      0
          64            16     float     sum      -1     8.86    0.01    0.01      0     8.75    0.01    0.01      0
         128            32     float     sum      -1     8.91    0.01    0.01      0     8.69    0.01    0.01      0
         256            64     float     sum      -1     8.86    0.03    0.03      0     8.71    0.03    0.03      0
         512           128     float     sum      -1     9.06    0.06    0.06      0     8.73    0.06    0.06      0
        1024           256     float     sum      -1     9.55    0.11    0.11      0     9.22    0.11    0.11      0
        2048           512     float     sum      -1     9.37    0.22    0.22      0     9.26    0.22    0.22      0
        4096          1024     float     sum      -1     9.57    0.43    0.43      0     9.28    0.44    0.44      0
        8192          2048     float     sum      -1    10.32    0.79    0.79      0    10.12    0.81    0.81      0
       16384          4096     float     sum      -1    11.13    1.47    1.47      0    10.80    1.52    1.52      0
       32768          8192     float     sum      -1    11.21    2.92    2.92      0    10.97    2.99    2.99      0
       65536         16384     float     sum      -1    13.35    4.91    4.91      0    12.80    5.12    5.12      0
      131072         32768     float     sum      -1    30.14    4.35    4.35      0    30.03    4.36    4.36      0
      262144         65536     float     sum      -1    32.36    8.10    8.10      0    32.42    8.09    8.09      0
      524288        131072     float     sum      -1    37.54   13.97   13.97      0    37.00   14.17   14.17      0
     1048576        262144     float     sum      -1    47.17   22.23   22.23      0    46.85   22.38   22.38      0
     2097152        524288     float     sum      -1    67.65   31.00   31.00      0    66.91   31.34   31.34      0
     4194304       1048576     float     sum      -1    102.7   40.83   40.83      0    101.9   41.18   41.18      0
     8388608       2097152     float     sum      -1    170.6   49.17   49.17      0    170.5   49.21   49.21      0
    16777216       4194304     float     sum      -1    307.9   54.49   54.49      0    305.5   54.92   54.92      0
    33554432       8388608     float     sum      -1    599.0   56.01   56.01      0    592.3   56.65   56.65      0
    67108864      16777216     float     sum      -1   1185.5   56.61   56.61      0   1171.0   57.31   57.31      0
   134217728      33554432     float     sum      -1   2344.4   57.25   57.25      0   2326.3   57.69   57.69      0
   268435456      67108864     float     sum      -1   4681.4   57.34   57.34      0   4637.0   57.89   57.89      0
   536870912     134217728     float     sum      -1   9346.6   57.44   57.44      0   9257.0   58.00   58.00      0
  1073741824     268435456     float     sum      -1    18693   57.44   57.44      0    18532   57.94   57.94      0
  2147483648     536870912     float     sum      -1    37361   57.48   57.48      0    37038   57.98   57.98      0
  4294967296    1073741824     float     sum      -1    74642   57.54   57.54      0    74055   58.00   58.00      0
  8589934592    2147483648     float     sum      -1   149286   57.54   57.54      0   147984   58.05   58.05      0
nc48v4-pg0-1:74653:74653 [1] NCCL INFO comm 0x563243d4a050 rank 1 nranks 2 cudaDev 1 busId 200000 - Destroy COMPLETE
nc48v4-pg0-1:74652:74652 [0] NCCL INFO comm 0x558b00385b60 rank 0 nranks 2 cudaDev 0 busId 100000 - Destroy COMPLETE
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 24.2941
#
```
#### Test with only `NCCL_TOPO_FILE`. Comment out `NCCL_GRAPH_FILE=/opt/microsoft/ncv4/graph.xml` in `/etc/nccl.conf`.
NCCL results
```bash
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1     8.74    0.00    0.00      0     8.59    0.00    0.00      0
          16             4     float     sum      -1     8.49    0.00    0.00      0     8.43    0.00    0.00      0
          32             8     float     sum      -1     8.56    0.00    0.00      0     8.50    0.00    0.00      0
          64            16     float     sum      -1     8.66    0.01    0.01      0     8.52    0.01    0.01      0
         128            32     float     sum      -1     8.62    0.01    0.01      0     8.43    0.02    0.02      0
         256            64     float     sum      -1     8.70    0.03    0.03      0     8.48    0.03    0.03      0
         512           128     float     sum      -1     8.61    0.06    0.06      0     8.51    0.06    0.06      0
        1024           256     float     sum      -1     9.22    0.11    0.11      0     8.87    0.12    0.12      0
        2048           512     float     sum      -1     9.27    0.22    0.22      0     9.94    0.21    0.21      0
        4096          1024     float     sum      -1     9.48    0.43    0.43      0     9.21    0.44    0.44      0
        8192          2048     float     sum      -1    10.49    0.78    0.78      0    10.00    0.82    0.82      0
       16384          4096     float     sum      -1    11.07    1.48    1.48      0    10.86    1.51    1.51      0
       32768          8192     float     sum      -1    11.26    2.91    2.91      0    11.82    2.77    2.77      0
       65536         16384     float     sum      -1    11.54    5.68    5.68      0    11.53    5.68    5.68      0
      131072         32768     float     sum      -1    12.16   10.78   10.78      0    11.90   11.02   11.02      0
      262144         65536     float     sum      -1    14.07   18.64   18.64      0    13.74   19.08   19.08      0
      524288        131072     float     sum      -1    17.20   30.48   30.48      0    17.21   30.47   30.47      0
     1048576        262144     float     sum      -1    33.18   31.60   31.60      0    32.99   31.78   31.78      0
     2097152        524288     float     sum      -1    40.34   51.99   51.99      0    40.17   52.21   52.21      0
     4194304       1048576     float     sum      -1    50.02   83.85   83.85      0    49.50   84.73   84.73      0
     8388608       2097152     float     sum      -1    79.09  106.07  106.07      0    77.09  108.82  108.82      0
    16777216       4194304     float     sum      -1    117.2  143.12  143.12      0    115.9  144.81  144.81      0
    33554432       8388608     float     sum      -1    209.2  160.39  160.39      0    208.4  161.03  161.03      0
    67108864      16777216     float     sum      -1    374.7  179.11  179.11      0    374.1  179.37  179.37      0
   134217728      33554432     float     sum      -1    724.9  185.16  185.16      0    724.0  185.38  185.38      0
   268435456      67108864     float     sum      -1   1393.9  192.58  192.58      0   1394.1  192.55  192.55      0
   536870912     134217728     float     sum      -1   2718.0  197.53  197.53      0   2722.0  197.24  197.24      0
  1073741824     268435456     float     sum      -1   5196.2  206.64  206.64      0   5206.0  206.25  206.25      0
  2147483648     536870912     float     sum      -1   9985.7  215.06  215.06      0   9954.4  215.73  215.73      0
  4294967296    1073741824     float     sum      -1    19344  222.03  222.03      0    19362  221.82  221.82      0
  8589934592    2147483648     float     sum      -1    38177  225.00  225.00      0    38158  225.11  225.11      0
nc48v4-pg0-1:74928:74928 [1] NCCL INFO comm 0x561a508c4cf0 rank 1 nranks 2 cudaDev 1 busId 200000 - Destroy COMPLETE
nc48v4-pg0-1:74927:74927 [0] NCCL INFO comm 0x562aaaf43cc0 rank 0 nranks 2 cudaDev 0 busId 100000 - Destroy COMPLETE
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 73.4005
#
```

### Conclusion
With `NCCL_GRAPH_FILE`, NC96v4 does not have NCCL performance difference. But on NC48v4, disabling `NCCL_GRAPH_FILE` will 4x NCCL_allreduce BW. 
