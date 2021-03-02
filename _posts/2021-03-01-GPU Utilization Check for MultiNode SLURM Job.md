---
layout: single
author_profile: false
---

Get a snapshot of GPU stats without [DCGM](https://developer.nvidia.com/dcgm).

GPU query command to get card utilization, temperature, fan speed and power consumption
```
nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,fan.speed,temperature.gpu,memory.used,memory.free
```

A complete list of options
```
nvidia-smi --help-query-gpu
```

ssh and check ultilization
```
NODES=$(scontrol show hostname `squeue -j 64799853 --noheader -o %N`)
for ssh_host in $NODES
do
  echo $ssh_host
  ssh -q $ssh_host "nvidia-smi --format=csv --query-gpu=utilization.gpu,utilization.memory"
done
```
