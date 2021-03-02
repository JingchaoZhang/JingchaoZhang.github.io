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

Two ways of ssh and check ultilization
```
#scontrol show hostname `squeue -j JOBID --noheader -o %N` | while read -r HOST
#do
#  echo "$HOST"
#  ssh -q $HOST "nvidia-smi --format=csv --query-gpu=utilization.gpu,memory.used,memory.free"
#done

scontrol show hostname `squeue -j JOBID --noheader -o %N` > ssh_hosts.txt
for ssh_host in $(cat ssh_hosts.txt)
do
  echo $ssh_host
  ssh -q $ssh_host "nvidia-smi --format=csv --query-gpu=utilization.gpu,utilization.memory"
done
rm ssh_hosts.txt
```
