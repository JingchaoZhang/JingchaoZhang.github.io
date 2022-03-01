---
layout: single
author_profile: false
---


Get the docker image from NVIDIA [wesite](https://docs.nvidia.com/deeplearning/modulus/index.html). You need to register and login.
Download file is `modulus_image_v21.06.tar.gz` (5.7G).
Build a singularity image.
```
ml load singularity/3.7.4 
singularity build --sandbox modulus docker-archive://modulus_image_v21.06.tar.gz
```

```
srun --pty -N 1 -n 1 --mem=80G --partition=gpu --gpus=a100:1 --time=108:00:00 --mpi=pmi2 bash
cd /home/jingchao.zhang/red/modulus/sif

ml load singularity/3.7.4 cuda/11.4.3
singularity shell --nv --writable --bind /home/jingchao.zhang/red/modulus:/mnt modulus
```

From within the sif shell
```
cd /mnt/simple_cubic/sif/
python simple_cubic.py
```
