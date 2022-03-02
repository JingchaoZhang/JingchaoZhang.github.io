---
layout: single
author_profile: false
---


Get the docker image from NVIDIA [website](https://docs.nvidia.com/deeplearning/modulus/index.html). You need to register and login.
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

Run in batch mode
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=500GB
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=72:00:00
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err

ml load singularity/3.7.4 cuda/11.4.3
cd /home/jingchao.zhang/red/modulus/simple_cubic/sif
#add srun here to pass the mpi flag
srun --mpi=pmi2 singularity exec --nv --writable --bind .:/mnt /home/jingchao.zhang/red/modulus/sif/modulus python -u /mnt/simple_cubic.py
```
  
  
  
  
  
  
  
Fix memory explosion issue. Need to edit source code in the container.
```
singularity shell --nv --writable --bind .:/mnt /home/jingchao.zhang/red/modulus/sif/modulus
vim /usr/local/lib/python3.8/dist-packages/modulus-21.6-py3.8.egg/modulus/solver.py
#add the following two lines after line 224 "config = tf.ConfigProto()"
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = str(hvd.local_rank())
#save the file and quit the container      
```

Sample submission file for 8 GPUs on a single node
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=500GB
#SBATCH --partition=gpu
#SBATCH --gpus=a100:8
#SBATCH --reservation=monai
#SBATCH --time=72:00:00
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err

ml load singularity/3.7.4 cuda/11.4.3

srun --mpi=pmi2 singularity exec --nv --writable --bind .:/mnt /home/jingchao.zhang/red/modulus/sif/modulus horovodrun -np 8 python -u /mnt/simple_cubic_multiGPU.py
```

After the fix, all GPUs are fully utilized
```
Wed Mar  2 01:01:36 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:07:00.0 Off |                    0 |
| N/A   42C    P0   158W / 400W |  18749MiB / 81251MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM...  On   | 00000000:0F:00.0 Off |                    0 |
| N/A   40C    P0   163W / 400W |  19901MiB / 81251MiB |     80%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM...  On   | 00000000:47:00.0 Off |                    0 |
| N/A   42C    P0   157W / 400W |  19901MiB / 81251MiB |     98%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM...  On   | 00000000:4E:00.0 Off |                    0 |
| N/A   42C    P0   165W / 400W |  19901MiB / 81251MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA A100-SXM...  On   | 00000000:87:00.0 Off |                    0 |
| N/A   55C    P0   173W / 400W |  19901MiB / 81251MiB |     99%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA A100-SXM...  On   | 00000000:90:00.0 Off |                    0 |
| N/A   55C    P0   178W / 400W |  19901MiB / 81251MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA A100-SXM...  On   | 00000000:B7:00.0 Off |                    0 |
| N/A   53C    P0   143W / 400W |  19901MiB / 81251MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA A100-SXM...  On   | 00000000:BD:00.0 Off |                    0 |
| N/A   56C    P0   194W / 400W |  18749MiB / 81251MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
```
