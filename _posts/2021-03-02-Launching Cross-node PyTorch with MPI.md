---
layout: single
author_profile: false
---

Running these via mpirun or srun like on https://developer.nvidia.com/blog/how-to-run-ngc-deep-learning-containers-with-singularity/ would help to ensure correlation of errors across nodes.  It would also take out the complexity of getting the torch.distributed.launch parameters right for each node, because you could just skip that entirely -- if you just start up the right number of processes per node through mpirun singularity run --nv ..., our built-in Singularity support will take care of the rest pretty much entirely.

Make sure your example.py understands how to get its LOCAL_RANK from the environment like with torch.distributed.launch --use_env. You can be compatible with both forms if you have your local_rank argument done like this in your training script: 
```bash
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK',0), type=int)
```
A full example command line using srun then might be like this: 
```bash
srun -N2 --ntasks-per-node=2 singularity run --nv pytorch_20.12-py3.sif python example.py
```
Note I'm assuming the --local_world_size=4 at the end is not needed; I've never seen a pyt distributed script that needed that.


If you're using mpirun without SLURM, you'll need to set the MASTER_ADDR and MASTER_PORT env vars beforehand, but those are easy because they will be the same for all nodes.  Maybe like this:
  
# run four tasks each on hosts aa and bb
```bash
export MASTER_ADDR=aa
export MASTER_PORT=1234 # or whatever
mpirun -H aa,bb -npernode 4 singularity run --nv pytorch_20.12-py3.sif python example.py
```
