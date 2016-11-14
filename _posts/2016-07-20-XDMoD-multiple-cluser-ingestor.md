---
layout: single
author_profile: false
---

 xdmod-shredder -r Crane -f slurm -i crane-slurm.log;
 xdmod-shredder -r Sandhills -f slurm -i sandhills-slurm.log;
 xdmod-shredder -r Tusker -f slurm -i tusker-slurm.log;
 xdmod-ingestor;
