---
layout: single
author_profile: false
---

Setup daily cron job on each cluster running command "sacct --allusers --parsable2 --noheader --allocations --format jobid,jobidraw,cluster,partition,account,group,gid,user,uid,submit,eligible,start,end,elapsed,exitcode,state,nnodes,ncpus,reqcpus,reqmem,timelimit,nodelist,jobname --state CANCELLED,COMPLETED,FAILED,NODE_FAIL,PREEMPTED,TIMEOUT --starttime 2016-10-30T00:00:00 > sandhills-slurm.log".  
Push results to XDMoD VM.  
xdmod-shredder -r Crane -f slurm -i crane-slurm.log  
xdmod-shredder -r Sandhills -f slurm -i sandhills-slurm.log  
xdmod-shredder -r Tusker -f slurm -i tusker-slurm.log  
xdmod-ingestor
