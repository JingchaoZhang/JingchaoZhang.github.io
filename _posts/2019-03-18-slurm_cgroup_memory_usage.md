---
layout: single
author_profile: false
---

Put the line below at the end of a slurm submit script to find out the memory usage.
```bash
cgget -g memory /slurm/uid_${UID}/job_${SLURM_JOB_ID}
```
