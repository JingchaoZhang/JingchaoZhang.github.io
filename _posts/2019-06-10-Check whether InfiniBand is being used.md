---
layout: single
author_profile: false
---

Check for number incrementals in files at /sys/class/infiniband/mlx4_0/ports/1/counters/
```bash
watch -n .5 -diff=cumm 'grep -H . /sys/class/infiniband/mlx4_0/ports/1/counters/*'
```
