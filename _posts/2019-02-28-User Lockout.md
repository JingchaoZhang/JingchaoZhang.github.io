---
layout: single
author_profile: false
---

Global lock. 
```bash
usermod.py --lock user
```

Locally lock for the current session. Limit CPU usage to 1. 
```bash
for i in `pgrep -u badperson` ; do taskset --cpu-list -p 0 $i ; done
```
