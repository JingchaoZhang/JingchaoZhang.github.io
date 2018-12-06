---
layout: single
author_profile: false
---

```console
[root@beta ~]# source keystonerc_admin

[root@beta ~(keystone_admin)]$ rbd -p vms ls | grep 87f7c8db-8f4b-43b4-be8f-348bbc88c403
87f7c8db-8f4b-43b4-be8f-348bbc88c403_disk

[root@beta ~(keystone_admin)]$ rbd map -p vms 87f7c8db-8f4b-43b4-be8f-348bbc88c403_disk
/dev/rbd0

[root@beta ~(keystone_admin)]$  fdisk -l /dev/rbd0
Disk /dev/rbd0: 42.9 GB, 42949672960 bytes, 83886080 sectors
Units = sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 8388608 bytes / 8388608 bytes
Disk label type: dos
Disk identifier: 0x000abb0d

     Device Boot      Start         End      Blocks   Id  System
/dev/rbd0p1   *        2048    83886046    41941999+  83  Linux
Partition 1 does not start on physical sector boundary.

[root@beta ~(keystone_admin)]$ mount /dev/rbd0p1 ./temp/

[root@beta ~(keystone_admin)]$ cd temp/

[root@beta etc(keystone_admin)]$ cd

[root@beta ~(keystone_admin)]$ umount ./temp

[root@beta ~(keystone_admin)]$ rbd unmap /dev/rbd0
```
