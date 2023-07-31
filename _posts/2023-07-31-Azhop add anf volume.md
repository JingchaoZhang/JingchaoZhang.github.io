---
layout: single
author_profile: false
---

### Network topology
When create AZHOP without NetApp volumes, a subnet for ANF will still be created as shown in figure below:  
![Figure_1](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-31-2-figures/1.png)  
You can manually add a ANF volume to the existing AZHOP cluster, which will change the network topology as below:  
![Figure_1](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-31-2-figures/1_1.png)  

### Create Azure Netapp Files
1. Select ANF service from MarketPlace  
![Figure_2](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-31-2-figures/2.png)  
2. Create NetApp account  
![Figure_3](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-31-2-figures/3.png)  
3. Create ANF capacity pool  
![Figure_4](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-31-2-figures/4.png)  
![Figure_5](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-31-2-figures/5.png)  
![Figure_6](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-31-2-figures/6.png)  
4. Create ANF volume  
![Figure_7](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-31-2-figures/7.png)  
![Figure_8](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-31-2-figures/8.png)  
5. Find the mounting instructions from the volume page  
![Figure_9](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-31-2-figures/9.png)  
![Figure_10](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-31-2-figures/10.png)  

### Mount ANF to the ondemand node
```bash
[root@ondemand ~]# cd /
[root@ondemand /]# sudo mkdir NewVolume
[root@ondemand /]# sudo mount -t nfs -o rw,hard,rsize=262144,wsize=262144,vers=3,tcp 10.107.0.36:/NewVolume NewVolume
[root@ondemand /]# df -h
Filesystem                                                                    Size  Used Avail Use% Mounted on
devtmpfs                                                                       16G     0   16G   0% /dev
tmpfs                                                                          16G     0   16G   0% /dev/shm
tmpfs                                                                          16G   33M   16G   1% /run
tmpfs                                                                          16G     0   16G   0% /sys/fs/cgroup
/dev/sda2                                                                      30G  4.3G   25G  15% /
/dev/sda1                                                                     494M   77M  418M  16% /boot
/dev/sda15                                                                    495M   12M  484M   3% /boot/efi
nfsfiles7nz25whnhti5ox.file.core.windows.net:/nfsfiles7nz25whnhti5ox/nfshome  1.0T  4.8G 1020G   1% /clusterhome
tmpfs                                                                         3.2G     0  3.2G   0% /run/user/0
tmpfs                                                                         3.2G     0  3.2G   0% /run/user/1000
10.107.0.36:/NewVolume                                                        4.0T  256K  4.0T   1% /NewVolume
```
You may need to edit folder permission and add user directories in a shared environment. 
