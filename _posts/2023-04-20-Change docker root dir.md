---
layout: single
author_profile: false
---

System information
```bash
# cat /etc/*release
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=18.04
DISTRIB_CODENAME=bionic
DISTRIB_DESCRIPTION="Ubuntu 18.04.6 LTS"
NAME="Ubuntu"
VERSION="18.04.6 LTS (Bionic Beaver)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 18.04.6 LTS"
VERSION_ID="18.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=bionic
UBUNTU_CODENAME=bionic
```

Disk volume
```bash
# df -h
Filesystem                              Size  Used Avail Use% Mounted on
udev                                    886G     0  886G   0% /dev
tmpfs                                   178G  1.5M  178G   1% /run
/dev/sda1                                29G   20G  9.1G  69% /
tmpfs                                   886G     0  886G   0% /dev/shm
tmpfs                                   5.0M     0  5.0M   0% /run/lock
tmpfs                                   886G     0  886G   0% /sys/fs/cgroup
/dev/sda15                              105M  5.2M  100M   5% /boot/efi
/dev/sdb1                               2.8T  620K  2.7T   1% /mnt
10.213.0.36:home-ozp60y9a/slurm/config  4.0T   19M  4.0T   1% /sched
10.213.0.36:/home-ozp60y9a              4.0T   19M  4.0T   1% /anfhome
/dev/md128                              7.0T  7.2G  7.0T   1% /mnt/nvme
tmpfs                                   178G     0  178G   0% /run/user/0
```

Docker version
```bash
# docker --version
Docker version 20.10.23+azure-2, build 715524332ff91d0f9ec5ab2ec95f051456ed1dba
```

Error when pulling images
```bash
# docker pull ghcr.io/deepmodeling/deepmd-kit:2.1.1_cuda11.6_gpu
2.1.1_cuda11.6_gpu: Pulling from deepmodeling/deepmd-kit
d5fd17ec1767: Pull complete
68803f8d6f15: Pull complete
d21588509682: Pull complete
20f9413a4963: Pull complete
7332e99e138c: Pull complete
93ce31275c05: Pull complete
5316c7231986: Pull complete
559d8c2b6130: Extracting [==================================================>]  3.296GB/3.296GB
e0efa6db707e: Download complete
ee671fbcee18: Download complete
failed to register layer: ApplyLayer exit status 1 stdout:  stderr: write /opt/deepmd-kit/lib/python3.10/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so: no space left on device
```

This is because the `Docker Root Dir` is set to `/var/lib/docker`
```bash
# docker info | grep "Docker Root Dir"
WARNING: No swap limit support
 Docker Root Dir: /var/lib/docker
```

To change `Docker Root Dir`, we need to edit file `/etc/docker/daemon.json`.
Change 
```bash
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
into
```bash
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "data-root": "/anfhome/root"
}
```

Now restart docker daemon
```bash
systemctl daemon-reload
systemctl restart docker
```

Verify `Docker Root Dir` has changed
```bash
# docker info | grep "Docker Root Dir"
WARNING: No swap limit support
WARNING: the aufs storage-driver is deprecated, and will be removed in a future release.
 Docker Root Dir: /anfhome/root
```

Now let's try `docker pull` again
```bash
# docker pull ghcr.io/deepmodeling/deepmd-kit:2.1.1_cuda11.6_gpu
2.1.1_cuda11.6_gpu: Pulling from deepmodeling/deepmd-kit
d5fd17ec1767: Pull complete
68803f8d6f15: Pull complete
d21588509682: Pull complete
20f9413a4963: Pull complete
7332e99e138c: Pull complete
93ce31275c05: Pull complete
5316c7231986: Pull complete
559d8c2b6130: Pull complete
e0efa6db707e: Pull complete
ee671fbcee18: Pull complete
Digest: sha256:50e065a9ddca9271c70096150eac5d27af91b8fe4d6154507d0a3afd637547c2
Status: Downloaded newer image for ghcr.io/deepmodeling/deepmd-kit:2.1.1_cuda11.6_gpu
ghcr.io/deepmodeling/deepmd-kit:2.1.1_cuda11.6_gpu
root@a100-pg0-1:/anfhome/root# pwd
/anfhome/root
root@a100-pg0-1:/anfhome/root# ls
aufs  buildkit  containers  image  network  plugins  runtimes  swarm  tmp  trust  volumes
# docker image list
REPOSITORY                        TAG                  IMAGE ID       CREATED         SIZE
ghcr.io/deepmodeling/deepmd-kit   2.1.1_cuda11.6_gpu   80e9ba4a6a0c   11 months ago   7.2GB
```

1. Reference [[1]](https://stackoverflow.com/questions/24309526/how-to-change-the-docker-image-installation-directory)
1. Reference [[2]](https://blog.adriel.co.nz/2018/01/25/change-docker-data-directory-in-debian-jessie/)








