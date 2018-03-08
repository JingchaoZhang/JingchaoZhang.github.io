---
layout: single
author_profile: false
---

- OpenStack
  - Flavor: general-xlarge
  - Boot from snapshot: hcc0xdmod prod snap 3/8/18
  - Security Groups: Port 8080/tcp; default
  - Networking: Cluster Interface

- Within VM
```bash
cd /etc/httpd/conf.d
mv xdmod.conf xdmod.conf.20180308
mv ssl.conf ssl.conf.20180308
mv http-trace-off.conf http-trace-off.conf.20180308
cd /etc/httpd/conf.modules.d
mv 00-ssl.conf 00-ssl.conf.20180308
hostnamectl set-hostname xdmod-upgrade
reboot
```
