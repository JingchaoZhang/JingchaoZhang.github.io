---
layout: single
author_profile: false
---

1. yum erase *docker*
1. yum-config-manager --add-repo "https://download.docker.com/linux/centos/docker-ce.repo"
1. yum install docker-ci
1. systemctl stop docker
1. rm -rf /var/lib/docker/*
1. systemctl start docker
1. docker info
