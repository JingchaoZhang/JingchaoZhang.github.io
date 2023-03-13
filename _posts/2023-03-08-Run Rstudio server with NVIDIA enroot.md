---
layout: single
author_profile: false
---

This is to explain how to run Rstudio Server container image using NVIDIA Enroot/Pyxis. You have the option to deploy the container using Docker. Here is a nice [blog](https://davetang.org/muse/2021/04/24/running-rstudio-server-with-docker/) explaining the steps. 

The container image is from [rocker/rstudio](https://hub.docker.com/r/rocker/rstudio/tags). Below is the docker recipe for [rstudio_4.2.2](https://github.com/rocker-org/rocker-versioned2/blob/master/dockerfiles/rstudio_4.2.2.Dockerfile).
```bash
FROM rocker/r-ver:4.2.2

LABEL org.opencontainers.image.licenses="GPL-2.0-or-later" \
      org.opencontainers.image.source="https://github.com/rocker-org/rocker-versioned2" \
      org.opencontainers.image.vendor="Rocker Project" \
      org.opencontainers.image.authors="Carl Boettiger <cboettig@ropensci.org>"

ENV S6_VERSION=v2.1.0.2
ENV RSTUDIO_VERSION=2022.12.0+353
ENV DEFAULT_USER=rstudio
ENV PANDOC_VERSION=default
ENV QUARTO_VERSION=default

RUN /rocker_scripts/install_rstudio.sh
RUN /rocker_scripts/install_pandoc.sh
RUN /rocker_scripts/install_quarto.sh

EXPOSE 8787

CMD ["/init"]
```
If everything works out, user just need to start the container without any arguments to initiate the Rstudio server. The `CMD ["/init"]` command will bootstrap the server process, as described [here](https://davetang.org/muse/2021/04/24/running-rstudio-server-with-docker/).  

This is what happens when you try to start the server with enroot: 
```bash
enroot import docker://rocker/rstudio:4.2.2
enroot create --name rstudio rocker+rstudio+4.2.2.sqsh
enroot start --root --rw --env USER=rstudio --env PASSWORD=password rstudio
```
```bash
$ enroot start --root --rw --env USER=rstudio --env PASSWORD=password rstudio
[s6-init] making user provided files available at /var/run/s6/etc...exited 0.
[s6-init] ensuring user provided files have correct perms...exited 0.
[fix-attrs.d] applying ownership & permissions fixes...
[fix-attrs.d] done.
[cont-init.d] executing container initialization scripts...
[cont-init.d] 01_set_env: executing...
skipping /var/run/s6/container_environment/HOME
skipping /var/run/s6/container_environment/PASSWORD
skipping /var/run/s6/container_environment/RSTUDIO_VERSION
[cont-init.d] 01_set_env: exited 0.
[cont-init.d] 02_userconf: executing... 
[cont-init.d] 02_userconf: exited 0.
[cont-init.d] done.
[services.d] starting services
[services.d] done.
TTY detected. Printing informational message about logging configuration. Logging configuration loaded from '/etc/rstudio/logging.conf'. Logging to 'syslog'.
2023-03-01T01:46:01.071242Z [rserver] ERROR system error 13 (Permission denied); OCCURRED AT rstudio::core::Error rstudio::core::system::posix::temporarilyDropPrivileges(const rstudio::core::system::User&, const rstudio_boost::optional<unsigned int>&) src/cpp/shared_core/system/PosixSystem.cpp:286; LOGGED FROM: int main(int, char* const*) src/cpp/server/ServerMain.cpp:879
/usr/lib/rstudio-server/bin/rstudio-server: line 7: systemctl: command not found
```

The reason for `ERROR system error 13` in this case is because Rstudio server is designed for multiple users and it needs root access so it can process each users' login and load their data. To run a single user session, modify the `/etc/rstudio/rserver.conf` file within the container. Link to [running without permissions](https://docs.posit.co/ide/server-pro/access_and_security/server_permissions.html#running-without-permissions). 

Start an interactive session
```bash
enroot start --root --rw rstudio /bin/bash
```
Edit the .conf file
```bash
# cat /etc/rstudio/rserver.conf
rsession-which-r=/usr/local/bin/R
# add the following two lines
server-user=$USER
auth-none=1 # Set to 1 to turn off user auth. 
```
Within the container, you can test to start the server after the above changes.
```bash
# /etc/init.d/rstudio-server start 
TTY detected. Printing informational message about logging configuration. Logging configuration loaded from '/etc/rstudio/logging.conf'. Logging to 'syslog'.
# ps aux | grep rstudio
root     14193  0.1  0.1 100584  6200 ?        Ssl  04:17   0:01 /usr/lib/rstudio-server/bin/rserver
root     17441  0.0  0.0   6460   960 ?        S+   04:37   0:00 grep rstudio
```

Rstudio has started and running in the background. 

Since user auth is turned off, we no logner need to pass username and password, and Rstudio can be started with the following command.
```bash
enroot import docker://rocker/rstudio:4.2.2
enroot create --name rstudio rocker+rstudio+4.2.2.sqsh
enroot start rstudio /usr/lib/rstudio-server/bin/rserver --www-port 61003
```
NOTE: The reason for starting the server using `/usr/lib/rstudio-server/bin/rserver --www-port 61003` is because enroot does not have port forwarding, as explained [here](https://github.com/NVIDIA/enroot/issues/16). If port 8787 is enabled on your server, you can just do `enroot start rstudio`. 
