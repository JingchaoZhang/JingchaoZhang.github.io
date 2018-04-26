---
layout: single
author_profile: false
---

[Boost](http://www.boost.org/)
```bash
$ ./bootstrap.sh --prefix=/home/username/usr
$ vim boost_1_55_0/tools/build/v2/user-config.jam
using mpi ；#add to last time
$ ./b2 install --prefix=/home/username/usr | tee install.log
```
