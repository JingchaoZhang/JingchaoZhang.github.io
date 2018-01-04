---
layout: single
author_profile: false
---

get the ACL’s for /home/usera
```bash
[usera@login-damiana ~]# lfs lgetfacl  /home/usera
# file: home/usera
# owner: usera
# group: users
user::rwx
group::---
other::---
```

set new (read+access) ACLs for user:apache on /home/usera
```bash
[usera@login-damiana ~]# lfs lsetfacl -m user:apache:rx /home/usera
[usera@login-damiana ~]# lfs lgetfacl  /home/usera
# file: home/usera
# owner: usera
# group: users
user::rwx
user:apache:r-x
group::---
mask::r-x
other::---
```
