---
layout: single
author_profile: false
---

- OpenStack
  - Flavor: general-xlarge
  - Boot from snapshot: "hcc-xdmod prod snap 3/8/18"
  - Security Groups: **Port 8080/tcp**; default
  - Networking: Cluster Interface

- Within VM
```bash
cd /etc/httpd/conf.d
mv xdmod.conf xdmod.conf.20180308
mv xdmod.conf.20170321 xdmod.conf
mv ssl.conf ssl.conf.20180308
mv http-trace-off.conf http-trace-off.conf.20180308
cd /etc/httpd/conf.modules.d
mv 00-ssl.conf 00-ssl.conf.20180308
hostnamectl set-hostname xdmod-upgrade
reboot
```

- Fix Hierarchy
```bash
./create_hierarchy_csv.py -o new-h.csv
xdmod-import-csv -t hierarchy -i new-h.csv
./create_group_to_hierarchy_csv.py -d 0 -o g2h.csv
xdmod-import-csv -t group-to-hierarchy -i g2h.csv
./create_names_csv.py -d 0 -o names.csv
xdmod-import-csv -t names -i names.csv
```

- Fix database
```mysql
use mod_hpcdb;
update hpcdb_jobs set groupname='gladich' where uid_number=4963;
update hpcdb_jobs set gid_number=11378 where uid_number=4963;
update hpcdb_jobs set person_id=3071 where uid_number=4963;
use modw;
update jobfact set group_name='gladich' where uid_number=4963;
update jobfact set gid_number=11378 where uid_number=4963;
update jobfact set person_id=3071 where uid_number=4963;
```

- Reingest data
```bash
xdmod-ingestor --start-date 2016-01-01 --end-date 2018-05-17
```
