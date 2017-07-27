---
layout: single
author_profile: false
---

CentOS 7 mysql server
```bash
yum install vim -y
yum group install "Development Tools" -y
yum install mariadb-server mariadb -y
systemctl start mariadb.service
systemctl enable mariadb.service
systemctl status mariadb.service
setenforce 0
```

mysql
```mysql
GRANT ALL PRIVILEGES ON *.* TO 'root'@'10.138.17.21' WITH GRANT OPTION;
flush privileges;
```


