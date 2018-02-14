---
layout: single
author_profile: false
---

```bash
yum install epel-release -y
yum install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm -y
yum groupinstall "Development Tools" -y
yum install vim wget httpd php php-cli php-mysql php-gd php-mcrypt \
              gmp-devel php-gmp php-pdo php-xml php-pear-Log \
              php-pear-MDB2 php-pear-MDB2-Driver-mysql \
              java-1.7.0-openjdk java-1.7.0-openjdk-devel \
              mariadb-server mariadb cronie logrotate -y
yum install php-pear-MDB2 php-pear-Log php-mcrypt php-pear-MDB2-Driver-mysql -y
```
```bash
systemctl start httpd
systemctl enable httpd
systemctl start mariadb.service 
systemctl enable mariadb.service
cp /usr/share/zoneinfo/America/Chicago /etc/localtime
vim /etc/php.ini; date.timezone = America/Chicago
vim /etc/sysconfig/selinux
    SELINUX=disabled
reboot
```
```bash
wget https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2
tar jxvf phantomjs-2.1.1-linux-x86_64.tar.bz2
yum install xdmod-6.5.0-1.0.el7.centos.noarch.rpm -y
```

SimpleSAML setup. [Federated Authentication](http://open.xdmod.org/simpleSAMLphp.html) and [LDAP Authentication](http://open.xdmod.org/simpleSAMLphp-ldap.html)
```bash
yum install php-ldap.x86_64 -y
vim /etc/httpd/conf.d/xdmod.conf
#uncomment below
    # SimpleSAML federated authentication.
    SetEnv SIMPLESAMLPHP_CONFIG_DIR /etc/xdmod/simplesamlphp/config
    Alias /simplesaml /usr/share/xdmod/vendor/simplesamlphp/simplesamlphp/www
    <Directory /usr/share/xdmod/vendor/simplesamlphp/simplesamlphp/www>
        Options FollowSymLinks
        AllowOverride All
        # Apache 2.4 access controls.
        <IfModule mod_authz_core.c>
            Require all granted
        </IfModule>
    </Directory>
```

[link](http://open.xdmod.org/software-requirements.html)
