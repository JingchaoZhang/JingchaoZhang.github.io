http://open.xdmod.org/software-requirements.html

```bash
yum install httpd php php-cli php-mysql php-gd php-mcrypt \
              gmp-devel php-gmp php-pdo php-xml php-pear-Log \
              php-pear-MDB2 php-pear-MDB2-Driver-mysql \
              java-1.7.0-openjdk java-1.7.0-openjdk-devel \
              mariadb-server mariadb cronie logrotate -y
```
```
systemctl start httpd
systemctl enable httpd
systemctl start mariadb.service 
systemctl enable mariadb.service
cp /usr/share/zoneinfo/America/Chicago /etc/localtime
vim /etc/php.ini; date.timezone = America/Chicago
```
```
wget https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2
yum groupinstall "Development Tools" -y
tar jxvf phantomjs-2.1.1-linux-x86_64.tar.bz2
yum install xdmod-6.5.0-1.0.el7.centos.noarch.rpm -y
```
