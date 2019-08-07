---
layout: single
author_profile: false
---

Only essential commands are listed. Confidential info varies by sites.  

```
yum install openldap.x86_64 openldap-devel.x86_64 -y
yum install sssd.x86_64 sssd-ldap.x86_64 -y
touch /etc/sssd/sssd.conf
chown -R root:root /etc/sssd
chmod 700 /etc/sssd/
chmod 600 /etc/sssd/sssd.conf
authconfig  --useshadow  --enablemd5 --enableldap --enableldapauth --enablelocauth --ldapserver= --ldapbasedn= --update
systemctl start sssd.service
systemctl enable sssd.service
#Install LDAP certs and enable TLS 
cat >/etc/openldap/cacerts/1.crt <<EOM
-----BEGIN CERTIFICATE-----
-----END CERTIFICATE-----
EOM

cat >/etc/openldap/cacerts/2.crt <<EOM
-----BEGIN CERTIFICATE-----
-----END CERTIFICATE-----
EOM

cat >/etc/openldap/cacerts/3.crt <<EOM
-----BEGIN CERTIFICATE-----
-----END CERTIFICATE-----
EOM

cacertdir_rehash /etc/openldap/cacerts
authconfig --enableldaptls --update

#change /etc/ssh/sshd_config
vim /etc/ssh/sshd_config
PasswordAuthentication yes
ChallengeResponseAuthentication yes
AllowUsers jingchao

#Restart service
systemctl restart sssd
```

[Link](https://hcc-docs.unl.edu/pages/viewpage.action?spaceKey=ADMIN&title=OpenConnect+VPN)

Make sure sssd configuration is correct.
/etc/sssd/sssd.conf
