---
layout: single
author_profile: false
---

- To reduce noise, turn off SElinux and firewall.
- Clone the [simplesamlphp-duosecurity](https://github.com/knasty51/simplesamlphp-duosecurity/blob/master/README.md) repo and put it in /usr/share/xdmod/vendor/simplesamlphp/simplesamlphp/modules
```bash
cd /usr/share/xdmod/vendor/simplesamlphp/simplesamlphp/modules
git clone https://github.com/knasty51/simplesamlphp-duosecurity.git
mv simplesamlphp-duosecurity duosecurity
```
- Login Duo web interface as admin. Applications -> +Protect an Application (up-right corner) -> create a new protection
- Edit /etc/xdmod/simplesamlphp/config/config.php. Follow instructions provided by the above link. 'ikey', 'skey' and 'host' can be directly extracted from the Duo Application details. 'akey' is a random 40 characters string that can be generated with Python. [link](https://github.com/Unicon/cas-mfa/issues/87)
```php
'usernameAttribute' => 'uid',  
```  
