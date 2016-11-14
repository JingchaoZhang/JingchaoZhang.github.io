---
layout: single
author_profile: false
---

- To reduce noise, turn off SElinux and firewall.
- Clone the [simplesamlphp-duosecurity](https://github.com/knasty51/simplesamlphp-duosecurity/blob/master/README.md) repo and put it in /usr/share/xdmod/vendor/simplesamlphp/simplesamlphp/modules
- Login Duo web interface as admin. Applications -> Protect an Application -> Web SDK -> Protect this Application
- Edit /etc/xdmod/simplesamlphp/config/config.php. Follow instructions provided by the above link. Extra caution with this line  
```
'usernameAttribute' => 'uid',  
```  
