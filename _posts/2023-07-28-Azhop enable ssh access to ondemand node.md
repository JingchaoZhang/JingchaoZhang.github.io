---
layout: single
author_profile: false
---

To enable ssh to the ondemand node, you need to  
1. Change the NSG rule to allow inbound traffic on port 22; 
1. Edit `/etc/ssh/sshd_config` file on the ondemand node, change `PasswordAuthentication` to yes, then restart sshd. 

Here are the details:
1. Find 'network settings' in the ondemand VM resource page. Click on it.  
   ![Figure_1](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-28-figures/1.png)
3. On the next page, click on ![Figure_2](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-28-figures/2.png), and select Inbound port rule. 
4. Add a NSG rule similar to this one. Feel free to limit 'Source' and 'port ranges' as needed.  
   ![Figure_3](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-07-28-figures/3.png)
6. Click 'Add' to save the changes. Please note the NSG change may take a few minutes to become effective. 
7. From the ondemand node
```bash
[hpcadmin@ondemand ~]$ sudo vim /etc/ssh/sshd_config
PasswordAuthentication yes
[hpcadmin@ondemand ~]$ sudo systemctl restart sshd
```

This should allow you to ssh into the ondemand node as any user.
