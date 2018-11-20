---
layout: single
author_profile: false
---

1. From Tusker/Crane, ssh beta.anvil.hcc.unl.edu
1. sudo su -
1. source /root/keystonerc_admin
1. /util/accounts-mgmt-cli/anvilman.py -v -g GROUPNAME -u USERNAME
1. keystone tenant-get GROUPNAME #Check if a tenant(group) exists
1. keystone user-list --tenant GROUPNAME #List users in a tenant
