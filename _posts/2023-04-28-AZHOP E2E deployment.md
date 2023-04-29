---
layout: single
author_profile: false
---

Azure HPC On-Demand Platform (az-hop) is a tool that provides an end-to-end deployment mechanism for a base HPC infrastructure on Azure. It uses industry standard tools like Terraform, Ansible and Packer to provision and configure a complete HPC cluster solution that is ready for users to run applications. It also includes features such as an HPC OnDemand Portal, an Active Directory, a Job Scheduler, dynamic resources provisioning and autoscaling, a Jumpbox, and various storage options [ref](https://azure.github.io/az-hop/).

Clone the repo
```bash
git clone --recursive https://github.com/Azure/az-hop.git
```

Create the `config.yml` file
```bash
---
project_name: az-hop
location: eastus
resource_group: JZ-azhop_v2
use_existing_rg: false

tags:
  env: dev
  project: azhop

log_analytics:
  create: false

monitoring:
  install_agent: false

alerting:
  enabled: false
  admin_email: email@email.com
  local_volume_threshold: 80

anf:
  create: false
  homefs_size_tb: 4
  homefs_service_level: Standard
  dual_protocol: false # true to enable SMB support. false by default
  alert_threshold: 80 # alert when ANF volume reaches this threshold

azurefiles:
  create: true
  size_gb: 1024

mounts:
  home: # This home name can't be changed
    type: azurefiles # anf or azurefiles, default to anf. One of the two should be defined in order to mount the home directory
    mountpoint: /anfhome # /sharedhome for example
    server: '{{anf_home_ip}}' # Specify an existing NFS server name or IP, when using the ANF built in use '{{anf_home_ip}}'
    export: '{{anf_home_path}}' # Specify an existing NFS export directory, when using the ANF built in use '{{anf_home_path}}'
    options: 'vers=4,minorversion=1,sec=sys' #'{{anf_home_opts}}' # Specify the mount options. Default to rw,hard,rsize=262144,wsize=262144,vers=3,tcp,_netdev

admin_user: hpcadmin

network:
  create_nsg: true
  vnet:
    name: hpcvnet # Optional - default to hpcvnet
    address_space: "10.101.0.0/23"
    subnets: # all subnets are optionals
      frontend:
        name: frontend
        address_prefixes: "10.101.0.0/29"
        create: true # create the subnet if true. default to true when not specified, default to false if using an existing VNET when not specified
      admin:
        name: admin
        address_prefixes: "10.101.0.16/28"
        create: true
      ad:
        name: ad
        address_prefixes: "10.101.0.8/29"
        create: true
      netapp:
        name: netapp
        address_prefixes: "10.101.0.32/28"
        create: true
      compute:
        name: compute
        address_prefixes: "10.101.1.0/24"
        create: true

locked_down_network:
  enforce: false
  public_ip: true # Enable public IP creation for Jumpbox, OnDemand and create images. Default to true

linux_base_image: "OpenLogic:CentOS:7_9-gen2:latest"
windows_base_image: "MicrosoftWindowsServer:WindowsServer:2019-Datacenter-smalldisk:latest" # publisher:offer:sku:version or image_id

deployer:
  vm_size: Standard_B2ms
ad:
  vm_size: Standard_B2ms
ondemand:
  vm_size: Standard_D4s_v5
  generate_certificate: true # Generate an SSL certificate for the OnDemand portal. Default to true
grafana:
  vm_size: Standard_B2ms
guacamole:
  vm_size: Standard_B2ms
scheduler:
  vm_size: Standard_B2ms
cyclecloud:
  vm_size: Standard_B2ms

users:
  - { name: hpcuser,   uid: 10001 }
  - { name: adminuser, uid: 10002, groups: [5001, 5002] }
  - { name: john.john,   uid: 10003 }

usergroups:
  - name: Domain Users # All users will be added to this one by default
    gid: 5000
  - name: az-hop-admins
    gid: 5001
    description: "For users with azhop admin privileges"
  - name: az-hop-localadmins
    gid: 5002
    description: "For users with sudo right or local admin right on nodes"

cvmfs_eessi:
  enabled: false

queue_manager: slurm

slurm:
  accounting_enabled: false
  slurm_version: 20.11.9

enroot:
  enroot_version: 3.4.1

database:
  user: sqladmin

bastion:
  create: false

vpn_gateway:
  create: false

authentication:
  httpd_auth: basic # oidc or basic

autoscale:
  idle_timeout: 1800 # Idle time in seconds before shutting down VMs - default to 1800 like in CycleCloud

queues:
  - name: execute
    vm_size: Standard_F2s_v2
    max_core_count: 20
    image: azhpc:azhop-compute:ubuntu-2004:latest
    spot: false
    ColocateNodes: false

enable_remote_winviz: false # Set to true to enable windows remote visualization

remoteviz:
  - name: winviz # This name is fixed and can't be changed
    vm_size: Standard_NV12s_v3 # Standard_NV8as_v4 Only NVsv3 and NVsV4 are supported
    max_core_count: 48
    image: "MicrosoftWindowsDesktop:Windows-10:21h1-pron:latest"
    ColocateNodes: false
    spot: false
    EnableAcceleratedNetworking: false

applications:
  bc_codeserver:
    enabled: true
  bc_jupyter:
    enabled: true
  bc_amlsdk:
    enabled: false
  bc_rstudio:
    enabled: true
  bc_ansys_workbench:
    enabled: false
  bc_vmd:
    enabled: false
  bc_paraview:
    enabled: false
  bc_vizer:
    enabled: false
```

Install dependences
```bash
sudo ./toolset/scripts/install.sh
```

Build the backbone using bicep
```bash
$ ./build.sh -a apply -l bicep
```

Find the deployer VM ip from the Azure portal. Connect to the deployer VM
```bash
$ ssh -i hpcadmin_id_rsa hpcadmin@20.231.50.26
The authenticity of host '20.231.50.26 (20.231.50.26)' can't be established.
ECDSA key fingerprint is SHA256:lPf4I4nZmZ7hzuxif9RZOVdMmGC6zMvSylTE79Tapwk.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added '20.231.50.26' (ECDSA) to the list of known hosts.
Welcome to Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-1036-azure x86_64)
```

Monitor the ansible installation
```bash
hpcadmin@deployer:~$ sudo -i
root@deployer:~# cd /var/log/
root@deployer:/var/log# ls
apt       azure  chrony                 cloud-init.log  dmesg     journal   landscape  private  ubuntu-advantage.log  waagent.log
auth.log  btmp   cloud-init-output.log  dist-upgrade    dpkg.log  kern.log  lastlog    syslog   unattended-upgrades   wtmp
root@deployer:/var/log# tail -f cloud-init-output.log
```

The end of a successful deployment:
```bash
PLAY RECAP *********************************************************************
ccportal                   : ok=3    changed=2    unreachable=0    failed=0    skipped=1    rescued=0    ignored=0
grafana                    : ok=3    changed=2    unreachable=0    failed=0    skipped=1    rescued=0    ignored=0
ondemand                   : ok=3    changed=2    unreachable=0    failed=0    skipped=1    rescued=0    ignored=0
scheduler                  : ok=3    changed=2    unreachable=0    failed=0    skipped=1    rescued=0    ignored=0

Saturday 29 April 2023  03:58:01 +0000 (0:00:01.488)       0:00:03.239 ********
===============================================================================
chrony ------------------------------------------------------------------ 3.06s
include_role ------------------------------------------------------------ 0.11s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
total ------------------------------------------------------------------- 3.17s
Command succeeded!
Cloud-init v. 23.1.1-0ubuntu0~20.04.1 running 'modules:final' at Sat, 29 Apr 2023 03:16:47 +0000. Up 27.90 seconds.
Cloud-init v. 23.1.1-0ubuntu0~20.04.1 finished at Sat, 29 Apr 2023 03:58:01 +0000. Datasource DataSourceAzure [seed=/dev/sr0].  Up 2501.89 seconds
```

Get the FQDN, and username/password:
```bash
root@deployer:/az-hop# cd /az-hop/
root@deployer:/az-hop# pwd
/az-hop
root@deployer:/az-hop# grep ondemand_fqdn playbooks/group_vars/all.yml
ondemand_fqdn: ondemandmxsmmtrkr6ehsx.eastus.cloudapp.azure.com
root@deployer:/az-hop# ./bin/get_secret john.john
j5hTzIyBXqVExNpB35pJGVra5sM=
```

