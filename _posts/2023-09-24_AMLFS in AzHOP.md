---
layout: single
author_profile: false
---

Integrating Azure Managed Lustre File System (AMLFS) with Blob storage into AzHOP provides a seamless environment for handling high-performance computing and large-scale data workflows. This blog post outlines a step-by-step guide to achieve this integration, covering aspects such as creating a DataLake v2 supported storage account, establishing vnet peering between the storage account and AzHOP, and configuring AMLFS with Blob integration. We will also touch upon the Network Security Group (NSG) settings and optional blob configurations. Although the current method relies on Azure Portal, future adaptations will focus on automation through tools like Terraform and Ansible. The guide is complemented with illustrations for each step, ensuring clarity and ease of implementation.

**Steps to integrate AMLFS with blob integration to AzHOP**
- Create a storage account with DataLake v2 support
- vnet peering between the storage account and azhop
- Create AMLFS with blob integration
- Update azhop network security group (NSG)
- Add AMLFS to cluster init (optional blob storage mount via blobfuse)

Note: This solution is currently portal based. Many of the following steps can be automated in the future using Terraform and Anisble. 

## Create a new StorageAccount with DL v2
### Create the resource group
![Figure_1](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/1.png) 

In the `Basics` and `Advanced` tabs, follow the setting below. Leave the rest values as default. 

![Figure_2](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/2.png) 

![Figure_3](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/3.png) 

### Create a subnet for AMLFS
Select vnet from the resource group. Select `Subnets` and create a submit following the instructions below:

![Figure_4](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/4.png) 

![Figure_5](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/5.png) 

![Figure_6](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/6.png) 

![Figure_7](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/7.png) 

### Change storage account networking
Change storage account networking from `Allow access from: Selected networks` to `Allow access from: All networks`. 

![Figure_9](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/9.png) 

### Add storage account roles
- **NOTE:** A storage account owner must add these roles `Storage Account Contributor` and `Storage Blob Data Contributor` before creating the file system.   
  1. [Blob integration prerequisites](https://learn.microsoft.com/en-us/azure/azure-managed-lustre/amlfs-prerequisites#blob-integration-prerequisites-optional)
  2. Open your storage account, and select Access control (IAM) in the left navigation pane.
  3. Select Add > Add role assignment to open the Add role assignment page.
  4. Assign the role.
  5. Then add the HPC Cache Resource Provider (search for storagecache) to that role.
  6. Repeat steps 3 and 4 for to add each role.
 
![Figure_14](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/14.png) 

![Figure_15](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/15.png) 

![Figure_16](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/16.png) 

![Figure_17](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/17.png) 

![Figure_18](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/18.png) 

### Add two containers in the storage account
- Create a container named `home`
- Create a container named `log`

![Figure_26](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/26.png) 

![Figure_27](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/27.png) 


## Create vnet peering
Navigate to the vnet in the storage account, click on `Peering` -> `Add`
![Figure_21](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/21.png) 

Fill out the filling as shown below. You can find the Resource ID from your AzHOP vnet page.
![Figure_24](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/24.png) 

![Figure_22](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/22.png) 

![Figure_23](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/23.png) 

## Create AMLFS
Fill out the `Basics` and `Advanced` tabs as below. Leave the rest as default. 

![Figure_25](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/25.png) 

![Figure_28](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/28.png) 

The AMLFS creation could take 10-15 minutes. After creation, you can find the mounting instructions like below:

![Figure_29](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/29.png) 

## Update azhop network security group
In AzHOP, we need to change NSG 3100 rules from deny to allow for both inbound and outbound networks.

![Figure_30](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/30.png) 

![Figure_30](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-10-figures/31.png) 


## Mount AMLFS
### Install AMLFS Client
- **Lustre client software** - Clients must have the appropriate Lustre client package installed. Pre-built client packages have been tested with Azure Managed Lustre. See Install client software for instructions and package download options. Client packages are available for several commonly-used Linux OS distributions. [Client installation](https://learn.microsoft.com/en-us/azure/azure-managed-lustre/client-install?pivots=centos-7)
```bash
# CentOS 7
cat > repo.bash << EOL
#!/bin/bash
set -ex

rpm --import https://packages.microsoft.com/keys/microsoft.asc

DISTRIB_CODENAME=el7

REPO_PATH=/etc/yum.repos.d/amlfs.repo
echo -e "[amlfs]" > ${REPO_PATH}
echo -e "name=Azure Lustre Packages" >> ${REPO_PATH}
echo -e "baseurl=https://packages.microsoft.com/yumrepos/amlfs-${DISTRIB_CODENAME}" >> ${REPO_PATH}
echo -e "enabled=1" >> ${REPO_PATH}
echo -e "gpgcheck=1" >> ${REPO_PATH}
echo -e "gpgkey=https://packages.microsoft.com/keys/microsoft.asc" >> ${REPO_PATH}
EOL

sudo bash repo.bash

sudo yum install amlfs-lustre-client-2.15.1_29_gbae0abe-$(uname -r | sed -e "s/\.$(uname -p)$//" | sed -re 's/[-_]/\./g')-1
```
```bash
# Ubuntu 2004
cat > repo.bash << EOL
#!/bin/bash
set -ex

apt update && apt install -y ca-certificates curl apt-transport-https lsb-release gnupg
source /etc/lsb-release
echo "deb [arch=amd64] https://packages.microsoft.com/repos/amlfs-\${DISTRIB_CODENAME}/ \${DISTRIB_CODENAME} main" | tee /etc/apt/sources.list.d/amlfs.list
curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null

apt update -y
EOL

sudo bash repo.bash

sudo apt install amlfs-lustre-client-2.15.1-29-gbae0abe=$(uname -r)
```
- **Network access to the file system** - Client machines need network connectivity to the subnet that hosts the Azure Managed Lustre file system. If the clients are in a different virtual network, you might need to use VNet peering.
- **Mount** - Clients must be able to use the POSIX mount command to connect to the file system.
- **To achieve advertised performance**
  - Clients must reside in the **same Availability Zone** in which the cluster resides.
  - **Be sure to enable accelerated networking on all client VMs**. If it's not enabled, then fully enabling accelerated networking requires a stop/deallocate of each VM.
- **Security type** - When selecting the security type for the VM, choose the Standard Security Type. Choosing Trusted Launch or Confidential types will prevent the lustre module from being properly installed on the client.


### Edit cluster-init
This is done on the deployer VM as `root`. After adding the file below, terminate the cluster and `./install.sh cccluster` and `./install.sh scheduler`.
```bash
root@deployer:/az-hop# pwd
/az-hop

root@deployer:/az-hop# cat > playbooks/roles/cyclecloud_cluster/projects/common/cluster-init/scripts/91-AMLFS.sh << EOL
#!/bin/bash

# Mount the lustre filesystem
mkdir /AMLFS
mount -t lustre -o noatime,flock 10.42.1.5@tcp:/lustrefs /AMLFS
chmod 777 /AMLFS

# Blob
# Download the Microsoft signing key
wget https://packages.microsoft.com/keys/microsoft.asc

# Convert the Microsoft signing key from armored ASCII format to binary format
gpg --dearmor microsoft.asc

# Create the directory for trusted keys if it does not already exist
mkdir -p /etc/apt/trusted.gpg.d/

# Copy the binary Microsoft signing key to the trusted keys directory
sudo cp microsoft.asc.gpg /etc/apt/trusted.gpg.d/

# Download and install the Microsoft repository configuration for Red Hat Enterprise Linux 7
sudo rpm -Uvh https://packages.microsoft.com/config/rhel/7/packages-microsoft-prod.rpm

# Install Blobfuse and Fuse using the Yum package manager
sudo yum install blobfuse fuse -y

# Open the configuration file 'fuse_connection.cfg' for editing using Vim
cat > fuse_connection.cfg <<EOL
accountName jzs310042023
accountKey 3IQ0aWKsdePJ+kaLNizGqAfomFgu7c2IEwZ2TLjxubxsUNfyu+XqBWfZqajJ41Sd7XFnNxtNc8VR+AStxn2Qhg==
containerName home
EOL

# Create a directory to serve as the mount point for the Blobfuse filesystem
mkdir /BLOB
# Mount the Azure Blob Storage container to the newly created directory
# Set temporary path, configuration file, and timeout options for Blobfuse
blobfuse /BLOB --tmp-path=/tmp/blobfusetmp_$$  --config-file=fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other
EOL
```

### Check /AMLFS /BLOB in the nodes
After the above changess, on a compute node, you will be able to find AMLFS in `/AMLFS` directory, and blob storage in `/BLOB` directory. Note if you also needs those directories on the scheduler node, you can manually run the `91-AMLFS.sh` script as root. You can move the files to `/AMLFS` for computation, and back to `/BLOB` for long term storage.

In conclusion, this blog post offers a comprehensive guide on integrating AMLFS with Blob storage intoAzHOP. The post covers a variety of crucial steps, from establishing a DataLake v2 supported storage account to configuring Network Security Group (NSG) settings. While the current approach utilizes Azure Portal, future efforts will aim to automate these steps using tools like Terraform and Ansible. The illustrative guide aims to facilitate both setup and troubleshooting, making it a valuable resource for anyone seeking to leverage the combined power of Azure's HPC and AI capabilities for large-scale data workflows.
