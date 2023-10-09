---
layout: single
author_profile: false
---


### Resize vnet in AzHOP
- Find hpcvnet -> Subnets -> Compute
- Update `xx.xx.0.128x/25` to `xx.xx.0.128/26`
- Create new subnet named AMLFS. Set range to `xx.xx.0.192/26`

### Create a new StorageAccount with DL v2
- Create new StorageAccount under AzHOP RG
- Set Data Lake Storage v2 with following properties:

| **Data Lake Storage **  | **Status**              |
|-------------------------|----------------------|
| **Hierarchical namespace**  | Enabled              |
| **Default access tier**     | Hot                  |
| **Blob anonymous access**   | Disabled             |
| **Blob soft delete**        | Enabled (7 days)     |
| **Container soft delete**   | Disabled             |
| **Versioning**              | Disabled             |
| **Change feed**             | Disabled             |
| **NFS v3**                  | Enabled              |
| **SFTP**                    | Enabled              |

- For Virtual network, select `hpcvnet` from AzHOP RG
- For Subnets, select `netapp`
- Create the storage account
- Go to resource
- Change `Allow access from: Selected networks` to `Allow access from: All networks`
- Create a container named `home`
- Create a container named `log`
- **NOTE:** A storage account owner must add these roles `Storage Account Contributor` and `Storage Blob Data Contributor` before creating the file system.   
  1. [Blob integration prerequisites](https://learn.microsoft.com/en-us/azure/azure-managed-lustre/amlfs-prerequisites#blob-integration-prerequisites-optional)
  2. Open your storage account, and select Access control (IAM) in the left navigation pane.
  3. Select Add > Add role assignment to open the Add role assignment page.
  4. Assign the role.
  5. Then add the HPC Cache Resource Provider (search for storagecache) to that role.
  6. Repeat steps 3 and 4 for to add each role.
- [Subnet access and permissions](https://learn.microsoft.com/en-us/azure/azure-managed-lustre/amlfs-prerequisites#subnet-access-and-permissions)
  - Use the default Azure-based DNS server.
  - Add an **outbound security rule** with the following properties:
    - Port: Any
    - Protocol: Any
    - Source: Virtual Network
    - Destination: "AzureCloud" service tag
    - Action: Allow
  - Your network security group must allow inbound and outbound access on **port 988 and ports 1019-1023**.
### Create AMLFS
- Storage capacity: `4 TB * 500 MB/s/TB = 2000 MB/s`
- For Virtual Network, select `hpcvnet` from AzHOP RG
- For Subnet, select AMLFS created in earlier step
- Select `Import/export data from blob`

| **Advanced**                  | **Status**            |
|-------------------------------|-----------------------|
| **Blob integration**          | Configured           |
| **Storage account**           | jzamlfs0924           |
| **Container**                 | home                  |
| **Logging container**         | log                   |
| **Import Prefix**             |                       |

### Create vnet peering
- This virtual network
  - Peering link name
    - amlfs
  - Peering status
    - Fully Synchronized
  - Peering state
    - Succeeded
  - Options
    - Allow 'jzs310042023_vNet7501856' to access 'hpcvnet'
    - Allow 'jzs310042023_vNet7501856' to receive forwarded traffic from 'hpcvnet'

- Remote virtual network
  - Remote Vnet Id
    - /subscriptions/f5a67d06-2d09-4090-91cc-e3298907a021/resourceGroups/JZ-azhopnoaa2/providers/Microsoft.Network/virtualNetworks/hpcvnet
  - Address space
    - 10.131.0.0/24

- This virtual network
  - Peering link name
    - amlfs
  - Peering status
    - Fully Synchronized
  - Peering state
    - Succeeded
  - Options
    - Allow 'hpcvnet' to access 'jzs310042023_vNet7501856'
    - Allow 'hpcvnet' to receive forwarded traffic from 'jzs310042023_vNet7501856'

- Remote virtual network
  - Remote Vnet Id
    - /subscriptions/f5a67d06-2d09-4090-91cc-e3298907a021/resourceGroups/JZ-s3/providers/Microsoft.Network/virtualNetworks/jzs310042023_vNet7501856
  - Address space
    - 10.42.0.0/16

### Connect clients to an AMLFS
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

### Mount AMLFS
```bash
sudo mount -t lustre -o noatime,flock 10.42.1.5@tcp:/lustrefs /AMLFS
# To unmount
sudo umount /AMLFS
```

### Mount the blob storage using blobfuse
```bash
#!/bin/bash

set -ex

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
accountName <your-azure-storage-account-name>
accountKey <your-azure-storage-account-key>
containerName <your-container-name>
EOL

# Create a directory to serve as the mount point for the Blobfuse filesystem
mkdir /myblob

# Mount the Azure Blob Storage container to the newly created directory
# Set temporary path, configuration file, and timeout options for Blobfuse
blobfuse /myblob --tmp-path=/tmp/blobfusetmp  --config-file=fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
```

### Edit cluster-init
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

### Transfer file from blob to AMLFS

```bash
cp /BLOB/image/blog.png /AMLFS/
```

