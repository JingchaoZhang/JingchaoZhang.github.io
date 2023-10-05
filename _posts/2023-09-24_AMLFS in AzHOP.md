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

### Connect clients to an AMLFS
- **Lustre client software** - Clients must have the appropriate Lustre client package installed. Pre-built client packages have been tested with Azure Managed Lustre. See Install client software for instructions and package download options. Client packages are available for several commonly-used Linux OS distributions. [Client installation](https://learn.microsoft.com/en-us/azure/azure-managed-lustre/client-install?pivots=centos-7)
- **Network access to the file system** - Client machines need network connectivity to the subnet that hosts the Azure Managed Lustre file system. If the clients are in a different virtual network, you might need to use VNet peering.
- **Mount** - Clients must be able to use the POSIX mount command to connect to the file system.
- **To achieve advertised performance**, Clients must reside in the same Availability Zone in which the cluster resides.
- **Be sure to enable accelerated networking on all client VMs**. If it's not enabled, then fully enabling accelerated networking requires a stop/deallocate of each VM.
- **Security type** - When selecting the security type for the VM, choose the Standard Security Type. Choosing Trusted Launch or Confidential types will prevent the lustre module from being properly installed on the client.
