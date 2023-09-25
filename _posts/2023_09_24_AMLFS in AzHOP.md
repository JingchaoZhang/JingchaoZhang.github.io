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
  5. Then add the HPC Cache Resource Provider to that role.
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

- 
