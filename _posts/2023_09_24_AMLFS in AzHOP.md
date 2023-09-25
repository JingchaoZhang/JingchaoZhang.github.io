---
layout: single
author_profile: false
---


### Resize vnet in AzHOP
- Find hpcvnet -> Subnets -> Compute
- Update `xx.xx.0.128x/25` to `xx.xx.0.128/26`
- Create new subnet named AMLFS. Set range to `xx.xx.0.192/26`

### Create a new StorageAccount with DL v2
- Create new StorageAccount under azhop RG
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

- Create a container named `home`
- Create a container named `log`

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
| **Import Prefix**             | /scracth              |

- 
