---
layout: single
author_profile: false
---

In the rapidly evolving landscape of High-Performance Computing (HPC) and Artificial Intelligence (AI), the quest for optimizing operational cost without compromising performance has become paramount. Microsoft Azure's HPC On-Demand Platform (AzHOP) serves as an innovative solution that addresses both scale and flexibility needs. However, one area that often warrants scrutiny is the daily cost associated with the backbone infrastructure of AzHOP, which includes critical components such as Management VMs, persistent storage volumes, and more.

We will compare the daily backbone costs associated with different AzHOP configurations. Specifically, we will look at setups with SLURM DB and Azure Active Directory (AAD) enabled. We will explore three different storage options to examine how each impacts the overall cost and performance: 
- 4TB Azure Files
- 4TB Premium Azure NetApp Files
- Azure Managed Lustre File System (AMLFS)  

The objective is to arm decision-makers and technical experts with concrete insights that can guide them in selecting the most cost-effective yet performant backbone infrastructure for their Azure HPC deployments.

## Azure Files (4TB)
The experiment was conducted over the period from September 2nd to September 4th. Cost data for both the starting day, September 2nd, and the concluding day, September 4th, are partial and therefore lower than the figures from September 3rd. In contrast, the data for September 3rd represents a complete 24-hour cycle.   
  
![Figure_1](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-09-02-figures/AF-daily.png)  
  
Let's break down the cost for 09/03 in the table below:  
  
| Date     | Service Name                 | Cost                |
|----------|------------------------------|---------------------|
| Sep 03   | Virtual Machines            | 9.960719999999998  |
| Sep 03   | Storage                      | 9.644774781549     |
| Sep 03   | Azure Database for MariaDB   | 2.989564258064516  |
| Sep 03   | Virtual Network              | 0.12240000000000001|
| Sep 03   | Azure DNS                    | 0.006294902258064519|
| Sep 03   | Bandwidth                    | 0.00022860545162111526|
| Sep 03   | Advanced Threat Protection   | 0.0000011999999999999997|

The primary cost components for running AzHOP include `Virtual Machines` and `Azure Files`. Specifically, this experiment allocates 4TB for Azure Files. However, this size can be scaled down to 1TB, depending on your storage requirements. An additional cost is associated with `Azure Database for MariaDB`, which serves as the database backend for SLURM accounting. If SLURM accounting is not a critical feature for your specific use case, you may opt to disable it to further reduce costs. By minimizing the Azure Files storage to 1TB and foregoing MariaDB, the estimated minimal daily expenditure stands at approximately **$12.5/day**.

## Azure Netapp Files (4TB Premium)
Analogous to the previous experiment, the cost data for both September 2nd and September 4th are partial and not representative of a full 24-hour cycle. In contrast, the data from September 3rd is complete and spans an entire 24-hour period.  
  
![Figure_3](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-09-02-figures/ANF-daily.png)  
  
Here is a table breakdown for 09/03:  
  
| Date     | Service Name               | Cost             |
|----------|----------------------------|------------------|
| Sep 03   | Azure NetApp Files         | 13.469614080000001 |
| Sep 03   | Virtual Machines           | 9.953625947537999 |
| Sep 03   | Azure Database for MariaDB | 2.989564258064516 |
| Sep 03   | Storage                    | 0.20762154493899998 |
| Sep 03   | Virtual Network            | 0.12240000000000001 |
| Sep 03   | Azure DNS                  | 0.006294712258064519 |
| Sep 03   | Bandwidth                  | 0.00023048860579729093 |
| Sep 03   | Advanced Threat Protection | 6e-7 |

With Azure NetApp Files (ANF), the smallest allowable volume size is 4TB, translating to an estimated daily cost of approximately $13.5. If you opt to run your setup without MariaDB, the projected cost increases to **$25/day**. This is roughly double the expense when compared to utilizing Azure Files (1TB without MariaDB).  

## AMLFS
NOTE: If you want to use integrated Azure Blob storage with AMLFS, you must specify it in the Blob integration section when you create the file system. You can't add an HSM-integrated blob container to an existing file system. Integrating blob storage when you create a file system is optional, but it's the only way to use Lustre Hierarchical Storage Management (HSM) features. If you don't want the benefits of Lustre HSM, you can import and export data for the Azure Managed Lustre file system by using client commands directly.

### Without Blob integration
Setup  

| Attribute                | Value                           |
|--------------------------|---------------------------------|
| Subscription             | XXXX          |
| Resource group           | JZ-AMLFS                        |
| Region                   | South Central US                |
| Availability zone        | 1                               |
| File system name         | lustre                          |
| Storage capacity         | 8 TiB                           |
| Throughput per TiB       | 250 MB/s                        |
| Total Throughput         | 2000 MB/s                       |
| Virtual network          | (New) lustre-vnet               |
| Subnet                   | (New) default (10.4.0.0/27)      |
| Maintenance window       | Sunday, 12:00                   |

### With Blob integration










## Details on AMLFS
### Determining network size
The size of subnet that you need depends on the size of the file system you create. The following table gives a rough estimate of the minimum subnet size for Azure Managed Lustre file systems of different sizes.
  
| Storage capacity     | Recommended CIDR prefix value |
|----------------------|-------------------------------|
| 4 TiB to 16 TiB      | /27 or larger                 |
| 20 TiB to 40 TiB     | /26 or larger                 |
| 44 TiB to 92 TiB     | /25 or larger                 |
| 96 TiB to 196 TiB    | /24 or larger                 |
| 200 TiB to 400 TiB   | /23 or larger                 |

- [Install pre-built client software](https://learn.microsoft.com/en-us/azure/azure-managed-lustre/client-install?source=recommendations&pivots=centos-7)
- [Connect clients to an Azure Managed Lustre file system](https://learn.microsoft.com/en-us/azure/azure-managed-lustre/connect-clients)
