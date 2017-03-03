---
layout: single
author_profile: false
---

```bash
mysql
Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MariaDB connection id is 1299
Server version: 5.5.50-MariaDB MariaDB Server

Copyright (c) 2000, 2016, Oracle, MariaDB Corporation Ab and others.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

MariaDB [(none)]> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mod_hpcdb          |
| mod_logger         |
| mod_shredder       |
| moddb              |
| modw               |
| modw_aggregates    |
| modw_filters       |
| mysql              |
| performance_schema |
| slurm_acct_db      |
| test               |
+--------------------+
12 rows in set (0.03 sec)

MariaDB [(none)]> use mod_hpcdb;
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A

Database changed
MariaDB [mod_hpcdb]> show tables;
+-----------------------------------+
| Tables_in_mod_hpcdb               |
+-----------------------------------+
| hpcdb_accounts                    |
| hpcdb_allocation_breakdown        |
| hpcdb_allocations                 |
| hpcdb_allocations_on_resources    |
| hpcdb_email_addresses             |
| hpcdb_fields_of_science           |
| hpcdb_fields_of_science_hierarchy |
| hpcdb_jobs                        |
| hpcdb_organizations               |
| hpcdb_people                      |
| hpcdb_people_on_accounts_history  |
| hpcdb_principal_investigators     |
| hpcdb_requests                    |
| hpcdb_resource_allocated          |
| hpcdb_resource_specs              |
| hpcdb_resource_types              |
| hpcdb_resources                   |
| hpcdb_system_accounts             |
| schema_version_history            |
+-----------------------------------+
19 rows in set (0.00 sec)

MariaDB [mod_hpcdb]> truncate hpcdb_fields_of_science;
Query OK, 0 rows affected (0.09 sec)

MariaDB [mod_hpcdb]> Bye
```

```bash
xdmod-import-csv -t hierarchy -i hierarchy.csv
xdmod-import-csv -t group-to-hierarchy -i group.csv
xdmod-admin --truncate --jobs
xdmod-shredder -r Tusker -f slurm -i tusker.txt && xdmod-shredder -r Crane -f slurm -i crane.txt && xdmod-shredder -r Sandhills -f slurm -i sandhills.txt
xdmod-ingestor
```
