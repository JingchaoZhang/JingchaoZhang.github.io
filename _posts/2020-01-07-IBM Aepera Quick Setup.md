---
layout: single
author_profile: false
---

Aspera is IBM's high-performance file transfer software which allows for the transfer large files and data sets with predictable, reliable and secure delivery regardless of file size or transfer distance from a site which has the aspera server running.  The NCBI recommend the use of aspera for transfer of data sets from their site.  

One-time initial setup
```bash
wget https://download.asperasoft.com/download/sw/cli/3.9.2/ibm-aspera-cli-3.9.2.1426.c59787a-linux-64-release.sh
bash ibm-aspera-cli-3.9.2.1426.c59787a-linux-64-release.sh
echo 'export PATH=~/.aspera/cli/bin/:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Copy from a remote server to local HPC directory
```bash
ascp -QT -l 100M user@source_hostname:source_file /local/destination/folder/
```
* \-T Disable encryption for maximum throughput.
* \-l max_rate  

For a full list of `ascp` options, check this [link](https://download.asperasoft.com/download/docs/ascp/3.5.2/html/index.html#dita/ascp_usage.html)
