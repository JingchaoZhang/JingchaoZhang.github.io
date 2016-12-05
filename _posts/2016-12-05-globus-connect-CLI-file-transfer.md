---
layout: single
author_profile: false
---

- Add ssh key to your Globus account. [instruction](https://docs.globus.org/faq/command-line-interface/#how_do_i_generate_an_ssh_key_to_use_with_the_globus_command_line_interface)
- ssh jingchao@cli.globusonline.org  
$ transfer '--' hcc#tusker/home/swanson/jingchao/file1 hcc#crane/home/swanson/jingchao/file2  
  
or in one command  
  
- ssh jingchao@cli.globusonline.org transfer '--' hcc#tusker/home/swanson/jingchao/file1 hcc#crane/home/swanson/jingchao/file2
