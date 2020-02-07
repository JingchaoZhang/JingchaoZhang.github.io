---
layout: single
author_profile: false
---

1. Install Miniconda if you haven't done so.
```bash
mkdir ~/conda; cd ~/conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
vim ~/.bashrc
export PATH=/home/$USER/miniconda3/bin:$PATH
```
2. Install rstudio package and start GUI
```bash
conda create -n rstudio rstudio=1.1.456
source activate rstudio
rstudio
```
