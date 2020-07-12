---
layout: single
author_profile: false
---

```bash
#install Miniconda
mkdir ~/conda; cd ~/conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
vim ~/.bashrc
export PATH=/home/$USER/miniconda3/bin:$PATH
source ~/.bashrc

#Create R env
conda create -n Renv r-essentials r-base

#Activate R env
conda activate Renv

#Start a R session by typing R
R

#install packages within the R session
install.packages(c('lme4','LaplacesDemon','quantreg','plyr','boot','broom','MASS','dplyr'))

#Exit the R session. Run script interactively using Rscript.
Rscript Rcode.R

#To run other R jobs in the future, repeat steps below
source ~/.bashrc
Rscript Rcode.R
```

