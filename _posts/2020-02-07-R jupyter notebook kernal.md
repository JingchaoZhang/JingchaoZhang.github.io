---
layout: single
author_profile: false
---

Install Miniconda and configure Jupyter notebook to run R kernals.  
First make sure X11 forwarding works on your local machine. Then follow the steps below.
```bash
#install Miniconda and install jupyter
mkdir ~/conda; cd ~/conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
vim ~/.bashrc
export PATH=/home/$USER/miniconda3/bin:$PATH
source ~/.bashrc
pip install jupyter

#configure R
conda create -n localR R=3.6.0
source activate localR
conda install r-png
conda install r-irkernel jupyter_client
conda install -c anaconda jupyter_client
R -e "IRkernel::installspec(name = '$CONDA_DEFAULT_ENV', displayname = 'R ($CONDA_DEFAULT_ENV)', user = TRUE)"
conda deactivate
jupyter notebook
```

To use it in the future, type the following lines.
```bash
source ~/.bashrc
jupyter notebook
```
