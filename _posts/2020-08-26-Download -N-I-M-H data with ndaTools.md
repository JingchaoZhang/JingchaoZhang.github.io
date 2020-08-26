---
layout: single
author_profile: false
---

In order to access NIMH Data Archive, users need to apply for an account using this [link](https://nda.nih.gov/abcd/request-access). Project PI needs to add the new user to their project for data access.  
  
  
Follow the steps below to create the data package:
1. Click the top tab "Get Data"
1. In "TEXT SEARCH" Search ABCD
1. Select the checkbox for the first result
1. Click "Add to Workspace"
1. Click the filter icon on the top-right corner
1. Click "Submit to Filter Cart"
1. wait for this to finish (takes a while): 
1. Click "create data package/add to study"
1. Select the data you want.
1. Click on "create data package" at the bottom of the page
1. Go to your account dashboard
1. Click on "Data packages" to view a list of your packages. 
  
  
Install and use [nda-tools](https://github.com/NDAR/nda-tools) for data package download. 
```bash
#install Miniconda
mkdir ~/conda; cd ~/conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

#If you do not want default conda base env
conda config --set auto_activate_base false

#Install nda-tools
conda create -n nda-tools pip
conda activate nda-tools
pip install nda-tools

#Now you can download using nda-tools
downloadcmd PACKAGEID -dp -d /path/to/your/download/directory -wt NUM_OF_THREADS -v -u $USER -p PASSWORD
```
  
  
For large dataset, perform the download in a SLURM job. Below is a sample SLURM file named "submit.sh":
```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=100G
#SBATCH --time=7-00
#SBATCH -p week-long-cpu
#SBATCH -e job.%J.err
#SBATCH -o job.%J.out

conda activate nda-tools
downloadcmd PACKAGEID -dp -d /path/to/your/download/directory -wt 32 -v -u $USER
```
  
  
Then submit this job to SLURM.
```
sbatch submit.sh
```
