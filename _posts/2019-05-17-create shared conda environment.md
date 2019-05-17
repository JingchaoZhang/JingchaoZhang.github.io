---
layout: single
author_profile: false
---

- create a shared location in $COMMON
```bash
mkdir /common/GROUP/conda
chown :GROUP conda
chmod 770 conda
```
- create conda env in the shared location
```bash
module load anaconda/4.6
conda create -p /common/GROUP/conda/new-env python=2.7
```
- all members in this group can load this env using the absolute path
```bash
module load anaconda/4.6
source activate /common/GROUP/conda
```
- To migrate existing conda env to the new location
```
conda list --explicit > spec-file.txt
conda create -p /common/GROUP/conda/new-env --file spec-file.txt
```
