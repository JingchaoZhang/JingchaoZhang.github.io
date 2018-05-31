---
layout: single
author_profile: false
---

```bash
module load anaconda
conda create -n tensorflow-keras tensorflow keras ipykernel python=3.6

source activate tensorflow-keras
python -m ipykernel install --user --name tensorflow-keras --display-name "Keras Tensorflow py36 (myenv)"
mkdir -p $WORK/.jupyter
mv ~/.local/share/jupyter/kernels $WORK/.jupyter
ln -s $WORK/.jupyter/kernels ~/.local/share/jupyter/kernels
```
