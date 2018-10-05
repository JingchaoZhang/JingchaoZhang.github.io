---
layout: single
author_profile: false
---

Tensorflow 1.10.1
https://www.tensorflow.org/install/source#tested_source_configurations  
https://github.com/tensorflow/tensorflow/blob/v1.10.1/tensorflow/tools/pip_package/setup.py  

```bash
bioconda-utils build --docker --loglevel=debug  --packages=freesurfer recipes config.yml
docker run --rm -it -v $PWD:/repo bioconda/bioconda-utils-build-env:latest bash
conda build -c hcc -m /opt/conda/conda_build_config.yaml -m /opt/conda/lib/python3.6/site-packages/bioconda_utils/bioconda_utils-conda_build_config.yaml .
conda render -c hcc -m /opt/conda/conda_build_config.yaml -m /opt/conda/lib/python3.6/site-packages/bioconda_utils/bioconda_utils-conda_build_config.yaml .
```

Kill all docker processes
``bash
docker ps -aq | xargs docker rm
```
