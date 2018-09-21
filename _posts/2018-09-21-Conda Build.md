---
layout: single
author_profile: false
---

```bash
bioconda-utils build --docker --loglevel=debug  --packages=freesurfer recipes config.yml
docker run -it -v $PWD:/repo bioconda/bioconda-utils-build-env:latest bash
conda build -c hcc -m /opt/conda/conda_build_config.yaml -m /opt/conda/lib/python3.6/site-packages/bioconda_utils/bioconda_utils-conda_build_config.yaml .
```
