---
layout: single
author_profile: false
---


New Project
```bash
mkdir packages/blast
cd packages/blast
cp ../../anaconda-project.yml.example ./anaconda-project.yml
anaconda-project add-env-spec -n blast-2.7.1 blast=2.7.1
anaconda-project remove-env-spec -n default
anaconda-project list-env-specs
anaconda-project lock
```

Existing Project
```bash
anaconda-project add-env-spec -n blast-2.7.1 blast=2.7.1
anaconda-project lock
```
