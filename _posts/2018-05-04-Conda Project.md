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

#push to repo
git checkout -b NewBatch
git add #files
git commit -a -m "new packages"
git push origin NewBatch
#create pull request
git checkout master; git pull
git branch -d NewBatch
```

Existing Project
```bash
anaconda-project add-env-spec -n blast-2.7.1 blast=2.7.1
anaconda-project lock
```
