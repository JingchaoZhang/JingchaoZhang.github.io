---
layout: single
author_profile: false
---

1.
```bash
echo "eval $(perl -Mlocal::lib)" >> ~/.bashrc
source ~/.bashrc
cpan
cpan> install Module::Load::Conditional  
```
2.  
```bash
curl -L http://cpanmin.us | perl - Module::Load::Conditional
```
