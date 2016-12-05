---
layout: single
author_profile: false
---

- option 1  
  
```bash
echo 'eval $(perl -Mlocal::lib)' >> ~/.bashrc
source ~/.bashrc
cpan
cpan> install Module::Load::Conditional  
```  

- option 2  
  
```bash
curl -L http://cpanmin.us | perl - Module::Load::Conditional
```
