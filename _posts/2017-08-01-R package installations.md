---
layout: single
author_profile: false
---

To see the directories that R searches for libraries, from within R you can do:
```R
.libPaths();
```

To install a package to user specified directory, do
```bash
export R_LIBS="/home/your_username/R_libs"
```
Then either from within R, do
```R
install.packages('package_name', repos="http://cran.r-project.org")
```
Or download the tarbar and install it directly from the command line
```bash
R CMD INSTALL -l /home/your_username/R_libs pkg_version.tar.gz
```

A handy repo to use
```R
install.packages("package_name",repos="http://cran.cnr.berkeley.edu")
```

To permanently change to the repo address, do
```bash
echo "options(repos = c(CRAN = "https://cran.revolutionanalytics.com"))" > ~/.Rprofile
```
