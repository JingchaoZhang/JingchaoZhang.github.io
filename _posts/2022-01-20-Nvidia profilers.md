---
layout: single
author_profile: false
---

- [Profiling PyTorch (PyProf)](https://docs.nvidia.com/deeplearning/frameworks/pyprof-user-guide/index.html)  
  - PyProf is a tool that profiles and analyzes the GPU performance of PyTorch models. PyProf aggregates kernel performance from Nsight Systems or NvProf. (Please note that **NVProf is currently being phased out**, and it is recommended to **use Nsight Systems to future proof the profile process**.)
  - On June 30th 2021, NVIDIA will no longer make contributions to the PyProf repository.
  - To profile models in PyTorch, please use [NVIDIA Deep Learning Profiler (DLProf)](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/).
  - Example code on [GitHub](https://github.com/NVIDIA/PyProf)
  
- [Deep Learning Profiler (DLProf)](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html)
  - The Deep Learning Profiler (DLProf) User Guide provides instructions on using the DLProf tool to improve the performance of deep learning models.
  - Version 1.8.0 is the **final release of DLProf**.
  - DLProf supports running multiple GPUs on a single node. Nothing special needs to be done by the user to have DLProf profile a multi-GPU training run.
  
- [Nsight Developer Tools](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
  - [Nsight Tutorial given by Max Katz/NVIDIA](https://drive.google.com/file/d/1TEPiRpxqZXK2iqzy1uAQoAlrH3u7z-iX/view?usp=sharing)
  
- [NVProf](https://docs.nvidia.com/cuda/profiler-users-guide/)
  - [Nvprof Transition Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvprof-guide)  
This guide provides tips for moving from nvprof to NVIDIA Nsight Compute CLI. NVIDIA Nsight Compute CLI tries to provide as much feature and usage parity as possible with nvprof, but some features are now covered by different tools and some command line options have changed their name or meaning.
  
- Useful Links
  - [NVIDIA’s Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
  - [NVIDIA Nsight Developer Tools](https://docs.nvidia.com/#nvidia-nsight-developer-tools)
  - [Legacy Developer Tools](https://docs.nvidia.com/#nvidia-nsight-developer-tools_legacy-developer-tools)
