---
layout: single
author_profile: false
---

- Code in this demo
[Github](https://github.com/JingchaoZhang/DLProf_Demo)

- Documentation  
[User Guide](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html#profiling)
[DLProf Viewer](https://docs.nvidia.com/deeplearning/frameworks/dlprof-viewer-user-guide/index.html)

- MNIST example code  
[PyTorch MNIST example](https://github.com/pytorch/examples/blob/master/mnist/main.py)
  
- DLProf installation
```
pip install nvidia-pyindex
pip install nvidia-dlprof[pytorch] #For PyTorch
pip install nvidia-dlprofviewer #DLProf Viewer
```
  
- PyTorch installation
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
  
- Profiling PyTorch with nvidia_dlprof_pytorch_nvtx
add the following lines to your PyTorch network
```python
import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init()
```
You should also run the training/inference loop with PyTorch’s NVTX Context Manager with the following:
```python
with torch.autograd.profiler.emit_nvtx():
```
  
- Training on GPU without DLProf
```bash
python mnist.py
```
  
- Training on GPU with DLProf
```bash
dlprof --mode=pytorch python mnist.py
```
  
- Sample DLProf terminal output
```bash
[DLProf-04:35:49] Aggregating profile data
[DLProf-04:35:53] Creating dlprof database at ./dlprof_dldb.sqlite
[DLProf-04:35:53] Writing profile data to dlprof database
[DLProf-04:36:00] Writing aggregated data to dlprof database
[DLProf-04:36:07] Writing expert_systems report to (stdout)
Expert Systems Feedback: 6 issues detected. Note that expert systems is still experimental as are all recommended changes

Problem detected: 
  48 ops were eligible to use tensor cores but none are using FP16
Recommended change: 
  Try enabling AMP (Automatic Mixed Precision). For more information: https://developer.nvidia.com/automatic-mixed-precision

Problem detected: 
  The GPU is underutilized: Only 0.6% of the profiled time is spent on GPU kernel operations
Recommended change: 
  Dataloader has the highest (non-GPU) usage at 87.3%. Investigate the dataloading pipeline as this often indicates too much time is being spent here

Problem detected: 
  87.3% of the aggregated run was spent in the dataloader while not simultaneously running on the GPU
Recommended change: 
  Focus on reducing time spent in the training data input process. This could be time spent in file reading, preprocessing and augmentation or file transfer.
Consider using NVIDIA DALI, a library that is a high performance alternative to built-in data loaders and data iterators. Learn more here: https://developer.nvidia.com/DALI

Problem detected: 
  The aggregated iteration range of 0 to 938 contains a lot of variation
Recommended change: 
  Try limiting the iteration range to a steady range by rerunning with the --database option and setting --iter_start=3 --iter_stop=107

Problem detected: 
  Convolution operations were detected but torch.backends.cudnn.benchmark was not enabled.
Recommended change: 
  Try setting torch.backends.cudnn.benchmark = True in your network. For best performance, the input shapes should be relatively stable.

Problem detected: 
  GPU Memory is underutilized: Only 7% of GPU Memory is used
Recommended change: 
  Try increasing batch size by 4x to increase data throughput
```
  
- Files
```bash
total 249M
drwxr-xr-x 15 jingchao jingchao 4.0K Jan 18 23:17 ../
-rw-rw-r--  1 jingchao jingchao  24M Jan 18 23:35 nsys_profile.qdrep
-rw-r--r--  1 jingchao jingchao 193M Jan 18 23:35 nsys_profile.sqlite
-rw-r--r--  1 jingchao jingchao  33M Jan 18 23:36 dlprof_dldb.sqlite
-rw-rw-r--  1 jingchao jingchao 5.5K Jan 18 23:40 mnist.py
drwxrwxr-x  2 jingchao jingchao 4.0K Jan 18 23:40 ./
```
  
- Visualize results
```bash
$ dlprofviewer dlprof_dldb.sqlite 
[dlprofviewer-04:46:36 AM UTC] dlprofviewer running at http://localhost:8000
```
![alt text](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/images/DLProf_sample_output.png)

