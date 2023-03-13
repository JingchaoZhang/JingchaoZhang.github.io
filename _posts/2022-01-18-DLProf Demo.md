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
  
- DLProf installation (you can skip DLProf and PyTorch installations if you use the .yml file from [Github](https://github.com/JingchaoZhang/DLProf_Demo))
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
![alt text](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/images/DLProf_terminal_output.png)

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

