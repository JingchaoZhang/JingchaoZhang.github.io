---
layout: single
author_profile: false
---

### Launch profiling session
```bash
docker run --rm --gpus=2 --ipc=host --shm-size=1g\
  --ulimit memlock=-1 --ulimit stack=67108864 -it -p 8000:8000 -v \
  /root/CloudClassification-main/:/data \
  nvcr.io/nvidia/tensorflow:21.07-tf2-py3
```

### Install dependences
```bash
pip install scikit-learn pandas==1.2.2
```

### Profile with DLProf
```bash
dlprof --force=true --nsys_opts='-t cuda,nvtx,cublas,cudnn -s none' \
  -m tensorflow2 python train.py

# Explanations of the flags
-f, --force=
   Possible values are 'true' or 'false'.
   If true, overwrite all existing result files
   with the same output filename (QDSTREM, QDREP, 
   SQLITE, CSV, JSON).
   Default is 'false'.
--nsys_opts=
   Specify nsys args within quotes '"[<nsys args>]"'.
   Customize the args passed to Nsight Systems.
   Option must include the default for DLProf to
   operate correctly.
   Default arguments are '"-t cuda,nvtx -s none"'.
-m, --mode=
   Possible values are 'simple', 'tensorflow2', 'tensorrt'.
   Specify the target framework being profiled. Use
   'simple' to generate only high level metrics agnostic
   to any framework. Use all other options to
   generate detailed metrics and reports specific to
   the framework.
   Default is 'tensorflow2'.

# Options from `nsys profile --help` related to `--nsys_opts`
-t, --trace=
   Possible values are 'cuda', 'nvtx', 'osrt', 'cublas', 'cudnn', 'opengl', 'opengl-annotations', 'mpi', 'oshmem', 'openacc', 'openmp', 'vulkan', 'vulkan-annotations' or 'none'.
   Select the API(s) to trace. Multiple APIs can be selected, separated by commas only (no spaces).
   If '<api>-annotations' is selected, the corresponding API will also be traced.
   If 'none' is selected, no APIs are traced.
-f, --force=
   Possible values are 'true' or 'false'.
   If true, overwrite all existing result files
   with the same output filename (QDSTREM, QDREP, 
   SQLITE, CSV, JSON).
   Default is 'false'.
-s, --sample=
   Possible values are 'cpu' or 'none'.
   Select the entity to sample. Select 'none' to disable sampling. 
   Default is 'cpu'. Application scope.
```

### Command line output from DLProf
```bash
Processing events...
Saving temporary "/tmp/nsys-report-4817-75a0-f379-4106.qdstrm" file to disk...

Creating final output files...
Processing [==============================================================100%]
Saved report file to "/tmp/nsys-report-4817-75a0-f379-4106.qdrep"
Exporting 10995948 events: [==============================================100%]

Exported successfully to
/tmp/nsys-report-4817-75a0-f379-4106.sqlite
Report file moved to "/data/./nsys_profile.qdrep"
Report file moved to "/data/./nsys_profile.sqlite"

[DLProf-02:53:59] DLprof completed system call successfully
2023-10-04 02:54:31.055796: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic librar
y libcudart.so.11.0
[DLProf-02:54:32] Initializing Nsight Systems database
[DLProf-02:56:03] Reading System Information from Nsight Systems database
[DLProf-02:56:03] Reading Domains from Nsight Systems database
[DLProf-02:56:03] Reading Ops from Nsight Systems database
[DLProf-02:57:45] Reading CUDA API calls from Nsight Systems database
[DLProf-02:59:58] Correlating network models with kernel and timeline data
[DLProf-02:59:58] Found 1 iteration using key_node ""
Iterations: [86461724964]enacc', 'openmp', 'vulk
Aggregating data over 1 iteration: iteration 0 start (0 ns) to iteration 0 end (86461724964 ns)

[DLProf-02:59:59] Aggregating profile data
[DLProf-03:01:03] Creating dlprof database at ./dlprof_dldb_1.sqlite
[DLProf-03:01:03] Writing profile data to dlprof database
[DLProf-03:03:19] Writing aggregated data to dlprof database
[DLProf-03:04:46] Writing expert_systems report to (stdout)
Expert Systems Feedback: 5 issues detected. Note that expert systems is still experimental as are all recommended changes

Problem detected: 
  XLA is not enabled: No XLA ops detected
Recommended change: 
  Try enabling XLA. See https://www.tensorflow.org/xla/#enable_xla_for_tensorflow_models for information on how to enable XLA.

Problem detected: 
  22 ops were eligible to use tensor cores but none are using FP16
Recommended change: 
  Try enabling AMP (Automatic Mixed Precision). For more information: https://developer.nvidia.com/automatic-mixed-precision

Problem detected: 
  The GPU is underutilized: Only 4.3% of the profiled time is spent on GPU kernel operations
Recommended change: 
  "Other" has the highest (non-GPU) usage at 67.8%. Investigate the dataloading pipeline as this often indicates too much time
 is being spent hereath. If this fails, 'op

Problem detected: 
  Unable to split profile into training iterations: key node  not found
Recommended change: 
  Specify key node by setting the --key_node argument

Problem detected: 
  GPU Memory is underutilized: Only 2% of GPU Memory is used
Recommended change: 
  Try increasing batch size by 4x to increase data throughput
```

The profiling gives the performance suggestions summarized below:
- Enable XLA to optimize TensorFlow computations.
- Activate AMP (Automatic Mixed Precision) to leverage tensor cores with FP16.
- Address GPU underutilization by investigating and optimizing the dataloading pipeline.
- Boost data throughput by quadrupling the batch size to better utilize GPU memory.

### Start DLProfView
We can visualize the profiling results using DLProfView. First, we need to start the server with the command below:
```bash
dlprofviewer -b 0.0.0.0 dlprof_dldb.sqlite
```

In your browser, go to `http://localhost:8000`. The profiling summary looks like below. It shows a high level view of GPU utlization.

![Figure_1](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-04-figures/dlprof_1.png)  

We can take a deeper look at the operations and kernal activities by switching to the `Ops and Kernel Summaries` tab. It shows the GPU/CPU time, number of calls, and data type for all operations. The most time consuming oprations in this test is SoftMax, NCCLAllReduce, and Adam optimization. 
![Figure_2](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-04-figures/dlprof_2.png)  

If you are profiling multiple GPUs, you can see the utlizations of each GPU. 
![Figure_3](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-04-figures/dlprof_3.png)  

### Drill deeper using Nsight System
The AzureHPC image comes with `NSight Systems` and `Nsight Compute` pre-installed. To view the profiling results using `NSight Systems`, find the app and open it from your Desktop. 
![Figure_4](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-04-figures/nsight_1.png)  

As part of the profiling process, a `.qdrep` file was generated. Find and load this file into the `NSight Systems` app. 
![Figure_5](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-04-figures/nsight_2.png)  

Once the loading is done, we can observe the timeline of the computations on the top panel. We can optionally open the events view on the bottom session. 
![Figure_6](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-04-figures/nsight_3.png)  

To gain further details on the profiling results. We can zoom in into the timeline. The first problem we can observe is the `Device to Host Memcpy` that is happening between the training batches. This inidicates the data loading process needs to be optimzied. Instead of sending data between the device and host, all computations should be done on the GPU. 
![Figure_7](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-04-figures/nsight_4.png)  

Another observation is the low `Theoretical occupancy` in some of the kernal call, which indicates low utlizations of the GPUs. Ideally we would like to see at least 50% of `Theoretical occupancy`. 
![Figure_9](https://raw.githubusercontent.com/JingchaoZhang/JingchaoZhang.github.io/master/_posts/2023-10-04-figures/nsight_6.png)  

