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

### Start DLProfView
```bash
dlprofviewer -b 0.0.0.0 dlprof_dldb.sqlite
```

In your browser, go to `http://localhost:8000`

### Drill deeper into a kernel using Nsight Compute
```bash

```