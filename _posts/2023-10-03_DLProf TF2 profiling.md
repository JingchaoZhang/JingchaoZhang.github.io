---
layout: single
author_profile: false
---

### Install dependences
```bash
pip install scikit-learn pandas==1.2.2
```

### Launch profiling session
```bash
docker run --rm --gpus=1 --shm-size=1g --ulimit memlock=-1 \
  --ulimit stack=67108864 -it -p 8000:8000 -v \
  /root/CloudClassification-main/:/data nvcr.io/nvidia/tensorflow:21.07-tf2-py3
```

### Start Dlprofview
```bash
dlprofviewer -b 0.0.0.0 dlprof_dldb.sqlite
```

## 1 GPU job output
```bash
--- 120.04520773887634 seconds ---
Expert Systems Feedback: 5 issues detected. Note that expert systems is still experimental as are all recommended changes

Problem detected: 
  XLA is not enabled: No XLA ops detected
Recommended change: 
  Try enabling XLA. See https://www.tensorflow.org/xla/#enable_xla_for_tensorflow_models for information on how to enable XLA.

Problem detected: 
  11 ops were eligible to use tensor cores but none are using FP16
Recommended change: 
  Try enabling AMP (Automatic Mixed Precision). For more information: https://developer.nvidia.com/automatic-mixed-precision

Problem detected: 
  The GPU is underutilized: Only 8.0% of the profiled time is spent on GPU kernel operations
Recommended change: 
  "Other" has the highest (non-GPU) usage at 60.2%. Investigate the dataloading pipeline as this often indicates too much time
 is being spent here

Problem detected: 
  Unable to split profile into training iterations: key node  not found
Recommended change: 
  Specify key node by setting the --key_node argument

Problem detected: 
  GPU Memory is underutilized: Only 2% of GPU Memory is used
Recommended change: 
  Try increasing batch size by 4x to increase data throughput

```

## 2 GPU job output
```bash
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
  The GPU is underutilized: Only 6.9% of the profiled time is spent on GPU kernel operations
Recommended change: 
  "Other" has the highest (non-GPU) usage at 47.4%. Investigate the dataloading pipeline as this often indicates too much time
 is being spent here

Problem detected: 
  Unable to split profile into training iterations: key node  not found
Recommended change: 
  Specify key node by setting the --key_node argument

Problem detected: 
  GPU Memory is underutilized: Only 2% of GPU Memory is used
Recommended change: 
  Try increasing batch size by 4x to increase data throughput
```
