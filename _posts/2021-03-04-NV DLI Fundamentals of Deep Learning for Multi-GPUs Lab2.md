---
layout: single
author_profile: false
---

# Lab 2: Multi-GPU DL Training Implementation using Horovod
(Horovod)[https://github.com/horovod/horovod] is a distributed deep learning training framework. It is available for TensorFlow, Keras, PyTorch, and Apache MXNet. In this lab you will learn about what Horovod is and how to use it, by distributing across multiple GPUs the training of the classification model we started with in Exercise 3 of Lab 1.

This lab draws heavily on content provided in the (Horovod tutorials)[https://github.com/horovod/tutorials].

## Intro to Horovod
Horovod is an open source tool originally (developed by Uber)[https://eng.uber.com/horovod/] to support their need for faster deep learning model training across their many engineering teams. It is part of a growing ecosystem of approaches to distributed training, including for example (Distributed TensorFlow)[https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md]. Uber decided to develop a solution that utilized MPI for distributed process communication, and the (NVIDIA Collective Communications Library (NCCL))[https://developer.nvidia.com/nccl] for its highly optimized implementation of reductions across distributed processes and nodes. The resulting Horovod package delivers on its promise to scale deep learning model training across multiple GPUs and multiple nodes, with only minor code modification and intuitive debugging.

Since its inception in 2017 Horovod has matured significantly, extending its support from just TensorFlow to Keras, PyTorch, and Apache MXNet. Horovod is extensively tested and has been used on some of the largest DL training runs done to date, for example, supporting exascale deep learning on the (Summit system, scaling to over 27,000 V100 GPUs)[https://arxiv.org/pdf/1810.01993.pdf]:

## Horovod's MPI Roots
Horovod's connection to MPI runs deep, and for programmers familiar with MPI programming, much of what you program to distribute model training with Horovod will feel very familiar. For those unfamiliar with MPI programming, a brief discussion of some of the conventions and considerations required when distributing processes with Horovod, or MPI, is worthwhile.

Horovod, as with MPI, strictly follows the (Single-Program Multiple-Data (SPMD))[https://en.wikipedia.org/wiki/SPMD] paradigm where we implement the instruction flow of multiple processes in the same file/program. Because multiple processes are executing code in parallel, we have to take care about (race conditions)[https://en.wikipedia.org/wiki/Race_condition] and also the synchronization of participating processes.

Horovod assigns a unique numerical ID or rank (an MPI concept) to each process executing the program. This rank can be accessed programmatically. As you will see below when writing Horovod code, by identifying a process's rank programmatically in the code we can take steps such as:
* Pin that process to its own exclusive GPU.
* Utilize a single rank for broadcasting values that need to be used uniformly by all ranks.
* Utilize a single rank for collecting and/or reducing values produced by all ranks.
* Utilize a single rank for logging or writing to disk.
As you work through this course, keep these concepts in mind and especially that Horovod will be sending your single program to be executed in parallel by multiple processes. Keeping this in mind will support your intuition and understanding about why we do what we do with Horovod, even though you will only be making edits to a single program.

With Horovod, which can run multiple processes across multiple GPUs, you typically use a single GPU per training process. Part of what makes Horovod simple to use is that it utilizes MPI, and as such, uses much of the MPI nomenclature. The concept of a rank in MPI is of a unique process ID. In this lab we will be using the term "rank" extensively. If you would like to know more about MPI concepts that are utilized heavily in Horovod, please refer to the Horovod documentation.

Schematically, let's look at how MPI can run multiple GPU processes across multiple nodes. Note how each process, or rank, is pinned to a specific GPU:
![Alt text](/images/NV DLI Fundamentals of Deep Learning for Multi-GPUs Lab2 Figure1.PNG?raw=true "Optional Title")
#Notebook
```python
import horovod.keras as hvd
#Baseline: train the model
!python fashion_mnist.py --epochs 5 --batch-size 512
#Modify the training script
!cp fashion_mnist.py fashion_mnist_original.py

```
#Python Script
```python


```
