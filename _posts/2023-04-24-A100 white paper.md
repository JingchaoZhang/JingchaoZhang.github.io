---
layout: single
author_profile: false
---

- Third generation Tensor Cores
  - Tensor Cores are specialized high-performance compute cores that perform mixed-precision matrix multiply and accumulate calculations in a single operation. 
  - Supported data types:
    - INT4
    - binary
    - TensorFloat32 (TF32)
    - IEEE Compliant FP64
    - BFloat16 (BF16) (BF16/FP32 mixed-precision Tensor Core operations perform at the same speed as FP16/FP32 mixed-precision Tensor Core operations)

- TF32
  - The new TF32 operations run **10X faster** than the FP32 FMA operations available with the previous generation data center GPU
  - TF32 combines the range of FP32 with the precision of FP16
  - Compared to FP32 on V100, TF32 on A100 provides over **6X speedup** for training the BERT-Large model
  - TF32 is the default mode for TensorFlow, PyTorch and MXNet, starting with NGC Deep Learning Container 20.06 Release

- Fine-grained Structured Sparsity
  - fine-grained structured sparsity and the 2:4 pattern
  - balanced workload distribution and even utilization of compute nodes
  - structured sparse matrices can be efficiently compressed
  - With fine-grained structured sparsity, _INT8_ Tensor Core operations on A100 offer **20X more performance** than on V100, and _FP16_ Tensor Core operations are **5X faster** than on V100

- Multi-instance GPU (MIG)
  - spatial partitioning
  - each GPU instance has its own memory, cache, and streaming multiprocessor (**isolated GPU memory and physical GPU resources**)

- NVIDIA® NVSwitch
  - Six second-generation nvswitch
  - GPU to GPU communication to peak at **600 GB/s**
  - If all GPUs are communicating with each other, the total amount of data transferred peaks at **4.8 TB/s** for both directions.

|Specs| 1st Generation| 2nd Generation | 3rd Generation|
|--------------|-----------|------------|------------|
|Number of GPUs with direct connection / node|Up to 8|Up to 8|Up to 8|
|NVSwitch GPU-to-GPU bandwidth|300GB/s|600GB/s|900GB/s|
|Total aggregate bandwidth|2.4TB/s|4.8TB/s|7.2TB/s|
|Supported NVIDIA Architectures|Volta|Ampere|Hopper|

- NVIDIA NVLink®
  - Third-generation nvlink
  - Each A100 GPU uses **twelve NVLink interconnects** to communicate with all six NVSwitches (two links from each GPU to each switch)

|Specs| 2nd Generation| 3rd Generation | 4th Generation|
|--------------|-----------|------------|------------|
|NVLink bandwidth per GPU|300GB/s|600GB/s|900GB/s|
|Maximum Number of Links per GPU|6|12|18|
|Supported NVIDIA Architectures|Volta|Ampere|Hopper|

- Mellanox ConnectX-6 HDR
  - 200 Gb/s per port (4 data lanes operating at 50 Gb/s or 200 Gb/s total)
  - **8** single-port Mellanox ConnectX-6 **200Gb/s HDR** InfiniBand ports (also configurable as 200Gb/s Ethernet ports) providing **3.2 Tb/s** of peak bandwidth
  - DGX A100 incorporates a one-to-one relationship between the IO cards and the GPUs, which means each GPU can communicate directly with external sources without blocking other GPUs’ access to the network.
  - DGX A100 includes an **additional dual-port** ConnectX-6 card that can be used for high-speed connection to **external storage**

- PCIe Gen4
  - NVIDIA A100 GPUs are connected to the PCI switch infrastructure over x16 PCI Express Gen 4 (PCIe Gen4) buses that provide 31.5 Gb/s each for a total of 252 Gb/s
  - These are the links that provide access to the **Mellanox ConnectX-6, the NVMe storage, and the CPUs**.
