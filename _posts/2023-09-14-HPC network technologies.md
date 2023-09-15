---
layout: single
author_profile: false
---

In the rapidly evolving landscape of High-Performance Computing (HPC) and Artificial Intelligence (AI), understanding the nuances between various networking protocols and libraries is crucial for performance optimization and system design. This blog aims to demystify key technologies such as RDMA, RoCE, TCP/IP, IPoIB, InfiniBand, NCCL, and MPI by categorizing them according to the OSI model layers at which they operate. By doing so, we provide a structured framework that aids in grasping how these technologies interact and complement one another in real-world applications. Whether you are an enterprise architect, a developer, or a researcher looking to harness the full potential of HPC and AI, this comprehensive guide will serve as a valuable reference point.

## Categorized by OSI layers
| Layer | RDMA         | RoCE        | TCP/IP     | IPoIB       | IB         | NCCL     | MPI     |
|-------|--------------|-------------|------------|-------------|------------|----------|---------|
| Physical (1) | - | - | - | - | Yes | - | - |
| Data Link (2) | - | Yes | - | - | Yes | - | - |
| Network (3) | - | - | Yes | Yes | - | - | - |
| Transport (4) | Yes | Yes | Yes | - | - | - | - |
| Application (7) | - | - | - | - | - | Yes | Yes |


## Categorized by function
| Term      | Full Form                                   | Layer       | Description                                                                                                    | Use-Cases                                            | Compatibility/Co-existence               |
|-----------|---------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------|-----------------------------------------|
| RDMA      | Remote Direct Memory Access                  | Data Link   | Direct memory access from one computer into another without involving either's OS                               | HPC, Data Transfer                                   | Can be used over IB or Ethernet (RoCE)  |
| RoCE      | RDMA over Converged Ethernet                 | Data Link   | An extension of RDMA, it allows RDMA to run over Ethernet networks                                              | Data Center, Cloud Networking                        | Ethernet-based                          |
| TCP/IP    | Transmission Control Protocol/IP             | Transport   | Standard communication protocol over the Internet, involves packet switching                                    | General Internet, Web Services                       | Most Networks                           |
| IPoIB     | IP over InfiniBand                           | Transport   | Allows the transmission of IP traffic over InfiniBand, making it compatible with existing IP-based applications  | IP Services on IB network                            | Shares InfiniBand port                   |
| IB        | InfiniBand                                   | Data Link   | High-throughput, low-latency networking stack, commonly used in HPC                                             | HPC, Data Centers                                    | Exclusive port usually                   |
| NCCL      | NVIDIA Collective Communications Library      | Application    | Optimized primitives library for collective communications in multi-GPU environments                            | Deep Learning, AI Training                           | Can work over IB, RoCE, or even TCP/IP   |
| MPI       | Message Passing Interface                    | Application | A standardized and portable API used for parallel computing, operates over various kinds of networks            | High-Performance Computing, parallelized applications | Can work over IB, RoCE, TCP/IP, and more |

- **RDMA**: This is the foundation for zero-copy networking. It offers lower latency and higher bandwidth.
  
- **RoCE**: It's RDMA adapted for Ethernet. It's useful in modern data center applications where you might not have InfiniBand but still want low latency.

- **TCP/IP**: This is the most commonly used protocol stack and is generally slower and more resource-intensive than RDMA or RoCE.

- **IPoIB**: It's a way to map IP over InfiniBand so that you can run IP-based applications without modification. It's generally slower than native InfiniBand but offers compatibility.

- **InfiniBand (IB)**: This is a high-performance network protocol that uses high-throughput and low-latency networking technologies. Typically, InfiniBand will have its own dedicated port, but it can share a port if running IPoIB.

- **NCCL**: This is a library for collective communication that's particularly useful in multi-GPU setups for machine learning. It's protocol agnostic to an extent and can work over InfiniBand, RoCE, or even TCP/IP if necessary.

- **MPI (Message Passing Interface)**: MPI is an application-layer API that allows for high-performance communication between nodes in a parallel computing environment. Unlike the other technologies listed, which are more focused on networking layers, MPI operates at the application layer and can be used on top of multiple kinds of networking technologies including InfiniBand, RoCE, and TCP/IP.
