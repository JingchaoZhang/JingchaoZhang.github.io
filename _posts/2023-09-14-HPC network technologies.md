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

## Network interfaces
Understanding the network interfaces used by InfiniBand, RoCE, and TCP/IP is crucial for their effective deployment and operation. Below is a brief explanation:

### InfiniBand (IB)
- **Network Interface**: InfiniBand Host Channel Adapter (HCA)
- **Details**: InfiniBand uses its own specialized network interfaces known as [HCAs](https://www.google.com/search?sca_esv=565545338&rlz=1C1CHBF_enUS1013US1013&sxsrf=AM9HkKnaY_cAe3tG3Uf77OinaP3Wgu8Qxg:1694748959609&q=InfiniBand+Host+Channel+Adapter+(HCA)&tbm=isch&source=lnms&sa=X&ved=2ahUKEwif4uHt16uBAxU2gGoFHdIsBv4Q0pQJegQIChAB&biw=2048&bih=995&dpr=1.25). These are different from standard Ethernet NICs (Network Interface Cards). HCAs are designed to provide low-latency and high-throughput communication.

### RDMA over Converged Ethernet (RoCE)
- **Network Interface**: Converged Network Adapter (CNA) or RDMA-enabled NIC
- **Details**: RoCE often uses [Converged Network Adapters](https://www.google.com/search?q=Converged+Network+Adapter+(CNA)&tbm=isch&ved=2ahUKEwiXgdnu16uBAxUHAWIAHRX5CLcQ2-cCegQIABAA&oq=Converged+Network+Adapter+(CNA)&gs_lcp=CgNpbWcQAzIFCAAQgAQ6BAgjECdQ0gRY0gRgxAdoAHAAeACAAW2IAcsBkgEDMS4xmAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=IdEDZdfsIYeCiLMPlfKjuAs&bih=995&biw=2048&rlz=1C1CHBF_enUS1013US1013) (CNAs) that support both RDMA and traditional Ethernet communications. These adapters can also be RDMA-enabled NICs specifically optimized for RDMA over Ethernet.

### TCP/IP
- **Network Interface**: Ethernet Network Interface Card (NIC)
- **Details**: The standard network interface for TCP/IP-based communication is an [Ethernet NIC](https://www.bing.com/images/search?q=Ethernet+NIC&form=HDRSC4&first=1). These are ubiquitous and come in various speeds like Gigabit Ethernet, 10 Gigabit Ethernet, etc.

It's important to note that each of these network interfaces is optimized for the particular protocol stack they are designed to support. While you can run different protocols over the same physical infrastructure (for example, RoCE and TCP/IP over Ethernet), the network interface card must support those protocols for them to operate efficiently.
