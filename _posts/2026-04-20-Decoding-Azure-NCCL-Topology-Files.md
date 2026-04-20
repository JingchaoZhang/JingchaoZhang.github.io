---
layout: single
title: "Decoding Azure's NCCL Topology Files"
author_profile: false
---

## Introduction

Spin up an `Standard_ND96isr_H100_v5` VM on Azure, poke around `/opt/microsoft/`, and you'll find a curious little file: `ndv5-topo.xml`. It's only ~3 KB, but it's the single most important configuration knob between your training job and full bandwidth.

On the Azure HPC images, NCCL picks it up automatically through `/etc/nccl.conf` — a config file that `libnccl.so` reads at `ncclCommInit()` time. Most users never notice it's there. On a vanilla Marketplace image neither the conf file nor the env var exists, so NCCL silently falls back to its (wrong) auto-discovered topology and runs slower. Either way, understanding *what's in that file* and *why it has to exist* is the fastest way to internalize how NUMA, PCIe, and GPU clusters actually work on a virtualized cloud host.

In this post I'll:

1. Decode the XML field-by-field, including the cryptic `affinity="00000000,0000ffff,ffffffff"` bitmask.
2. Explain what NUMA, UPI, and GPUDirect RDMA mean in the context of an H100 VM.
3. Walk through reconstructing the file from scratch using only `/sys` data on a live VM.
4. Provide a script that generates the XML automatically and diffs it against Azure's official one.

By the end, you'll be able to explain to a colleague exactly why this 3 KB file determines whether your `all_reduce` runs at line rate or at half speed.

## The big picture: what's inside an NDv5 VM

An `ND96isr_H100_v5` is a **two-socket** machine. Each socket is an Intel Sapphire Rapids Xeon 8480C with 48 vCPUs, ~957 GB of local DRAM, 4 NVIDIA H100 GPUs, and 4 NVIDIA ConnectX-7 InfiniBand NICs (400 Gb/s each). The two sockets are stitched together by Intel's **UPI** (Ultra Path Interconnect) link.

That gives us two **NUMA nodes** — independent memory + I/O domains — that share a single OS image:

<div class="mermaid">
flowchart TB
    subgraph NUMA0
        CPU0[CPU0 - vCPUs 0-47]
        MEM0[Local DRAM ~957GB]
        B0[PCIe Switch - virtual bridge]
        G0[GPU 0 - H100]
        NIC0[NIC 0 - IB 400Gb/s]
        DOTS0[... 3 more GPU+NIC pairs]

        CPU0 --- MEM0
        CPU0 --- B0
        B0 --- G0
        B0 --- NIC0
        G0 --- NIC0
    end

    subgraph NUMA1
        CPU1[CPU1 - vCPUs 48-95]
        MEM1[Local DRAM ~957GB]
        B1[PCIe Switch - virtual bridge]
        G1[GPU 4 - H100]
        NIC1[NIC 4 - IB 400Gb/s]
        DOTS1[... 3 more GPU+NIC pairs]

        CPU1 --- MEM1
        CPU1 --- B1
        B1 --- G1
        B1 --- NIC1
        G1 --- NIC1
    end

    CPU0 -.UPI - slow.- CPU1

    NVS[NVSwitch - all 8 GPUs, 900 GB/s NVLink]
    G0 -.- NVS
    G1 -.- NVS

    FAB[InfiniBand fabric - to other VMs]
    NIC0 --> FAB
    NIC1 --> FAB

    classDef gpu fill:#76B900,stroke:#000,color:#fff
    classDef nic fill:#0078D4,stroke:#000,color:#fff
    classDef cpu fill:#444,stroke:#000,color:#fff
    classDef mem fill:#fff3b0,stroke:#a06000,color:#000
    classDef br fill:#eee,stroke:#888,color:#333

    class G0,G1 gpu
    class NIC0,NIC1 nic
    class CPU0,CPU1 cpu
    class MEM0,MEM1 mem
    class B0,B1 br
</div>

Three traffic patterns matter:

1. **GPU ↔ NIC inside the same NUMA node** — this is the **GPUDirect RDMA** path. The NIC reads/writes GPU HBM directly over PCIe peer-to-peer, never copying through host RAM. This only works when the GPU and NIC sit under the same PCIe switch.
2. **GPU ↔ GPU** — handled by **NVSwitch** at 900 GB/s of NVLink. Doesn't touch the CPU or PCIe at all.
3. **CPU ↔ remote NUMA** — this is the **UPI** path. Going across UPI roughly doubles memory latency and halves effective bandwidth. If a process on socket 0 ends up driving a GPU on socket 1, every CUDA kernel launch and every DMA descriptor pays the UPI tax.

The whole job of `ndv5-topo.xml` is to make sure NCCL never accidentally picks pattern #3.

## Why is UPI such a problem?

UPI is Intel's high-speed point-to-point link between sockets — successor to QPI. On Sapphire Rapids it runs at 16 GT/s with ~45 GB/s aggregate per direction. Sounds fast, but compare it to local DRAM:

| Path | Latency | Bandwidth |
|---|---|---|
| Local DRAM | ~80 ns | ~300 GB/s per socket |
| Remote DRAM via UPI | ~140 ns | ~90 GB/s effective |
| Local GPUDirect RDMA | ~2 µs | 400 Gb/s line rate |
| GPUDirect RDMA across UPI | ~3–5 µs | ~250 Gb/s, often falls back to host bounce buffers |

Linux already knows about this asymmetry — `numactl --hardware` reports it explicitly:

```
node distances:
node   0   1
  0:  10  21
  1:  21  10
```

Local access is 1.0×, remote is 2.1× slower. That's the cost of UPI in one number.

For training workloads, the worst case isn't slow memory — it's **losing GPUDirect RDMA**. If NCCL routes GPU0's traffic through NIC4 (which lives on the other socket), the DMA descriptor table has to traverse UPI on every transfer. Often the driver gives up on peer-to-peer entirely and falls back to staging through host memory. AllReduce throughput drops by 30–40%.

## What's actually in `ndv5-topo.xml`

Here's the first half of the real file, slightly trimmed:

```xml
<system version="1">
  <cpu numaid="0" affinity="00000000,0000ffff,ffffffff"
       arch="x86_64" vendor="GenuineIntel" familyid="6" modelid="143">

    <pci busid="ffff:ff:01.0" class="0x060400"
         link_speed="32.0 GT/s PCIe" link_width="16"
         vendor="0x0000" device="0x0000"
         subsystem_vendor="0x0000" subsystem_device="0x0000">
      <pci busid="0001:00:00.0" class="0x030200"
           link_speed="32.0 GT/s PCIe" link_width="16"/>
      <pci busid="0101:00:00.0" class="0x020700"
           link_speed="32.0 GT/s PCIe" link_width="16"/>
    </pci>

    <!-- 3 more bridges for the other GPU+NIC pairs in NUMA 0 -->
  </cpu>

  <cpu numaid="1" affinity="ffffffff,ffff0000,00000000" ...>
    <!-- 4 more bridges for NUMA 1 -->
  </cpu>
</system>
```

The structure is dead simple once you see it:

```
<system>
  <cpu>           ← one per NUMA node
    <pci>         ← synthetic PCIe bridge, one per GPU+NIC pair
      <pci/>      ← the GPU
      <pci/>      ← the NIC
    </pci>
    ... × 4
  </cpu>
  ... × 2
</system>
```

Two `<cpu>` blocks (one per NUMA node), each containing four `<pci>` "bridges," each containing one GPU and one NIC. That's it. Eight bridges total = the 8 GPU+NIC pairs of an NDv5 VM.

### Decoding the PCI device classes

The `class` attribute is a standard PCI device class code:

| Class | Meaning |
|---|---|
| `0x060400` | PCI-to-PCI bridge — the synthetic switches Azure declares |
| `0x030200` | 3D controller — the H100 GPUs |
| `0x020700` | InfiniBand controller — the ConnectX-7 NICs |

The bridges have `vendor="0x0000"` (i.e. unidentified) because they don't actually exist in hardware — Azure invents them so NCCL has somewhere to hang the GPU+NIC pairing.

### Decoding the bus IDs

PCI bus IDs in Linux look like `domain:bus:device.function`. Azure assigns each H100 and each IB VF its own PCI segment, so you see things like `0001:00:00.0`, `0002:00:00.0`, `0101:00:00.0`. The pairing is:

| NUMA | GPUs | IB NICs |
|---|---|---|
| 0 | `0001`, `0002`, `0003`, `0008` | `0101`, `0102`, `0103`, `0104` |
| 1 | `0009`, `000a`, `000b`, `000c` | `0105`, `0106`, `0107`, `0108` |

The synthetic bridges live at `ffff:ff:01.0` through `ffff:ff:08.0` — `ffff:ff` is a deliberately invalid domain/bus that no real device would ever use, signaling "this bridge is fictitious."

## The `affinity` bitmask, demystified

This is the part that confuses everyone the first time. Let's break down NUMA 0's mask:

```
affinity = "00000000, 0000ffff, ffffffff"
            └─ high ─┘ └─ mid ─┘ └─ low ─┘
            vCPU 64-95  32-63     0-31
```

It's a **96-bit binary number** describing which logical CPUs belong to this NUMA node, written in **three 32-bit hex words, most-significant first**.

Why 8 hex characters per word? Because hex packs 4 bits per digit, and 32 ÷ 4 = 8. Why three words? Because the VM has 96 vCPUs and `ceil(96 / 32) = 3`. Nothing magical.

Each hex digit expands to 4 bits, where **1 = "this vCPU is in the node"** and **0 = "it's not"**:

| Hex | Binary | vCPUs in this 4-bit group |
|---|---|---|
| `0` | `0000` | none |
| `f` | `1111` | all 4 |
| `3` | `0011` | the lowest 2 |
| `8` | `1000` | only the highest |

For NUMA 0:

```
00000000  →  bits 64-95: all zero      → no vCPUs
0000ffff  →  bits 32-47: sixteen ones  → vCPUs 32-47
ffffffff  →  bits 0-31:  thirty-two ones → vCPUs 0-31
```

Total: 16 + 32 = **48 vCPUs** (vCPU 0 through 47). For NUMA 1, the bits flip and you get vCPUs 48–95. Together they cover all 96 vCPUs, with no overlap. ✅

The reason we only see `0` and `f` (never `3`, `7`, `c`, etc.) is that the NUMA boundary on NDv5 happens to fall on a 16-CPU multiple. If a SKU had, say, 44 cores per node instead of 48, you'd see partial nybbles like `00000fff` mixed in.

## Why does this file even need to exist?

On a **bare-metal** server, NCCL discovers all this automatically by walking `/sys/class/pci_bus/`. Linux exposes the real PCIe tree — every device's parent switch, every switch's parent root complex, every root complex's NUMA node. NCCL just reads it.

On an **Azure VM**, the hypervisor abstracts the host PCIe topology. The guest sees:

- 8 GPUs at flat PCI segments `0001:..` through `000c:..`
- 8 IB NICs at flat PCI segments `0101:..` through `0108:..`
- **No parent PCIe switches anywhere in the tree**

From NCCL's perspective, every device looks like it's hanging off the system root. With nothing to distinguish "GPU0 is close to NIC0" from "GPU0 is close to NIC4," NCCL falls back to heuristics — and on a virtualized host it often picks the wrong NIC for a given GPU. The XML file restores the missing information by telling NCCL "pretend these four GPU+NIC pairs sit under one virtual switch, and that switch lives on NUMA 0."

It's a **synthetic topology hint**: a contract between Azure (which knows the real wiring) and NCCL (which can't see through the hypervisor).

## Reconstructing the file from a live VM

The fun part. Everything in the XML can be recovered from `/sys` on the VM. Let's do it step by step.

### Step 1: NUMA cpumaps

```bash
cat /sys/devices/system/node/node0/cpumap
# 00000000,0000ffff,ffffffff

cat /sys/devices/system/node/node1/cpumap
# ffffffff,ffff0000,00000000
```

Those are exactly the strings that go into `affinity="..."`. Done.

### Step 2: Walk PCI devices

```bash
for d in /sys/bus/pci/devices/*; do
    cls=$(cat "$d/class" 2>/dev/null)
    ven=$(cat "$d/vendor" 2>/dev/null)
    numa=$(cat "$d/numa_node" 2>/dev/null)

    case "$cls:$ven" in
        0x030200:0x10de) kind=GPU ;;
        0x020700:0x15b3) kind=NIC ;;
        *) continue ;;
    esac
    echo "$kind  numa=$numa  $(basename "$d")"
done | sort
```

On my VM this prints exactly the 16 devices, with the right NUMA mapping:

```
GPU  numa=0  0001:00:00.0
GPU  numa=0  0002:00:00.0
GPU  numa=0  0003:00:00.0
GPU  numa=0  0008:00:00.0
GPU  numa=1  0009:00:00.0
GPU  numa=1  000a:00:00.0
GPU  numa=1  000b:00:00.0
GPU  numa=1  000c:00:00.0
NIC  numa=0  0101:00:00.0
NIC  numa=0  0102:00:00.0
NIC  numa=0  0103:00:00.0
NIC  numa=0  0104:00:00.0
NIC  numa=1  0105:00:00.0
NIC  numa=1  0106:00:00.0
NIC  numa=1  0107:00:00.0
NIC  numa=1  0108:00:00.0
```

Note: there's also an **Ethernet** Mellanox VF (`mlx5_an0`, vendor `15b3` device `101a`) that powers Azure Accelerated Networking. That one is *not* part of the IB topology — filter it out by matching device id `0x101e` (the IB VF) specifically.

The PCIe bridges in the XML (class `0x068000`, NVIDIA device `22a3`) at `0004..0007` show up in `lspci` too — these are **real** NVSwitch bridges that the GPUs use to reach the NVLink fabric. NCCL discovers NVSwitch through a separate path (`nvidia-smi nvlink`), so the topology XML deliberately omits them.

### Step 3: PCIe link rate

The XML claims every device runs at "32.0 GT/s PCIe" with "link_width 16" — i.e. PCIe Gen5 x16. You can verify that for the GPUs:

```bash
cat /sys/bus/pci/devices/0001:00:00.0/current_link_speed
# 32.0 GT/s PCIe
cat /sys/bus/pci/devices/0001:00:00.0/current_link_width
# 16
```

For the IB NICs you'll see `Unknown` and `0` instead — that's because they're SR-IOV virtual functions and sysfs only exposes link state for the physical function on the host. The XML hard-codes the Gen5 x16 numbers because that's what the PF underneath actually negotiates.

### Step 4: Generate and diff

Putting it all together, here's the script — short enough to read in one sitting:

```bash
#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-$PWD/my-ndv5-topo.xml}"
REF="/opt/microsoft/ndv5-topo.xml"

# Pad sysfs cpumap to 3 32-bit words.
pad_cpumap() {
    local raw="$1"; local IFS=','; local -a w; read -ra w <<<"$raw"
    while (( ${#w[@]} < 3 )); do w=("00000000" "${w[@]}"); done
    echo "${w[*]}"
}

emit_pair() {
    cat <<XML
    <pci busid="ffff:ff:$1.0" class="0x060400" link_speed="32.0 GT/s PCIe" link_width="16" vendor="0x0000" device="0x0000" subsystem_vendor="0x0000" subsystem_device="0x0000">
      <pci busid="$2" class="0x030200" link_speed="32.0 GT/s PCIe" link_width="16"/>
      <pci busid="$3" class="0x020700" link_speed="32.0 GT/s PCIe" link_width="16"/>
    </pci>
XML
}

N0=$(pad_cpumap "$(cat /sys/devices/system/node/node0/cpumap)")
N1=$(pad_cpumap "$(cat /sys/devices/system/node/node1/cpumap)")

mapfile -t GPUS < <(for d in /sys/bus/pci/devices/*; do
    [[ $(cat $d/class)  == 0x030200 ]] || continue
    [[ $(cat $d/vendor) == 0x10de   ]] || continue
    echo "$(cat $d/numa_node) $(basename $d)"
done | sort)

mapfile -t NICS < <(for d in /sys/bus/pci/devices/*; do
    [[ $(cat $d/class)  == 0x020700 ]] || continue
    [[ $(cat $d/vendor) == 0x15b3   ]] || continue
    [[ $(cat $d/device) == 0x101e   ]] || continue
    echo "$(cat $d/numa_node) $(basename $d)"
done | sort)

# Split per NUMA node.
g0=(); g1=(); n0=(); n1=()
for x in "${GPUS[@]}"; do [[ ${x%% *} == 0 ]] && g0+=("${x#* }") || g1+=("${x#* }"); done
for x in "${NICS[@]}"; do [[ ${x%% *} == 0 ]] && n0+=("${x#* }") || n1+=("${x#* }"); done

{
    echo '<system version="1">'
    echo "  <cpu numaid=\"0\" affinity=\"$N0\" arch=\"x86_64\" vendor=\"GenuineIntel\" familyid=\"6\" modelid=\"143\">"
    b=1
    for i in "${!g0[@]}"; do printf -v h "%02x" $b; emit_pair $h "${g0[$i]}" "${n0[$i]}"; b=$((b+1)); done
    echo "  </cpu>"
    echo "  <cpu numaid=\"1\" affinity=\"$N1\" arch=\"x86_64\" vendor=\"GenuineIntel\" familyid=\"6\" modelid=\"143\">"
    for i in "${!g1[@]}"; do printf -v h "%02x" $b; emit_pair $h "${g1[$i]}" "${n1[$i]}"; b=$((b+1)); done
    echo "  </cpu>"
    echo "</system>"
} > "$OUT"

[[ -r $REF ]] && diff -w -q "$REF" "$OUT" && echo "MATCH ✓"
```

When I ran this on my VM, the diff against `/opt/microsoft/ndv5-topo.xml` came back **byte-identical**. Every field — the cpumaps, the busids, the bridge numbering, even the GPU+NIC pairing order — is mechanically derivable from `/sys`.

## How the Azure HPC image wires it up

On my live `ND96isr_H100_v5` VM, `echo $NCCL_TOPO_FILE` prints **nothing** — the env var isn't set anywhere in the user's shell. Yet NCCL still finds the file. The mechanism is `/etc/nccl.conf`:

```bash
$ cat /etc/nccl.conf
NCCL_IB_PCI_RELAXED_ORDERING=1
NCCL_TOPO_FILE=/opt/microsoft/ndv5/topo.xml
NCCL_IGNORE_CPU_AFFINITY=1
```

NCCL reads this file at `ncclCommInit()` time and treats every line as if it were exported in the environment. The conf file is created at image build time by `/opt/azurehpc/customizations/ndv5.sh`, which also symlinks `/opt/microsoft/ndv5/topo.xml` → `/opt/microsoft/ndv5-topo.xml` and starts the NVIDIA Fabric Manager / `nvidia-peermem` module.

The two companion knobs are worth knowing about:

- **`NCCL_IB_PCI_RELAXED_ORDERING=1`** — enables PCIe relaxed ordering for IB DMA. Lower latency on Sapphire Rapids; harmless on older hosts.
- **`NCCL_IGNORE_CPU_AFFINITY=1`** — tells NCCL *not* to pin its own background threads using the `affinity` mask from the topology XML. It still uses the mask to figure out NUMA-locality for GPU/NIC pairing, but leaves CPU thread placement to your training framework (PyTorch, MPI, etc.).

## Verifying NCCL actually uses it

First, confirm the conf file (and/or env var) is in place:

```bash
cat /etc/nccl.conf 2>/dev/null
echo "NCCL_TOPO_FILE=$NCCL_TOPO_FILE"
```

If neither has the topology path, set it yourself:

```bash
export NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml
```

Then run an actual collective. The cleanest way on Azure is the NVIDIA PyTorch container — it already has NCCL + the HPC-X IB plugin compiled in. Note the `--device=/dev/infiniband --network=host --privileged` flags — without them, NCCL can't see the IB NICs and silently falls back to TCP over the Docker bridge:

```bash
sudo docker run --rm --gpus all \
  --network=host --ipc=host --privileged \
  --device=/dev/infiniband \
  -v /opt/microsoft:/opt/microsoft:ro \
  -e NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml \
  -e NCCL_DEBUG=INFO \
  nvcr.io/nvidia/pytorch:24.10-py3 \
  bash -c "cd /tmp && \
    git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && make -j MPI=0 CUDA_HOME=/usr/local/cuda >/dev/null && \
    ./build/all_reduce_perf -b 1G -e 1G -g 8 -n 5"
```

The first important lines in the NCCL log confirm the topology file was picked up and the IB transport (not sockets) is in use:

```
NCCL INFO NCCL_TOPO_FILE set by environment to /opt/microsoft/ndv5-topo.xml
NCCL INFO Setting affinity for GPU 0 to ffff,ffffffff
NCCL INFO NET/IB : Using [0]mlx5_ib0:1/IB/SHARP [1]mlx5_ib1:1/IB/SHARP
                        [2]mlx5_ib2:1/IB/SHARP [3]mlx5_ib3:1/IB/SHARP
                        [4]mlx5_ib4:1/IB/SHARP [5]mlx5_ib5:1/IB/SHARP
                        [6]mlx5_ib6:1/IB/SHARP [7]mlx5_ib7:1/IB/SHARP
                        [8]mlx5_an0:1/RoCE [RO]; OOB ib0:172.16.10.58<0>
NCCL INFO Using network IBext_v8
NCCL INFO DMA-BUF is available on GPU device 0
```

Things to check:

1. **`NCCL_TOPO_FILE set by environment`** — the XML was loaded. ✓
2. **`Setting affinity for GPU 0 to ffff,ffffffff`** — 48 bits set, matching NUMA 0's cpumap exactly. The topology XML's `affinity="..."` was parsed correctly. ✓
3. **`NET/IB : Using [0]mlx5_ib0 ... [7]mlx5_ib7`** — all 8 IB NICs visible (plus `mlx5_an0` exposed as RoCE fallback, which NCCL will deprioritize). ✓
4. **`Using network IBext_v8`** — the native IB transport via the HPC-X plugin, *not* the Socket (TCP) fallback. If you see `Using network Socket` instead, something is wrong with the IB device passthrough. ✗
5. **`DMA-BUF is available`** — the kernel facility GPUDirect RDMA uses for zero-copy GPU↔NIC transfers is enabled. ✓
6. **`SHARP`** tags on every NIC — NVIDIA's in-network reduction engine. NCCL only enables SHARP when it trusts the topology, so seeing it on all 8 NICs is a strong signal that the XML's GPU↔NIC pairing is being honored.

## What this single-VM run can't show

Single-node AllReduce is carried almost entirely by NVSwitch, so the topology file's value is partially hidden. The file's biggest impact is on **multi-node** runs, where every byte actually crosses the IB fabric:

- Per-channel routing tags like `via NET/IB/0/GDRDMA` (emitted only when there's a remote peer).
- **`PXB`** (same PCIe switch / same NUMA — good), **`PIX`** (same PCIe bridge — best), or **`SYS`** (routed through system bus, i.e. crossed UPI — bad) tags in the channel setup log.
- Effective per-NIC IB busbw. With the topology file: ~370 Gb/s per 400-Gb link. Without: often ~250 Gb/s because NCCL crosses UPI or loses GDR.

To reproduce those on multiple VMs, run the same `all_reduce_perf` binary under `mpirun` or `torchrun` across 2+ nodes.

## TL;DR

Azure NDv5 VMs have **2 NUMA nodes**, each owning 48 vCPUs, ~957 GB DRAM, 4 H100 GPUs, and 4 IB NICs. The hypervisor hides the real PCIe switch hierarchy, so NCCL can't autodiscover which GPU pairs with which NIC. `/opt/microsoft/ndv5-topo.xml` is a **synthetic topology hint** that restores the missing information — two `<cpu>` blocks with four virtual `<pci>` bridges each, every bridge grouping one GPU with its co-located NIC. Without it, traffic crosses the slow inter-socket UPI link and loses GPUDirect RDMA. Every value in the file is mechanically derivable from `/sys`, and the script in this post regenerates it byte-for-byte.

The whole file is 3 KB of XML. Worth understanding.
