---
layout: single
title: "Three VMSS Uniform Settings for HPC/AI Users"
author_profile: false
---

*If you deploy GPU VMs on Azure using [VMSS Uniform mode](https://learn.microsoft.com/en-us/azure/virtual-machine-scale-sets/virtual-machine-scale-sets-orchestration-modes), three settings can make the difference between a successful allocation and a mysterious failure. Here's what `platform_fault_domain_count`, `single_placement_group`, and `overprovision` do — and why you may want to change them for HPC.*

---

## The Terraform Cheat Sheet

If you're in a hurry, here's the answer:

```hcl
resource "azurerm_linux_virtual_machine_scale_set" "gpu" {
  # ...
  platform_fault_domain_count = 1
  single_placement_group      = true
  overprovision               = false
}
```

> **Important:** Use `azurerm_linux_virtual_machine_scale_set`, not the legacy `azurerm_virtual_machine_scale_set`. The legacy resource is deprecated and doesn't expose `platform_fault_domain_count` as a top-level property.

Or with Azure CLI:

```bash
az vmss create \
  --platform-fault-domain-count 1 \
  --single-placement-group true \
  --disable-overprovision \
  ...
```

Now let's understand *why*.

---

## 1. Fault Domain Count: Set to 1

### What is a Fault Domain?

A [**Fault Domain (FD)**](https://learn.microsoft.com/en-us/azure/virtual-machines/availability-set-overview#fault-domains) is Azure's way of grouping VMs that share common physical hardware — think of it as a rack or power/network boundary in a datacenter. If that rack loses power or a top-of-rack switch fails, only VMs in that FD are affected.

```
┌─────────────────── Datacenter ───────────────────┐
│                                                   │
│  ┌─── FD 0 ───┐  ┌─── FD 1 ───┐  ┌─── FD 2 ───┐ │
│  │  Power A    │  │  Power B    │  │  Power C    │ │
│  │  Switch A   │  │  Switch B   │  │  Switch C   │ │
│  │             │  │             │  │             │ │
│  │  ┌──────┐   │  │  ┌──────┐   │  │  ┌──────┐   │ │
│  │  │ VM-1 │   │  │  │ VM-2 │   │  │  │ VM-3 │   │ │
│  │  │ VM-4 │   │  │  │ VM-5 │   │  │  │ VM-6 │   │ │
│  │  └──────┘   │  │  └──────┘   │  │  └──────┘   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                   │
└───────────────────────────────────────────────────┘
```

For **web servers**, spreading across FDs is great — if one rack goes down, the others keep serving traffic.

### How FD Balancing Causes Allocation Failures

When `platformFaultDomainCount` is not explicitly set, Azure assigns a region-dependent default (up to 5). Azure **tries to evenly distribute** your VMs across those fault domains, placing the next VM on the FD with the fewest machines.

This creates a problem for GPU SKUs:

```
You request 4 GPU VMs with FD count = 5:

FD 0: ✅ Capacity available  → [VM-1]
FD 1: ✅ Capacity available  → [VM-2]
FD 2: ✅ Capacity available  → [VM-3]
FD 3: ❌ No GPU capacity     → ALLOCATION FAILS
FD 4: ✅ Capacity available  → (never reached)

Result: ❌ Entire request rejected
        Even though FDs 0, 1, 2, and 4 have room.
```

GPU VMs like `Standard_ND96isr_H100_v5` are **scarce**. Unlike general-purpose VMs where every rack has hundreds of slots, H100 nodes exist in limited quantities. Requiring even distribution across multiple FDs dramatically reduces the chance of a successful allocation.

### With FD = 1

```
You request 4 GPU VMs with FD count = 1:

Azure places VMs wherever capacity exists — no balancing math.

FD 0: [VM-1] [VM-2]
FD 2: [VM-3]
FD 4: [VM-4]

Result: ✅ Allocated successfully
```

**Setting FD = 1 does not mean "force all VMs into one physical rack."** It means "remove the FD balancing constraint." Azure can still spread VMs across racks — it just doesn't *require* it.

### Why This Trade-off is Safe for HPC/AI

| Concern | Web Servers | GPU Training |
|---------|:-----------:|:------------:|
| If 1 node dies... | Other nodes keep serving traffic | Entire training job fails anyway |
| Partial availability useful? | Yes — users see degraded, not broken | No — distributed training is all-or-nothing |
| FD spreading benefit? | High | None |
| Allocation success priority? | Lower — capacity is abundant | Critical — GPU VMs are scarce |

For distributed GPU training, the biggest risk isn't a rack failure — it's **not getting your VMs at all**.

### Key Facts

| Property | Detail |
|----------|--------|
| Default value (VMSS Uniform, no AZ) | 5 |
| Default value (VMSS Flexible) | 1 |
| Mutable after creation? | **No** — immutable, requires new VMSS |
| CLI parameter | `--platform-fault-domain-count` |
| Terraform property | `platform_fault_domain_count` |
| ARM API property | `properties.platformFaultDomainCount` |

---

## 2. Single Placement Group: Set to true

### What is a Placement Group?

A [placement group](https://learn.microsoft.com/en-us/azure/virtual-machine-scale-sets/virtual-machine-scale-sets-placement-groups) is a logical grouping that ensures all VMs within it are deployed on the same physical network fabric — close enough for low-latency, high-bandwidth communication.

```
┌───────── Placement Group ──────────┐
│                                     │
│  ┌──────┐  ┌──────┐  ┌──────┐      │
│  │ VM-1 │  │ VM-2 │  │ VM-3 │      │
│  │ GPU×8│  │ GPU×8│  │ GPU×8│      │
│  └──┬───┘  └──┬───┘  └──┬───┘      │
│     │         │         │           │
│  ═══╪═════════╪═════════╪═══ IB ══  │
│     InfiniBand Fabric               │
│     400 Gb/s per port               │
│                                     │
└─────────────────────────────────────┘
```

### Why single_placement_group = true?

For **InfiniBand-enabled VMs** (ND-series), all VMs need to be on the same IB fabric to use RDMA for inter-node communication. Setting `single_placement_group = true` helps ensure this.

```
single_placement_group = true          single_placement_group = false
┌──── Placement Group A ────┐          ┌──── PG A ────┐  ┌──── PG B ────┐
│ VM-1  VM-2  VM-3  VM-4    │          │ VM-1  VM-2   │  │ VM-3  VM-4   │
│ ════ IB Fabric ═══════    │          │ ═══ IB A ══  │  │ ═══ IB B ══  │
│ All VMs can RDMA to all   │          │              │  │              │
└───────────────────────────┘          └──────────────┘  └──────────────┘
                                       ❌ VM-1 cannot RDMA to VM-3
                                       (different IB fabrics)
```

With `single_placement_group = false`, Azure may split VMs across multiple placement groups on **different physical clusters**. Those VMs can still communicate over TCP/Ethernet, but not over InfiniBand. For multi-node training that depends on NCCL + RDMA, this can significantly degrade performance.

### The Trade-off

| `single_placement_group` | Max VMs | IB/RDMA | Use Case |
|:---:|:---:|:---:|---|
| `true` | 100 (default, expandable via tag) | ✅ Guaranteed | GPU/HPC training |
| `false` | 1,000 | ❌ Not guaranteed | Stateless scale-out, web apps |

> **Note:** The 100-VM limit can be raised beyond 100 using a specific resource tag. Contact your Microsoft account team for details.

---

## 3. Overprovision: Set to false

### What is Overprovisioning?

When [overprovisioning](https://learn.microsoft.com/en-us/azure/virtual-machine-scale-sets/virtual-machine-scale-sets-design-overview#overprovisioning) is enabled (the default), Azure creates **more VMs than you requested**, waits for enough to provision successfully, then deletes the extras. This speeds up deployment for common VM SKUs with abundant capacity.

```
overprovision = true (default)
─────────────────────────────
You request 4 VMs.
Azure creates 6.
5 succeed, 1 fails.
Azure keeps 4, deletes 1 extra.

Result: Fast deployment, but needed 6 slots to get 4.
```

### Why Disable It for GPU VMs?

**Problem 1: Transient extra VMs disrupt orchestration.** Azure [temporarily spins up more VMs than requested](https://learn.microsoft.com/en-us/azure/virtual-machine-scale-sets/virtual-machine-scale-sets-design-overview#overprovisioning), then deletes the extras once enough succeed. The docs warn this "can cause confusing behavior for an application that is not designed to handle extra VMs appearing and then disappearing." For multi-node GPU jobs that discover peers by enumerating VMSS instances, VMs that appear and vanish mid-setup can break rendezvous.

**Problem 2: It doesn't help for GPU SKUs.** Overprovisioning is a speed optimization for scenarios where individual VMs occasionally fail to provision (e.g., spot eviction, transient host issues). For GPU VMs in a single placement group, either the cluster has capacity or it doesn't — provisioning extra VMs won't change that.

> **Note:** The extra VMs created by overprovisioning are [not billed and do not count toward quota](https://learn.microsoft.com/en-us/azure/virtual-machine-scale-sets/virtual-machine-scale-sets-design-overview#overprovisioning). The concern is operational disruption, not cost.

```
overprovision = false (recommended for GPU)
────────────────────────────────────────────
You request 4 VMs.
Azure creates exactly 4.
All 4 succeed.

Result: No transient extra VMs, clean orchestration.
```

### The Trade-off

| `overprovision` | Deployment Speed | Use Case |
|:---:|:---:|---|
| `true` | Faster for common SKUs | Web, app servers |
| `false` | Same for GPU SKUs | GPU/HPC, scarce capacity |

---

## How These Three Settings Work Together

These three settings are not independent — they work as a system for GPU/HPC workloads:

```
┌────────────────────────────────────────────────────────┐
│              VMSS Uniform for GPU/HPC                   │
│                                                        │
│  platform_fault_domain_count = 1                        │
│  ├── Removes FD balancing constraint                    │
│  └── Azure places VMs wherever capacity exists          │
│                                                        │
│  single_placement_group = true                          │
│  ├── All VMs on the same IB fabric                      │
│  └── Required for RDMA/InfiniBand communication         │
│                                                        │
│  overprovision = false                                  │
│  ├── No transient extra VMs disrupting job orchestration         │
│  └── Predictable instance IDs for job orchestration     │
│                                                        │
│  Combined effect:                                       │
│  ✅ Maximum allocation success rate                     │
│  ✅ Guaranteed IB connectivity between all nodes        │
│  ✅ No capacity waste on overprovisioned extras          │
└────────────────────────────────────────────────────────┘
```

---

## Important Caveats

**All three are immutable.** `platform_fault_domain_count` and `single_placement_group` cannot be changed after VMSS creation. `overprovision` can technically be updated, but changing the others requires creating a new VMSS.

**Terraform will destroy and recreate.** If you change `platform_fault_domain_count` or `single_placement_group` in your Terraform config, the `azurerm_linux_virtual_machine_scale_set` resource is marked `ForceNew` — Terraform will destroy the existing VMSS and create a new one.

**Check your current settings:**

```bash
az vmss show --name myVMSS -g myRG \
  --query "{fd:virtualMachineProfile.platformFaultDomainCount, \
            spg:singlePlacementGroup, \
            op:overprovision}" -o table
```

---

## Summary

| Setting | Default | GPU/HPC Value | Why |
|---------|:-------:|:-------------:|-----|
| `platform_fault_domain_count` | 5 | **1** | Removes FD balancing — prevents allocation failure on scarce GPU SKUs |
| `single_placement_group` | true | **true** | Keeps all VMs on same IB fabric for RDMA |
| `overprovision` | true | **false** | Prevents transient extra VMs from disrupting GPU job orchestration |

```hcl
# The complete recommended configuration
resource "azurerm_linux_virtual_machine_scale_set" "gpu" {
  name                = "gpu-vmss"
  sku                 = "Standard_ND96isr_H100_v5"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location

  platform_fault_domain_count = 1
  single_placement_group      = true
  overprovision               = false

  instances = 4

  # ... network, OS disk, source image, etc.
}
```

These are common best practices for GPU/HPC deployments on Azure. Consider applying them at VMSS creation time to reduce allocation friction.

---

*This is a personal blog. Opinions and recommendations are my own, not Microsoft's.*
