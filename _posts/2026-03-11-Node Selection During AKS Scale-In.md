---
layout: single
title: "Node Selection During AKS Scale-In"
author_profile: false
---

When scaling down an AKS node pool, which node gets deleted? The answer depends on how you trigger the scale-in and whether any nodes have been cordoned beforehand. We set up a test cluster and systematically explored every combination:

| Scenario | Scale-In Method | Cordon First? |
|----------|----------------|---------------|
| **A** | `az aks nodepool scale` (CLI) | No |
| **B** | `az aks nodepool scale` (CLI) | Yes |
| **C** | `terraform apply` (Terraform) | Yes |
| **D** | VMSS scale-in policy inspection | N/A |
| **E** | `az aks nodepool delete-machines` | N/A |

Each cordon test ran three iterations — cordoning the oldest, middle, and newest node respectively — then checked which node was actually deleted.

---

## Setup

| Component | Details |
|-----------|---------|
| **Cluster** | AKS v1.33.7, East US |
| **System pool** | 2x Standard_D4ads_v5 |
| **GPU pool** | 4x Standard_ND96isr_H100_v5 (H100) |
| **Workload** | 8 nginx replicas, topologySpreadConstraints = 2 per node |
| **IaC** | Terraform, azurerm provider v4.64.0 |

---

## Results

### Test A: Baseline — CLI Scale-Down, No Cordon

Scaled from 4 to 3 with no cordon applied.

| Before | After | Deleted |
|--------|-------|---------|
| vmss000000, vmss000001, vmss000002, vmss000003 | vmss000000, vmss000001, vmss000002 | **vmss000003** (highest ID) |

The node with the highest VMSS instance ID was deleted. This is consistent with the Default VMSS scale-in policy (see Test D below).

### Test B: Cordon + CLI Scale-Down

Used `az aks nodepool scale` after cordoning one node:

| Iteration | Cordoned Node | Deleted Node | Match? |
|-----------|--------------|--------------|--------|
| 1 (oldest) | vmss000001 | vmss000001 | **YES** |
| 2 (middle) | vmss000004 | vmss000004 | **YES** |
| 3 (newest) | vmss000006 | vmss000006 | **YES** |

**3/3 — cordoned node was always deleted**, regardless of its position (oldest, middle, or newest). The cordon overrides the default "highest instance ID" behavior.

### Test C: Cordon + Terraform Scale-Down

Used `terraform apply -var "gpu_node_count=3"` after cordoning one node:

| Iteration | Cordoned Node | Deleted Node | Match? |
|-----------|--------------|--------------|--------|
| 1 (oldest) | vmss000002 | vmss000002 | **YES** |
| 2 (middle) | vmss000004 | vmss000004 | **YES** |
| 3 (newest) | vmss000006 | vmss000006 | **YES** |

**3/3 — same result as CLI.** Terraform ultimately calls the same ARM API, so the behavior is identical.

### Test D: The Underlying VMSS Policy

We inspected the VMSS scale-in policy AKS set on the GPU node pool:

```json
{
  "scaleInPolicy": null
}
```

A `null` scale-in policy means **"Default"** — Azure balances VMs across fault domains, then deletes the VM with the highest instance ID. This explains the Test A result and confirms that cordoning (Tests B and C) overrides this default behavior.

### Test E: Explicit Node Deletion

Used `az aks nodepool delete-machines --machine-names <node>`:

| Targeted Node | Deleted Node | Match? |
|--------------|--------------|--------|
| vmss000000 | vmss000000 | **CONFIRMED** |

This is the deterministic path — you control exactly which node is removed, no ambiguity.

---

## The Terraform Gotcha: gpu_driver Forces Replacement

During our first run of Test C, we noticed something alarming. Instead of:

```
# azurerm_kubernetes_cluster_node_pool.gpu will be updated in-place
  ~ node_count = 4 -> 3
```

We got:

```
# azurerm_kubernetes_cluster_node_pool.gpu must be replaced
-/+ resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
      - gpu_driver = "Install" -> null  # forces replacement
      ~ node_count = 4 -> 3
    }

Plan: 1 to add, 1 to change, 1 to destroy.
```

**Terraform destroyed the entire 4-node GPU pool and recreated it with 3 nodes.** Every time. This is because:

1. Azure automatically sets `gpu_driver = "Install"` on GPU node pools
2. If your Terraform config does not declare it, Terraform sees `"Install" -> null` as drift
3. `gpu_driver` is a **ForceNew** attribute in the azurerm provider — any change triggers full replacement

The fix is simple — add `gpu_driver` to your node pool resource:

```hcl
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpupool"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = "Standard_ND96isr_H100_v5"
  node_count            = var.gpu_node_count

  # CRITICAL: Without this, Terraform destroys and recreates
  # the entire pool on every apply instead of scaling in-place
  gpu_driver = "Install"
}
```

After adding this, Terraform correctly did in-place updates:

```
Plan: 0 to add, 1 to change, 0 to destroy.
```

**If you manage GPU node pools with Terraform and have not set `gpu_driver`, check your plans carefully** — you may be unknowingly destroying and rebuilding your entire pool on every change.

---

## Recommendations

### If you need deterministic control: use delete-machines

```bash
# Drain the node first (graceful pod eviction)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Delete the specific node
az aks nodepool delete-machines \
    --resource-group <rg> \
    --cluster-name <cluster> \
    --name <nodepool> \
    --machine-names <node-name>
```

This is the **only fully deterministic** path. You choose exactly which node goes away.

### If you want to guide scale-in without explicit targeting: cordon first

```bash
# Mark the node as unschedulable
kubectl cordon <node-name>

# Then scale down (CLI or Terraform)
az aks nodepool scale --resource-group <rg> --cluster-name <cluster> \
    --name <nodepool> --node-count <N-1>
```

Our tests show this **works consistently** — AKS deletes the cordoned node. However:

- This behavior is **not documented** in Azure official docs as a guarantee
- It likely works because AKS/VMSS considers unschedulable nodes as preferred candidates for removal
- We would not recommend relying on this for production-critical workflows without explicit confirmation from the AKS team

### For Terraform users: pin gpu_driver

Always include `gpu_driver = "Install"` in GPU node pool resources. Without it, any `terraform apply` that touches the node pool will destroy and recreate it — losing all nodes, draining all pods, and creating a completely new VMSS.

Run `terraform plan` and look for `-/+ must be replaced` before every apply.

---

## Summary

| Scenario | Which node gets deleted? |
|----------|------------------------|
| Scale-in, no cordon | Highest VMSS instance ID (Default policy) |
| Scale-in, one node cordoned (CLI) | The cordoned node |
| Scale-in, one node cordoned (Terraform) | The cordoned node |
| `delete-machines` | Exact node you specify |

Cordoning a node before scaling down does reliably select it for deletion — both via CLI and Terraform. But for production use, `az aks nodepool delete-machines` is the safer, documented, and deterministic approach.
