---
layout: single
title: "Azure Managed Lustre for GPU VMSS Clusters"
author_profile: false
---

## The Problem: Every Node Needs the Same Data

When you scale a GPU cluster beyond a single node, you immediately hit a file distribution problem. Model weights, training data, and checkpoints need to be accessible from every node. The typical workarounds — `scp`-ing files to each VM, downloading from blob storage on each boot, or running NFS on one of your GPU nodes — all break down as you scale. `scp` doesn't scale past a handful of nodes, blob downloads waste GPU-hour billing while you wait, and NFS on a single VM becomes a throughput bottleneck.

What you really want is a shared POSIX file system that every node can mount at boot — fast enough to feed multi-node training, and managed so you don't have to babysit storage servers.

[Azure Managed Lustre File System (AMLFS)](https://learn.microsoft.com/en-us/azure/azure-managed-lustre/amlfs-overview) is exactly that. It's a fully managed Lustre deployment that lives in your VNet, presents a standard mount point, and can push up to 500 MBps per TiB of provisioned capacity. In this post, I walk through deploying an 8-node H100 VMSS cluster with an 8 TiB AMLFS — from VNet design to a working `/lustre` mount on every node.

## Architecture Overview

The setup has three main pieces:

1. **VNet with two subnets** — one for the GPU VM Scale Set, one dedicated to AMLFS
2. **Azure Managed Lustre File System** — deployed into its own subnet via ARM template
3. **VMSS with cloud-init** — installs the Lustre client and mounts the file system at first boot

```
┌─────────────────────────────────────────────────────┐
│  VNet: 10.0.0.0/16                                  │
│                                                     │
│  ┌─────────────────────┐  ┌──────────────────────┐  │
│  │ VMSS Subnet         │  │ AMLFS Subnet         │  │
│  │ 10.0.0.0/23         │  │ 10.0.2.0/24          │  │
│  │                     │  │                       │  │
│  │  ND96isr_H100_v5    │  │  Azure Managed Lustre │  │
│  │  × 8 nodes          │  │  8 TiB                │  │
│  │  (8× H100 each)     │  │  2 GB/s throughput    │  │
│  └─────────────────────┘  └──────────────────────┘  │
│            │                        │                │
│            └──── mount -t lustre ───┘                │
│                  → /lustre                           │
└─────────────────────────────────────────────────────┘
```

## Step 1: VNet and Subnet Design

AMLFS requires its own dedicated subnet — it cannot share a subnet with VMs. The subnet needs to be large enough for the Lustre infrastructure (a /24 is sufficient) and must not have any Network Security Groups that would block Lustre traffic.

```bash
# Create VNet
az network vnet create \
  --resource-group "$RG_NAME" \
  --name "$VNET_NAME" \
  --address-prefix "10.0.0.0/16" \
  --location "eastus"

# Subnet for GPU VMs (10.0.0.0/23 = 512 addresses, enough for large VMSS)
az network vnet subnet create \
  --resource-group "$RG_NAME" \
  --vnet-name "$VNET_NAME" \
  --name "vmss-subnet" \
  --address-prefix "10.0.0.0/23"

# Dedicated subnet for AMLFS (10.0.2.0/24 = 256 addresses)
az network vnet subnet create \
  --resource-group "$RG_NAME" \
  --vnet-name "$VNET_NAME" \
  --name "amlfs-subnet" \
  --address-prefix "10.0.2.0/24"
```

I use a /23 for the VMSS subnet because each ND96isr_H100_v5 gets multiple NICs (one primary + IB interfaces), and Azure VMSS with Flexible orchestration can consume addresses quickly. A /24 would limit you to ~250 VMs, which is fine for most GPU clusters, but /23 gives headroom.

## Step 2: Deploy AMLFS via ARM Template

There is no single `az` CLI command to create an AMLFS — you need an ARM template. Here's the minimal template:

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "resources": [
    {
      "type": "Microsoft.StorageCache/amlFileSystems",
      "apiVersion": "2024-03-01",
      "name": "my-amlfs",
      "location": "eastus",
      "zones": ["2"],
      "sku": {
        "name": "AMLFS-Durable-Premium-250"
      },
      "properties": {
        "storageCapacityTiB": 8,
        "filesystemSubnet": "<AMLFS_SUBNET_RESOURCE_ID>",
        "maintenanceWindow": {
          "dayOfWeek": "Saturday",
          "timeOfDayUTC": "02:00"
        }
      }
    }
  ],
  "outputs": {
    "mgsAddress": {
      "type": "string",
      "value": "[reference(resourceId('Microsoft.StorageCache/amlFileSystems', 'my-amlfs')).clientInfo.mgsAddress]"
    }
  }
}
```

Key details:

- **`zones` is required.** AMLFS must be pinned to a specific availability zone. If one zone is at capacity, try another — I hit `OverconstrainedZonalAllocationRequestFailure` in zone 1 and succeeded with zone 2.
- **`filesystemSubnet`** must be the full resource ID of the dedicated AMLFS subnet, e.g., `/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Network/virtualNetworks/<vnet>/subnets/<subnet>`.
- **`storageCapacityTiB`** must follow the SKU's increment rules (see table below).

Deploy it:

```bash
# Get the subnet resource ID
AMLFS_SUBNET_ID=$(az network vnet subnet show \
  --resource-group "$RG_NAME" \
  --vnet-name "$VNET_NAME" \
  --name "amlfs-subnet" \
  --query "id" --output tsv)

# Deploy (takes 10-15 minutes)
az deployment group create \
  --name "amlfs-deploy" \
  --resource-group "$RG_NAME" \
  --template-file amlfs_template.json \
  --no-wait

# Wait for completion
az deployment group wait \
  --name "amlfs-deploy" \
  --resource-group "$RG_NAME" \
  --created

# Retrieve the MGS IP address
AMLFS_MGS_IP=$(az deployment group show \
  --name "amlfs-deploy" \
  --resource-group "$RG_NAME" \
  --query "properties.outputs.mgsAddress.value" --output tsv)

echo "MGS IP: $AMLFS_MGS_IP"
```

### AMLFS SKU Reference

| SKU | Throughput/TiB | Min Size | Max Size | Increment |
|-----|---------------|----------|----------|-----------|
| AMLFS-Durable-Premium-40 | 40 MBps | 48 TiB | 768 TiB | 48 TiB |
| AMLFS-Durable-Premium-125 | 125 MBps | 16 TiB | 128 TiB | 16 TiB |
| AMLFS-Durable-Premium-250 | 250 MBps | 8 TiB | 128 TiB | 8 TiB |
| AMLFS-Durable-Premium-500 | 500 MBps | 4 TiB | 128 TiB | 4 TiB |

For my use case — storing model weights and checkpoints for multi-node training — 8 TiB at the 250 tier gives 2 GB/s aggregate throughput, which is more than enough. If you're streaming large training datasets, the 500 tier at 4 TiB minimum gives you the same 2 GB/s in a smaller footprint.

## Step 3: Mount Lustre on Every Node via Cloud-Init

Once AMLFS is deployed and you have the MGS IP, each VM needs two things: the Lustre client package and a mount command. I handle both in cloud-init so every VMSS node comes up ready:

```yaml
#cloud-config
runcmd:
  # Install Lustre client for Ubuntu 22.04 HPC
  - |
    source /etc/lsb-release
    echo "deb [arch=amd64] https://packages.microsoft.com/repos/amlfs-${DISTRIB_CODENAME}/ ${DISTRIB_CODENAME} main" \
      | tee /etc/apt/sources.list.d/amlfs.list
    curl -sL https://packages.microsoft.com/keys/microsoft.asc \
      | gpg --dearmor | tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null
    apt update
    apt install -y amlfs-lustre-client-2.15.7-33-g79ddf99=$(uname -r)

  # Mount AMLFS
  - mkdir -p /lustre
  - mount -t lustre -o noatime,flock 10.0.2.5@tcp:/lustrefs /lustre

  # Persist across reboots
  - echo "10.0.2.5@tcp:/lustrefs /lustre lustre noatime,flock,_netdev 0 0" >> /etc/fstab
```

A few things to note:

- **The Lustre client version must match your kernel.** The `=$(uname -r)` suffix ensures the package matches the running kernel. The `microsoft-dsvm:ubuntu-hpc:2204` image ships a kernel that's compatible with `amlfs-lustre-client-2.15.7-33-g79ddf99`.
- **`noatime`** avoids unnecessary metadata updates on reads. **`flock`** enables POSIX file locking, which some training frameworks need.
- **`_netdev`** in fstab tells systemd to wait for the network before attempting the mount on reboot.

The MGS IP (`10.0.2.5` in my case) comes from the ARM deployment output. In my deploy script, I template this into cloud-init by substituting placeholders before passing it to `az vmss create --custom-data`.

## Step 4: Verify

SSH into any node and check:

```bash
azureuser@vmssWFXAVI:~$ df -h /lustre/
Filesystem              Size  Used Avail Use% Mounted on
10.0.2.5@tcp:/lustrefs  7.9T  1.3M  7.5T   1% /lustre
```

The mount is live on all 8 nodes. Any file written to `/lustre` on one node is immediately visible on every other node — no copying, no syncing.

```bash
# From node 0: write a file
echo "hello from node 0" > /lustre/test.txt

# From node 5: read it immediately
cat /lustre/test.txt
# hello from node 0
```

For the GPU training use case, this means:
- **Download the model once** to `/lustre/models/` — all nodes see it
- **Write checkpoints** to `/lustre/checkpoints/` — any node can resume
- **Share scripts and configs** without `scp`

## Lessons Learned

**AMLFS requires a zone.** I initially tried deploying without the `zones` field, hoping Azure would place it automatically. It won't — the API returns `InvalidParameter: Please specify a single availability zone`. This means you also need to be aware of zonal capacity. My first deployment failed in zone 1 with `OverconstrainedZonalAllocationRequestFailure`; zone 2 worked immediately.

**AMLFS needs its own subnet.** Don't try to put it in the same subnet as your VMs. The deployment will fail. A /24 is plenty.

**The Lustre client package name is specific.** It's version-locked to both the Lustre release and the kernel version. If you update your VM image and the kernel changes, the install will fail silently unless you check for it. I add a fallback log message in cloud-init to catch this.

**Deployment takes 10-15 minutes.** Plan your automation around this — I deploy AMLFS with `--no-wait`, then use `az deployment group wait --created` to block until it's ready before generating cloud-init and creating the VMSS.

## When to Use AMLFS vs. Alternatives

| Option | Throughput | Shared? | Managed? | Best For |
|--------|-----------|---------|----------|----------|
| **AMLFS** | Up to 500 MBps/TiB | Yes (POSIX) | Yes | Multi-node training, shared model weights |
| **NFS on VM** | Limited by single VM NIC | Yes (NFS) | No | Small clusters, prototyping |
| **Local NVMe** | Highest | No | N/A | Single-node scratch, per-node caching |

AMLFS hits the sweet spot for GPU clusters: it's POSIX-compatible (so `torch.save()` and `huggingface-cli download` just work), it scales throughput with capacity, and you don't have to manage any infrastructure. The main trade-off is cost — Lustre is more expensive per TiB than blob storage, so use it for active working sets and archive to blob when you're done.

## Summary

Deploying AMLFS with a GPU VMSS cluster requires:

1. A VNet with a **dedicated subnet** for AMLFS
2. An **ARM template** deployment with an explicit **availability zone**
3. **Cloud-init** that installs the Lustre client and mounts the file system at boot

Once it's up, every node in the cluster shares a high-throughput POSIX file system — no file copying, no NFS bottleneck, no manual syncing. For multi-node GPU training at scale, it eliminates one of the most annoying operational headaches.
