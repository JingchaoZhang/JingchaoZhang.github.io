---
layout: single
title: "GPU Fault Detection with NVSentinel on Azure AKS"
author_profile: false
---

## Introduction

In my [previous posts]({% post_url 2026-02-04-IB vs Ethernet Fine-Tuning at Scale %}), I benchmarked InfiniBand vs. Ethernet for distributed training and built a [monitoring pipeline]({% post_url 2026-02-19-Monitoring InfiniBand Health on Azure H100 Clusters %}) for InfiniBand health on Azure H100 clusters. Those posts answered two questions: *how fast is the network?* and *is the network healthy?*

This post tackles the next question: **when a GPU fails, what happens?**

The answer, for most teams, is "someone gets paged, investigates for 20 minutes, cordons the node, drains the workloads, and files a support ticket." On a 4-node training job, a single faulty GPU can idle 31 healthy GPUs while a human figures out what went wrong. At $32/hr per H100, that's $992/hr of wasted compute.

[NVSentinel](https://github.com/NVIDIA/NVSentinel) is NVIDIA's open-source answer to this problem. It's a GPU fault detection and remediation system for Kubernetes that automatically detects hardware faults via DCGM, cordons the faulty node, drains workloads, and triggers repair workflows — all without human intervention. It [reached v1.0.0](https://github.com/NVIDIA/NVSentinel/releases/tag/v1.0.0) in March 2026, covering 40,000+ GPUs across AWS, GCP, Azure, OCI, and bare metal.

In this post, I deploy NVSentinel on an Azure AKS cluster with `Standard_ND96isr_H100_v5` GPU nodes, enable the full remediation pipeline, inject a GPU fault, and measure the end-to-end response. The complete pipeline — from fault injection to node cordon — completes in **~20 seconds**.

## Test Environment

| Component | Detail |
|-----------|--------|
| **Cluster** | Azure Kubernetes Service (AKS), Kubernetes v1.33.7 |
| **GPU Node Pool** | 2× Standard_ND96isr_H100_v5 |
| **GPUs per node** | 8× NVIDIA H100 80 GB HBM3 |
| **System Node Pool** | 2× Standard_D4ads_v5 |
| **Region** | East US |
| **Kubernetes** | 1.30.x (AKS managed) |
| **GPU Operator** | NVIDIA GPU Operator (latest) |
| **NVSentinel** | v1.0.0 |
| **Storage backend** | MongoDB (in-cluster, single replica with change streams) |
| **Monitoring** | Prometheus + Grafana (kube-prometheus-stack) |
| **NVIDIA Driver** | 580.126.09 |
| **CUDA** | 13.0 |

## What NVSentinel Does

NVSentinel is a set of independent Kubernetes microservices that coordinate through MongoDB change streams and the Kubernetes API. No module communicates directly with another. The pipeline has four stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPU Node (DaemonSet)                     │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────────┐               │
│  │  GPU Health       │    │  Syslog Health        │              │
│  │  Monitor (DCGM)   │    │  Monitor (journalctl) │              │
│  └────────┬─────────┘    └────────┬─────────────┘               │
│           │ gRPC                   │ gRPC                        │
└───────────┼────────────────────────┼────────────────────────────┘
            │                        │
            ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NVSentinel Core                            │
│                                                                 │
│  ┌───────────────────┐     ┌────────────────────┐               │
│  │ Platform           │────▶│ MongoDB             │              │
│  │ Connectors (gRPC)  │     │ (Event Store)       │              │
│  └───────────────────┘     └─────────┬──────────┘               │
│                                      │ change streams            │
│               ┌──────────────────────┼──────────────────┐       │
│               ▼                      ▼                  ▼       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐   │
│  │ Fault Quarantine  │  │ Node Drainer     │  │ Fault       │   │
│  │ (CEL rules →      │  │ (graceful        │  │ Remediation │   │
│  │  cordon node)     │  │  eviction)       │  │ (CRD→reboot)│   │
│  └──────────────────┘  └──────────────────┘  └─────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Stage 1 — Detection.** DaemonSet pods on every GPU node run continuous health checks. The GPU health monitor polls DCGM for thermal issues, ECC errors, XID events, and InfoROM corruption. The syslog health monitor watches journalctl for kernel panics, driver crashes, and NVLink errors.

**Stage 2 — Classification.** Platform connectors receive health events via gRPC, validate them, persist them to MongoDB, and update Kubernetes node conditions. Each event is tagged with severity (fatal, warning, info), component class (GPU, NVLink, NVSwitch), and error codes.

**Stage 3 — Quarantine.** The fault quarantine module watches MongoDB change streams. When a fatal event arrives, it evaluates CEL (Common Expression Language) rules — for example, `event.agent == 'gpu-health-monitor' && event.isFatal == true` — and cordons the node. A circuit breaker prevents mass quarantines: if >50% of nodes would be cordoned, the circuit breaker trips and logs a warning instead.

**Stage 4 — Drain & Remediation.** The node drainer gracefully evicts workloads with per-namespace eviction strategies. After drain completes, fault remediation creates a maintenance CRD (e.g., `RebootNode`), and the janitor executes it via the Azure API.

The entire pipeline is event-driven. There's no polling loop between stages — MongoDB change streams deliver events in real time.

## Deployment on Azure AKS

### Step 1: Create the AKS Cluster

The cluster uses Terraform. The key decisions:

- **GPU node pool**: 2× `Standard_ND96isr_H100_v5` with a `nvidia.com/gpu=present:NoSchedule` taint to keep system pods off GPU nodes
- **System node pool**: 2× `Standard_D4ads_v5` for NVSentinel core, MongoDB, Prometheus, and cert-manager
- **Network**: Azure CNI with Azure network policy (required for NVSentinel's network policies)

```hcl
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpupool"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = "Standard_ND96isr_H100_v5"
  node_count            = 2
  os_disk_size_gb       = 256

  node_labels = {
    "nvidia.com/gpu.present" = "true"
    "workload-type"          = "gpu"
  }

  node_taints = [
    "nvidia.com/gpu=present:NoSchedule"
  ]
}
```

```bash
# Create cluster
./01_create_cluster.sh
```

### Step 2: Install Prerequisites

Three Helm charts before NVSentinel:

1. **cert-manager** (v1.19.1) — NVSentinel uses TLS for gRPC communication between health monitors and platform connectors.
2. **kube-prometheus-stack** — NVSentinel exposes Prometheus metrics for every module. PodMonitor CRDs auto-discover them.
3. **NVIDIA GPU Operator** — Provides DCGM, the NVIDIA driver, and the device plugin. This is the foundation: without DCGM, the GPU health monitor has nothing to poll.

```bash
# Install all three
./02_install_prereqs.sh
```

After the GPU Operator finishes (~10 minutes for driver compilation), verify:

```bash
$ kubectl get nodes -l nvidia.com/gpu.present=true \
    -o custom-columns='NAME:.metadata.name,GPU:.status.capacity.nvidia\.com/gpu'
NAME                                 GPU
aks-gpupool-12345678-vmss000000      8
aks-gpupool-12345678-vmss000001      8
```

Each node reports 8 allocatable GPUs. DCGM pods are running in the `gpu-operator` namespace.

### Step 3: Install NVSentinel

The default `values.yaml` enables only health monitoring — safe for observation. For the full pipeline, I use a custom values file that enables every stage:

```yaml
# values/nvsentinel-azure.yaml (key sections)
global:
  gpuHealthMonitor:
    enabled: true
  syslogHealthMonitor:
    enabled: true
  faultQuarantine:
    enabled: true       # ← auto-cordon
  nodeDrainer:
    enabled: true       # ← graceful eviction
  faultRemediation:
    enabled: true       # ← maintenance CRD
  janitor:
    enabled: true       # ← execute via Azure API
  mongodbStore:
    enabled: true       # ← required for change streams

fault-quarantine:
  circuitBreaker:
    enabled: true
    percentage: 50      # don't cordon more than half the cluster

janitor:
  csp:
    provider: "azure"
    azure:
      subscriptionId: "..."
      resourceGroup: "..."
```

```bash
helm upgrade --install nvsentinel oci://ghcr.io/nvidia/nvsentinel \
    --namespace nvsentinel --create-namespace \
    --version v1.0.0 \
    -f values/nvsentinel-azure.yaml \
    --timeout 15m --wait
```

### What Gets Deployed

After installation, the `nvsentinel` namespace contains:

| Component | Kind | Replicas | Description |
|-----------|------|----------|-------------|
| `platform-connectors` | Deployment | 1 | gRPC server, event persistence, K8s node conditions |
| `fault-quarantine` | Deployment | 1 | CEL rule engine, cordons nodes |
| `node-drainer` | Deployment | 1 | Graceful workload eviction |
| `fault-remediation` | Deployment | 1 | Creates maintenance CRDs |
| `health-events-analyzer` | Deployment | 1 | Event pattern detection |
| `labeler` | Deployment | 1 | Auto-labels nodes with driver versions |
| `gpu-health-monitor` | DaemonSet | 2 (one per GPU node) | DCGM health checks |
| `syslog-health-monitor` | DaemonSet | 2 (one per GPU node) | Journalctl log analysis |
| `metadata-collector` | DaemonSet | 2 (one per GPU node) | GPU/NVSwitch topology |
| `mongodb` | StatefulSet | 3 | Event store with change streams |

```
$ kubectl get pods -n nvsentinel
NAME                                         READY   STATUS    RESTARTS   AGE
fault-quarantine-545b88d4d8-6smjr            1/1     Running   0          5m
fault-remediation-7db46f67d9-cblfr           1/1     Running   0          5m
gpu-health-monitor-dcgm-4.x-75brz            1/1     Running   0          5m
gpu-health-monitor-dcgm-4.x-76ffl            1/1     Running   0          5m
health-events-analyzer-67774776b4-f5mjq      1/1     Running   0          5m
janitor-5978d8cd84-bbk72                     1/1     Running   0          5m
janitor-provider-699597fb6b-l6vdc            1/1     Running   0          5m
kubernetes-object-monitor-7ddc746495-65mhq   1/1     Running   0          5m
labeler-5c6cfd75db-dn47x                     1/1     Running   0          5m
mongodb-0                                    2/2     Running   0          5m
node-drainer-56d599d584-sz2th                1/1     Running   0          5m
platform-connectors-b8x5q                    1/1     Running   0          5m
platform-connectors-h6wrj                    1/1     Running   0          5m
platform-connectors-pcxr6                    1/1     Running   0          5m
platform-connectors-zrrt2                    1/1     Running   0          5m
```

## Validation

Before testing faults, confirm the healthy baseline:

```bash
$ ./04_validate.sh

>>> Namespace & Pods
  ✓ nvsentinel namespace exists
  ✓ deployment/fault-quarantine exists
  ✓ deployment/node-drainer exists
  ✓ deployment/fault-remediation exists
  ✓ deployment/health-events-analyzer exists
  ✓ deployment/labeler exists

>>> GPU Nodes
  GPU nodes: 2

  --- aks-gpupool-40476767-vmss000000 ---
    ✓ DCGM pod: nvidia-dcgm-t4dmw
    ✓ GPU health monitor: gpu-health-monitor-dcgm-4.x-75brz
    Node conditions (33 total):
      GpuInforomWatch: False (No Health Failures)
      GpuNvlinkWatch: False (No Health Failures)
      GpuThermalWatch: False (No Health Failures)
      GpuMemWatch: False (No Health Failures)
      ... all healthy
```

Both GPU nodes are monitored. DCGM is reachable. No health events yet — the cluster is healthy.

## Fault Injection Test

The real test: inject a GPU fault and see if NVSentinel automatically quarantines the node.

DCGM supports error injection via `dcgmi test --inject`. We inject field 84 (InfoROM Valid) with value 0 to simulate a corrupt InfoROM — a fatal GPU hardware fault. The GPU health monitor detects this through its normal DCGM polling cycle, identical to how a real fault would be caught.

```bash
# Inject on the DCGM pod running on the target node
kubectl exec -n gpu-operator "$DCGM_POD" -c nvidia-dcgm-ctr -- \
    dcgmi test --inject --gpuid 0 -f 84 -v 0

Target node: aks-gpupool-40476767-vmss000000

>>> Pre-test state
--- Schedulable? ---
  unschedulable: false

>>> Injecting GPU fault (InfoROM corruption)...
Successfully injected field info.

Fault injected. Waiting for NVSentinel to detect and quarantine...
  Waiting... (0s / 180s)
  Waiting... (5s / 180s)
  Waiting... (10s / 180s)
  Waiting... (15s / 180s)

  ✓ Node aks-gpupool-40476767-vmss000000 has been CORDONED by NVSentinel!
  Time to quarantine: ~20 seconds
```

**~20 seconds** from fault injection to node cordon. This includes DCGM's polling interval, gRPC delivery, MongoDB persistence, change stream propagation, and CEL rule evaluation.

### What Happened Under the Hood

Tracing the event through the logs:

**1. GPU Health Monitor** detects the fault via DCGM health check:

```
Overriding action from COMPONENT_RESET to RESTART_VM for aks-gpupool-40476767-vmss000000
Updated cache for key GpuInforomWatch|GPU|0 with value
  EntityCacheEntry(active_errors={'DCGM_FR_CORRUPT_INFOROM'}) after successful send
```

**2. Platform Connectors** updates the Kubernetes node condition:

```
GpuInforomWatch: True
  ErrorCode:DCGM_FR_CORRUPT_INFOROM GPU:0
  A corrupt InfoROM has been detected in GPU 0.
  Flash the InfoROM to clear this corruption.
  Recommended Action=RESTART_VM;
```

**3. Fault Quarantine** evaluates CEL rules and cordons:

```
INFO Evaluating NodeRuleEvaluator for node
INFO Cordoning node
INFO Setting annotations on node
INFO Adding labels on node
INFO Document updated with status
INFO Node quarantine duration
```

After quarantine, the node shows `SchedulingDisabled` and carries NVSentinel annotations:

```bash
$ kubectl get nodes
NAME                              STATUS                     ROLES   AGE   VERSION
aks-gpupool-40476767-vmss000000   Ready,SchedulingDisabled   <none>  36m   v1.33.7
aks-gpupool-40476767-vmss000001   Ready                      <none>  36m   v1.33.7
aks-system-33979574-vmss000000    Ready                      <none>  44m   v1.33.7
aks-system-33979574-vmss000001    Ready                      <none>  44m   v1.33.7
```

The node condition tells the full story:

```bash
$ kubectl get node aks-gpupool-40476767-vmss000000 \
    -o jsonpath='{.status.conditions[?(@.type=="GpuInforomWatch")]}' | jq .
{
  "type": "GpuInforomWatch",
  "status": "True",
  "message": "ErrorCode:DCGM_FR_CORRUPT_INFOROM GPU:0
    A corrupt InfoROM has been detected in GPU 0.
    Flash the InfoROM to clear this corruption.
    Recommended Action=RESTART_VM;"
}
```

## Prometheus Metrics

NVSentinel exposes Prometheus metrics for every stage of the pipeline via PodMonitors. Key metrics include:

| Metric | Description |
|--------|-------------|
| `nvsentinel_fault_quarantine_nodes_cordoned_total` | Nodes cordoned |
| `nvsentinel_fault_quarantine_nodes_uncordoned_total` | Nodes uncordoned |
| `nvsentinel_fault_quarantine_circuit_breaker_triggered_total` | Circuit breaker activations |
| `nvsentinel_node_drainer_nodes_drained_total` | Nodes drained |
| `nvsentinel_node_drainer_pods_evicted_total` | Pods evicted |
| `nvsentinel_platform_connectors_health_events_received_total` | Health events received via gRPC |
| `nvsentinel_gpu_health_monitor_errors_detected_total` | GPU errors detected |

These metrics feed directly into alerting. A simple PromQL alert:

{% raw %}
```yaml
- alert: NVSentinelNodeCordoned
  expr: increase(nvsentinel_fault_quarantine_nodes_cordoned_total[5m]) > 0
  for: 0m
  labels:
    severity: critical
  annotations:
    summary: "NVSentinel cordoned a GPU node"
    description: "Node {{ $labels.node }} was quarantined due to a GPU fault."
```
{% endraw %}

## Circuit Breaker

The circuit breaker is critical for avoiding a cluster-wide outage from a false-positive cascade. With `percentage: 50` and 2 GPU nodes, NVSentinel will cordon at most 1 node (50% of 2 = 1). If a second fault arrives before the first is resolved, the circuit breaker trips:

```
level=warn msg="Circuit breaker triggered" percentage=50
  cordonedNodes=1 totalNodes=2
  msg="Refusing to cordon — would exceed 50% threshold"
```

At larger scale — say 100 GPU nodes — this means NVSentinel will never cordon more than 50 simultaneously, even if a bad driver update causes every GPU to report errors. The cooldown period (`duration: "5m"`) ensures there's time for humans to investigate before the pipeline resumes.

## Recovery

After hardware repair (or in our case, clearing the injected fault), uncordoning the node triggers NVSentinel to clean up:

```bash
$ kubectl uncordon aks-gpupool-40476767-vmss000000
node/aks-gpupool-40476767-vmss000000 uncordoned

$ kubectl get nodes
NAME                              STATUS   ROLES   AGE   VERSION
aks-gpupool-40476767-vmss000000   Ready    <none>  38m   v1.33.7
aks-gpupool-40476767-vmss000001   Ready    <none>  37m   v1.33.7
```

Quarantine annotations are removed, the node is schedulable again, and new workloads can be placed on it.

## Integration with IB Monitoring

NVSentinel's **Kubernetes Object Monitor** can watch any Kubernetes resource using CEL expressions. This means you can feed your existing IB monitoring into NVSentinel's quarantine pipeline:

```yaml
# Example: quarantine a node if an IB health ConfigMap reports errors
kubernetesObjectMonitor:
  enabled: true
  policies:
    - name: "ib-error-detection"
      resource:
        apiVersion: v1
        kind: ConfigMap
        namespace: monitoring
        labelSelector:
          matchLabels:
            app: ib-health-checker
      expression: |
        int(object.data.packet_seq_err) > 1000 || int(object.data.local_ack_timeout_err) > 0
      healthEvent:
        componentClass: "Network"
        isFatal: true
        recommendedAction: "QUARANTINE"
```

Combined with the IB hardware counter monitoring from my [previous post]({% post_url 2026-02-19-Monitoring InfiniBand Health on Azure H100 Clusters %}), this creates a full-stack GPU + network health pipeline: IB counters detect network degradation → custom exporter reports to Prometheus → Kubernetes Object Monitor evaluates CEL rules → NVSentinel quarantines the node.

## Key Takeaways

1. **NVSentinel works on Azure AKS out of the box.** The Helm chart supports Azure as a first-class CSP. Install cert-manager, GPU Operator, deploy the chart, fill in your subscription ID — done.

2. **The full pipeline is fast.** Fault detection to node cordon in ~20 seconds. For recoverable faults, GPU reset takes seconds instead of minutes for a full reboot.

3. **The circuit breaker is essential.** At any scale, you need a safety valve that prevents a monitoring system from causing more damage than the fault it detected.

4. **CEL rules are flexible.** You can quarantine based on GPU errors, syslog patterns, NVLink failures, or any Kubernetes resource. The same rule engine handles all of them.

5. **Start with monitoring only.** The default values enable only health monitors and the labeler — no cordoning, no draining. Turn on each stage as you build confidence.

6. **NVSentinel adds 33 node conditions out of the box.** From `GpuInforomWatch` to `GpuNvlinkWatch` to `GpuThermalWatch`, every GPU health dimension gets its own Kubernetes condition. This is immediately visible via `kubectl get node` — no additional dashboards needed for basic triage.

## Cost Context

The H100 nodes used in this test cost ~$32/hr each on Azure. A 4-node, 32-GPU training job costs $128/hr. If a GPU fault takes 30 minutes to detect and remediate manually, that's $64 in idle compute. NVSentinel reduces this to ~20 seconds — effectively zero cost. Over a month of continuous training with a 1% GPU failure rate (realistic for large clusters), the savings compound to thousands of dollars.

## Reproducing These Results

All scripts are in the [`Azure_AKS_NVSentinel/`](https://github.com/YOUR_REPO) directory:

```bash
# 1. Create AKS cluster with GPU nodes
./01_create_cluster.sh

# 2. Install cert-manager, Prometheus, GPU Operator
./02_install_prereqs.sh

# 3. Deploy NVSentinel with full pipeline
./03_install_nvsentinel.sh

# 4. Validate everything is running
./04_validate.sh

# 5. Inject a fault and watch NVSentinel respond
./05_test_fault_injection.sh

# 6. Test recovery
./06_test_recovery.sh

# 7. Collect metrics for analysis
./07_collect_metrics.sh

# 99. Tear everything down
./99_cleanup.sh
```

## What's Next

NVSentinel's [roadmap](https://github.com/NVIDIA/NVSentinel/blob/main/ROADMAP.md) includes preflight checks (DCGM diagnostics + NCCL all-reduce tests before workload scheduling) and enhanced GPU reset support. I plan to test both on Azure as they mature — preflight checks are particularly interesting for catching hardware issues at node provisioning time, before they waste compute during training.

The broader pattern here is worth noting: GPU clusters are becoming complex enough that they need their own reliability engineering stack, separate from general Kubernetes observability. NVSentinel fills the GPU-specific gap. Combined with IB monitoring, thermal monitoring, and NCCL diagnostics, you get a complete picture of GPU infrastructure health.

---

*This is a personal blog. Opinions and recommendations are my own, not Microsoft's.*
