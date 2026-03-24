---
layout: single
title: "Gang Scheduling on AKS: Volcano vs Kueue vs KAI Scheduler"
author_profile: false
---

## Introduction

In my [previous post]({% post_url 2026-03-23-GPU Fault Detection with NVSentinel on Azure AKS %}), I deployed NVSentinel for GPU fault detection on AKS. That post assumed workloads were already scheduled — but *how* they get scheduled matters just as much, especially for multi-node GPU training.

The default Kubernetes scheduler has a fundamental problem for distributed training: it places pods independently. If you submit a 4-node training job, the scheduler might place 3 pods immediately and leave the 4th pending for hours — burning GPU-hours on the 3 idle workers. In SLURM, this never happens: `srun --nodes=4` waits until all 4 nodes are available, then starts everything simultaneously. This is **gang scheduling**.

Kubernetes doesn't have gang scheduling built in. Three open-source projects fill that gap: **Volcano** (CNCF, started at Huawei), **Kueue** (Kubernetes SIG, Google), and **KAI Scheduler** (NVIDIA, ex-Run:ai). Each takes a different architectural approach.

In this post, I install all three on the same AKS cluster with H100 GPU nodes and submit the same job through each. The goal isn't a benchmark — the scheduling overhead is nearly identical (~8–9 seconds). The goal is to compare the **user experience**: how you submit jobs, how you check status, how the queue systems work, and what tradeoffs each scheduler makes.

## Test Environment

| Component | Detail |
|-----------|--------|
| **Cluster** | Azure Kubernetes Service (AKS), Kubernetes v1.33.7 |
| **GPU Node Pool** | 2× Standard_ND96isr_H100_v5 (8× H100 per node, 16 GPUs total) |
| **System Node Pool** | 2× Standard_D4ads_v5 |
| **GPU Operator** | NVIDIA GPU Operator (driver pre-installed on AKS) |
| **Volcano** | v1.14.1 (Helm chart) |
| **Kueue** | v0.16.4 (manifest install) |
| **KAI Scheduler** | v0.13.4 (Helm chart) |

All three schedulers run simultaneously on the same cluster. Each uses a different `schedulerName`, so there's no conflict — a pod specifies which scheduler should place it.

## How Each Scheduler Works

Before looking at the test results, it helps to understand the architectural difference:

```
VOLCANO:
  Pod created → volcano scheduler picks it up → places it
  (replaces kube-scheduler for those pods)
  Gang: PodGroup CRD tracks which pods must start together

KUEUE:
  Job created with suspend: true → Kueue holds it in queue →
  when quota allows, Kueue unsuspends → default kube-scheduler places pods
  (works WITH the default scheduler, not instead of it)
  Gang: all pods start when job is unsuspended

KAI:
  Pod created → kai-scheduler picks it up → places it
  (replaces kube-scheduler for those pods)
  Gang: pod-grouper auto-creates PodGroups
```

The key distinction: **Kueue is a queue manager, not a scheduler.** It decides *when* a job should start, but the default Kubernetes scheduler decides *where* pods go. Volcano and KAI replace the scheduler entirely — they control both *when* and *where*.

## The Test Job

A simple 2-pod job, each requesting 1 GPU, running `nvidia-smi` and sleeping 30 seconds. Identical workload, three different submission methods.

### Volcano

Volcano uses the standard Kubernetes `batch/v1 Job` with a pod annotation to assign the queue:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: volcano-test
spec:
  parallelism: 2
  completions: 2
  completionMode: Indexed
  template:
    metadata:
      annotations:
        scheduling.volcano.sh/queue-name: "default"
    spec:
      schedulerName: volcano      # ← Volcano's scheduler
      containers:
        - name: gpu
          image: nvidia/cuda:12.6.0-base-ubuntu22.04
          command: ["sh", "-c", "nvidia-smi -L && sleep 30"]
          resources:
            limits:
              nvidia.com/gpu: "1"
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      restartPolicy: Never
```

### Kueue

Kueue uses a standard Job with a label and `suspend: true`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: kueue-test
  labels:
    kueue.x-k8s.io/queue-name: gpu-queue  # ← Kueue queue assignment
spec:
  parallelism: 2
  completions: 2
  completionMode: Indexed
  suspend: true                            # ← Kueue will unsuspend when ready
  template:
    spec:
      # No schedulerName — uses default kube-scheduler
      containers:
        - name: gpu
          image: nvidia/cuda:12.6.0-base-ubuntu22.04
          command: ["sh", "-c", "nvidia-smi -L && sleep 30"]
          resources:
            limits:
              nvidia.com/gpu: "1"
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      restartPolicy: Never
```

### KAI Scheduler

KAI uses a standard Job with a queue label and `schedulerName`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: kai-test
  labels:
    kai.scheduler/queue: team-training     # ← KAI queue assignment
spec:
  parallelism: 2
  completions: 2
  completionMode: Indexed
  template:
    metadata:
      labels:
        kai.scheduler/queue: team-training # ← must also be on pod template
    spec:
      schedulerName: kai-scheduler         # ← KAI's scheduler
      containers:
        - name: gpu
          image: nvidia/cuda:12.6.0-base-ubuntu22.04
          command: ["sh", "-c", "nvidia-smi -L && sleep 30"]
          resources:
            limits:
              nvidia.com/gpu: "1"
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      restartPolicy: Never
```

## Results

Each scheduler was installed in isolation — install, test, uninstall, pause 10 seconds, repeat — to ensure no interference. Same job, same cluster, clean state each time.

| | **Volcano** | **Kueue** | **KAI** |
|---|---|---|---|
| Schedule time | **~9s** | **~7s** | **~8s** |
| Total time | ~37s | ~36s | ~35s |
| Worker 0 node | vmss000001 | vmss000001 | vmss000000 |
| Worker 1 node | vmss000000 | vmss000001 | vmss000000 |
| Placement strategy | **Spread** | **Bin-packed** | **Bin-packed** |

Scheduling overhead is nearly identical — 7–9 seconds from `kubectl apply` to pods running. The 30-second `sleep` dominates the total time.

The placement difference is the real finding:

- **Volcano spread** the pods across both GPU nodes (one per node)
- **Kueue placed both pods on the same node** (vmss000001)
- **KAI placed both pods on the same node** (vmss000000)

This is consistent and reproducible — not an artifact of running schedulers side-by-side. Each scheduler was the *only* scheduler on the cluster during its test.

For distributed training with FSDP/NCCL, you typically *want* pods spread across nodes (to use all available GPUs and InfiniBand links). For independent inference jobs, bin-packing is better (keeps one node free for other work). Both Kueue and KAI support spread scheduling via topology-aware features, but their defaults favor bin-packing. Volcano spreads by default.

### What About Real Training Jobs?

The 1-GPU-per-pod test revealed default placement preferences. But in practice, LLM training requests **all 8 GPUs per node** — and at that point, the scheduler *must* spread because two 8-GPU pods can't fit on the same 8-GPU node.

To confirm, I ran Qwen2.5-7B fine-tuning with 8 GPUs per worker through Kueue:

```yaml
resources:
  limits:
    nvidia.com/gpu: "8"   # all GPUs on the node
```

```
Pod placement:
qwen-finetune-kueue-0   aks-gpupool-09344442-vmss000001   8 GPUs
qwen-finetune-kueue-1   aks-gpupool-09344442-vmss000000   8 GPUs
```

Both workers land on separate nodes — the scheduler has no choice. Training completed in ~57 seconds per worker (5 steps, Qwen2.5-7B, bf16, wikitext-2):

```
{'loss': 0.7141, 'grad_norm': 46.75, 'learning_rate': 5e-05, 'epoch': 0.31}
{'loss': 0.3203, 'grad_norm': 28.5, 'learning_rate': 4e-05, 'epoch': 0.62}
{'loss': 0.0184, 'grad_norm': 1.546875, 'learning_rate': 3e-05, 'epoch': 0.92}
{'train_runtime': 56.4s, 'train_samples_per_second': 2.836}
[Kueue] Rank 0 finished training in 56.6s
```

The bin-pack vs spread distinction only matters for **partial-GPU jobs** (inference, dev notebooks, small experiments). For full-node training — the primary use case for H100 clusters — all three schedulers behave identically.

## Queue Setup Comparison

Before submitting jobs, each scheduler needs queue infrastructure. The setup complexity varies:

### Volcano — Minimal

Volcano creates a `default` queue automatically during Helm install. No additional setup needed for basic use.

```bash
# Already exists after helm install
kubectl get queues.scheduling.volcano.sh
# NAME      PARENT
# default   root
# root
```

### Kueue — Most Verbose

Kueue requires three CRDs: a `ResourceFlavor` (what kind of hardware), a `ClusterQueue` (capacity limits), and a `LocalQueue` (namespace-scoped entry point):

```yaml
# 1. What hardware flavors exist
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: gpu-h100
spec:
  nodeLabels:
    nvidia.com/gpu.present: "true"

# 2. Cluster-wide capacity
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: gpu-cluster-queue
spec:
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: gpu-h100
          resources:
            - name: "nvidia.com/gpu"
              nominalQuota: 16

# 3. Namespace entry point
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: gpu-queue
  namespace: default
spec:
  clusterQueue: gpu-cluster-queue
```

More setup, but the separation provides fine-grained multi-tenant control.

### KAI — Hierarchical

KAI uses a two-level queue hierarchy (parent + child):

```yaml
apiVersion: scheduling.run.ai/v2
kind: Queue
metadata:
  name: department-gpu
spec:
  resources:
    gpu:
      quota: 16

---
apiVersion: scheduling.run.ai/v2
kind: Queue
metadata:
  name: team-training
spec:
  parentQueue: department-gpu
  resources:
    gpu:
      quota: 16
```

## Command Cheat Sheet

| Operation | Volcano | Kueue | KAI |
|-----------|---------|-------|-----|
| **Submit** | `kubectl apply -f job.yaml` | `kubectl apply -f job.yaml` | `kubectl apply -f job.yaml` |
| **Queue assignment** | Pod annotation: `scheduling.volcano.sh/queue-name` | Job label: `kueue.x-k8s.io/queue-name` | Pod label: `kai.scheduler/queue` |
| **Scheduler** | `schedulerName: volcano` | (default scheduler) | `schedulerName: kai-scheduler` |
| **Gang mechanism** | `minAvailable` or auto PodGroup | `suspend: true` → unsuspend | Auto PodGroup via pod-grouper |
| **Job status** | `kubectl get vcjob` or `kubectl get job` | `kubectl get workloads` | `kubectl get job` |
| **Queue status** | `kubectl get queues.scheduling.volcano.sh` | `kubectl get clusterqueues` | `kubectl get queues.scheduling.run.ai` |
| **Pod logs** | `kubectl logs -l job-name=<name>` | `kubectl logs -l job-name=<name>` | `kubectl logs -l app=<name>` |
| **Cancel** | `kubectl delete job <name>` | `kubectl delete job <name>` | `kubectl delete job <name>` |

## SLURM Equivalents

For those coming from SLURM (like me), here's the mental mapping:

| SLURM | Volcano | Kueue | KAI |
|-------|---------|-------|-----|
| `sbatch` | `kubectl apply -f job.yaml` | same | same |
| `squeue` | `kubectl get vcjob` | `kubectl get workloads` | `kubectl get pods -l app=<name>` |
| `scancel` | `kubectl delete job` | same | same |
| `scontrol show job` | `kubectl describe vcjob` | `kubectl describe workload` | `kubectl describe job` |
| Partition | Queue (`scheduling.volcano.sh`) | ClusterQueue + LocalQueue | Queue (`scheduling.run.ai`) |
| QOS / Priority | PriorityClass | WorkloadPriorityClass | `priorityClassName` label |
| Fair-share | DRF plugin | Cohort-based fair sharing | Time-based fairshare |
| `--nodes=N` (gang) | `minAvailable: N` | `parallelism: N` + `suspend: true` | `parallelism: N` (auto pod group) |

The biggest UX gap vs. SLURM: there's no single `squeue` equivalent that shows all pending jobs across all schedulers. Each scheduler has its own status command.

## Which One Should You Use?

**Volcano** if you're coming from HPC/SLURM and want the closest mental model. It has the widest ecosystem (Spark, MPI, PyTorch, Kubeflow), 6 years of production use, and CNCF backing. The VolcanoJob CRD feels most like `sbatch`. Azure ML on AKS uses Volcano under the hood.

**Kueue** if you want minimal disruption to your existing K8s setup. It doesn't replace the scheduler — it just adds a queueing layer. This means you keep all existing scheduler plugins, topology rules, and pod affinity/anti-affinity. Best for teams that already have a mature K8s platform and want to add batch job management.

**KAI** if you're running NVIDIA GPUs and want GPU-specific features: fractional GPU sharing, MIG-aware scheduling, NVLink topology-aware placement, and integration with the NVIDIA ecosystem (NVSentinel, GPU Operator, Run:ai). Youngest of the three but backed by NVIDIA's production experience.

All three are free, open-source, and actively maintained. You can even run all three on the same cluster (as I did) and let different teams choose their preferred scheduler.

## Gotcha: Volcano `schedulerName`

One issue I hit: Volcano's Helm chart registers the scheduler as `volcano`, not `volcano-scheduler`. If you use the wrong name, pods stay `Pending` forever with no error message — the Volcano scheduler simply never sees them.

```yaml
# ✗ Wrong — pods will be Pending forever
schedulerName: volcano-scheduler

# ✓ Correct
schedulerName: volcano
```

Also, the annotation-based approach (standard K8s Jobs with `scheduling.volcano.sh/queue-name` on the pod template) is more reliable than the VolcanoJob CRD for gang scheduling on newer Kubernetes versions.

## Reproducing These Results

The cluster setup uses Terraform (AKS with `Standard_ND96isr_H100_v5` GPU node pool) and Helm for each scheduler. The key install commands:

```bash
# Volcano
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace

# Kueue
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.4/manifests.yaml

# KAI
helm install kai-scheduler \
  oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler \
  -n kai-scheduler --create-namespace --version v0.13.4
```

The job manifests shown in the [Test Job](#the-test-job) section are complete and copy-pasteable — adjust the `tolerations` and GPU resource requests for your cluster.

---

*This is a personal blog. Opinions and recommendations are my own, not Microsoft's.*
