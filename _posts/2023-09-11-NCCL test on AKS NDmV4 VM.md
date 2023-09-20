---
layout: single
author_profile: false
---

This write-up aims to replicate the blog [Deploy NDm_v4 (A100) Kubernetes Cluster](https://techcommunity.microsoft.com/t5/azure-high-performance-computing/deploy-ndm-v4-a100-kubernetes-cluster/ba-p/3838871) by [Cormac Garvey](https://techcommunity.microsoft.com/t5/user/viewprofilepage/user-id/364170). The original blog assumes you have an exising ACR.

All following commands run on your local laptop, except for the NCCL docker container creation step, which needs to run on a NDmv4 VM. 
## Login to your az account
```bash
az login
az account set -s YourSubscription
```

## Add AKS extension and enable IB
```bash
az extension add --name aks-preview
az feature register --name AKSInfinibandSupport --namespace Microsoft.ContainerService
```

## Define environment variables
```bash
export AKS_RG='JZ-AKS'
export LOCATION='southcentralus'
export NODE_RG='JZ-AKSnode'
export AKS_NAME='JZ-akscluster'
export AGENT_POOL_NAME='jzpool' #lower case letter and number only
export ACR_NAME='jzacr2' #lower case letter and number only
export NDMv4_POOL_NAME='jzndmv4' #lower case letter and number only
```

## Create a resource group
```bash
az group create --resource-group $AKS_RG --location $LOCATION
```

## Create Azure Container Registry (ACR)
```bash
az acr create --resource-group $AKS_RG --name $ACR_NAME --sku Standard
```
Without this step, the follwoing create AKS cluster command with `--attach-acr` will fail. 

## Create NCCL container (this step needs to be done on a NDmv4 VM, not your local environment)
Login to ACR
```
az login
az account set -s YourSubscription
az acr login -n $ACR_NAME # az acr login -n jzacr2; DO NOT use the full "loginServer" name: "jzacr2.azurecr.io"
```

Create first file `nccl-tests.sh`, and `chmod +x nccl-tests.sh`
```bash
#!/bin/bash

git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/usr/local/mpi
```

Create second file `ndv4-topo.xml`
```bash
<system version="1">
  <cpu numaid="0" affinity="0000ffff,0000ffff" arch="x86_64" vendor="AuthenticAMD" familyid="23" modelid="49">
    <pci busid="ffff:ff:01.0" class="0x060400" link_speed="16 GT/s" link_width="16">
      <pci busid="0001:00:00.0" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0101:00:00.0" class="0x020700" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0002:00:00.0" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0102:00:00.0" class="0x020700" link_speed="16 GT/s" link_width="16"/>
    </pci>
  </cpu>
  <cpu numaid="1" affinity="0000ffff,0000ffff" arch="x86_64" vendor="AuthenticAMD" familyid="23" modelid="49">
    <pci busid="ffff:ff:02.0" class="0x060400" link_speed="16 GT/s" link_width="16">
      <pci busid="0003:00:00.0" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0103:00:00.0" class="0x020700" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0004:00:00.0" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0104:00:00.0" class="0x020700" link_speed="16 GT/s" link_width="16"/>
    </pci>
  </cpu>
  <cpu numaid="2" affinity="0000ffff,0000ffff" arch="x86_64" vendor="AuthenticAMD" familyid="23" modelid="49">
      <pci busid="ffff:ff:03.0" class="0x060400" link_speed="16 GT/s" link_width="16">
      <pci busid="000b:00:00.0" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0105:00:00.0" class="0x020700" link_speed="16 GT/s" link_width="16"/>
      <pci busid="000c:00:00.0" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0106:00:00.0" class="0x020700" link_speed="16 GT/s" link_width="16"/>
    </pci>
  </cpu>
  <cpu numaid="3" affinity="0000ffff,0000ffff" arch="x86_64" vendor="AuthenticAMD" familyid="23" modelid="49">
    <pci busid="ffff:ff:04.0" class="0x060400" link_speed="16 GT/s" link_width="16">
      <pci busid="000d:00:00.0" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0107:00:00.0" class="0x020700" link_speed="16 GT/s" link_width="16"/>
      <pci busid="000e:00:00.0" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0108:00:00.0" class="0x020700" link_speed="16 GT/s" link_width="16"/>
    </pci>
  </cpu>
</system>
```

Create third file `Dockerfile`
```bash
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.03-py3

FROM ${FROM_IMAGE_NAME}

RUN apt update
RUN apt-get -y install build-essential
RUN apt-get -y install infiniband-diags
RUN apt-get -y install openssh-server
RUN apt-get -y install kmod
COPY nccl-tests.sh .
RUN ./nccl-tests.sh
COPY ndv4-topo.xml .
```


Put above three files in the same directory, then build and push to ACR. 
```bash
docker build -t jzacr2.azurecr.io/pytorch_nccl_tests_2303 .
docker push jzacr2.azurecr.io/pytorch_nccl_tests_2303:latest
```

## Create AKS cluster (now back to your local laptop)
```bash
az aks create \
      -g $AKS_RG \
      --node-resource-group $NODE_RG \
      -n $AKS_NAME \
      --enable-managed-identity \
      --node-count 2 \
      --generate-ssh-keys \
      -l $LOCATION \
      --node-vm-size Standard_D2s_v3 \
      --nodepool-name $AGENT_POOL_NAME \
      --os-sku Ubuntu \
      --attach-acr $ACR_NAME
```

## Add a node pool
```bash
az aks nodepool add --resource-group $AKS_RG --cluster-name $AKS_NAME --name $NDMv4_POOL_NAME --node-count 1 --node-vm-size Standard_ND96amsr_A100_v4 --node-osdisk-size 128 --os-sku Ubuntu --tags SkipGPUDriverInstallation=true
or
az aks nodepool add --resource-group $AKS_RG --cluster-name $AKS_NAME --name $NDMv4_POOL_NAME --node-count 1 --node-vm-size Standard_ND96amsr_A100_v4 --node-osdisk-size 128 --os-sku Ubuntu --tags SkipGPUDriverInstall=true
```
Note: Need to verify which tag is right. The blog has the second one. I tested the first one which worked.

## Save the credentials to your local config file
```bash
$ az aks get-credentials --overwrite-existing --resource-group $AKS_RG --name $AKS_NAME
Merged "JZ-akscluster" as current context in /home/jingchao/.kube/config
```

## Check the created nodes
```
$ kubectl get nodes
NAME                              STATUS   ROLES   AGE    VERSION
aks-jzndmv4-29195301-vmss000000   Ready    agent   135m   v1.26.6
aks-jzpool-33093035-vmss000000    Ready    agent   153m   v1.26.6
aks-jzpool-33093035-vmss000001    Ready    agent   153m   v1.26.6
```

## Install GPU and network drivers
Save the following script to a script `driver.sh`, and execute it with `bash driver.sh`
```bash
#! /bin/bash

# Apply required manifests
kubectl get namespace nvidia-operator 2>/dev/null || kubectl create namespace nvidia-operator

# Install node feature discovery
helm upgrade -i --wait \
  -n nvidia-operator node-feature-discovery node-feature-discovery \
  --repo https://kubernetes-sigs.github.io/node-feature-discovery/charts \
  --set-json master.nodeSelector='{"kubernetes.azure.com/mode": "system"}' \
  --set-json worker.nodeSelector='{"kubernetes.azure.com/accelerator": "nvidia"}' \
  --set-json worker.config.sources.pci.deviceClassWhitelist='["02","03","0200","0207"]' \
  --set-json worker.config.sources.pci.deviceLabelFields='["vendor"]'

# Install the network-operator
helm upgrade -i --wait \
  -n nvidia-operator network-operator network-operator \
  --repo https://helm.ngc.nvidia.com/nvidia \
  --set deployCR=true \
  --set nfd.enabled=false \
  --set ofedDriver.deploy=true \
  --set rdmaSharedDevicePlugin.deploy=false \
  --set secondaryNetwork.deploy=true \
  --set secondaryNetwork.ipamPlugin.deploy=true \
  --set secondaryNetwork.ipoib.deploy=true \
  --set secondaryNetwork.multus.deploy=true \
  --set sriovDevicePlugin.deploy=true \
  --set-json sriovDevicePlugin.resources='[{"name":"mlnxnics","linkTypes": ["infiniband"], "vendors":["15b3"]}]'
# Note: use --set ofedDriver.version="<MOFED VERSION>"
#       to install a specific MOFED version
#
# Install the gpu-operator
helm upgrade -i --wait \
  -n nvidia-operator gpu-operator gpu-operator \
  --repo https://helm.ngc.nvidia.com/nvidia \
  --set nfd.enabled=false \
  --set driver.enabled=true \
  --set driver.version="525.60.13" \
  --set driver.rdma.enabled=true \
  --set toolkit.enabled=true

# Apply the hostdev-net configuration for Infiniband
cat <<EOF | kubectl apply -f -
apiVersion: mellanox.com/v1alpha1
kind: HostDeviceNetwork
metadata:
   name: hostdev-net
spec:
  networkNamespace: "default"
  resourceName: "mlnxnics"
  ipam: |
    {
      "type": "whereabouts",
      "datastore": "kubernetes",
      "kubernetes": {
        "kubeconfig": "/etc/cni/net.d/whereabouts.d/whereabouts.kubeconfig"
      },
      "range": "100.127.0.0/16",
      "exclude": [],
      "log_file" : "/var/log/whereabouts.log",
      "log_level" : "info"
    }
EOF
```

## Verify the drivers are installed
```bash
$ kubectl describe node $NDmv4_AKS_node | grep -e "nvidia.com/mlnxnics" -e "nvidia.com/gpu"
                    nvidia.com/gpu-driver-upgrade-state=upgrade-done
                    nvidia.com/gpu.compute.major=8
                    nvidia.com/gpu.compute.minor=0
                    nvidia.com/gpu.count=8
                    nvidia.com/gpu.deploy.container-toolkit=true
                    nvidia.com/gpu.deploy.dcgm=true
                    nvidia.com/gpu.deploy.dcgm-exporter=true
                    nvidia.com/gpu.deploy.device-plugin=true
                    nvidia.com/gpu.deploy.driver=true
                    nvidia.com/gpu.deploy.gpu-feature-discovery=true
                    nvidia.com/gpu.deploy.mig-manager=true
                    nvidia.com/gpu.deploy.node-status-exporter=true
                    nvidia.com/gpu.deploy.nvsm=
                    nvidia.com/gpu.deploy.operator-validator=true
                    nvidia.com/gpu.family=ampere
                    nvidia.com/gpu.machine=Virtual-Machine
                    nvidia.com/gpu.memory=81920
                    nvidia.com/gpu.present=true
                    nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB
                    nvidia.com/gpu.replicas=1
                    nvidia.com/gpu-driver-upgrade-enabled: true
  nvidia.com/gpu:       8
  nvidia.com/mlnxnics:  8
  nvidia.com/gpu:       8
  nvidia.com/mlnxnics:  8
  nvidia.com/gpu       0           0
  nvidia.com/mlnxnics  0           0
```

## Install Volcano Kubernetes scheduler
```bash
$ kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/release-1.7/installer/volcano-development.yaml
$ kubectl get all -n volcano-system
NAME                                      READY   STATUS      RESTARTS   AGE
pod/volcano-admission-7b864f5d49-x8bv9    1/1     Running     0          129m
pod/volcano-admission-init-pb7nr          0/1     Completed   0          129m
pod/volcano-controllers-5d784c876-hxmdz   1/1     Running     0          129m
pod/volcano-scheduler-65fb9b4dd-5pmhm     1/1     Running     0          129m

NAME                                TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)    AGE 
service/volcano-admission-service   ClusterIP   10.0.104.73   <none>        443/TCP    129m
service/volcano-scheduler-service   ClusterIP   10.0.8.41     <none>        8080/TCP   129m

NAME                                  READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/volcano-admission     1/1     1            1           129m
deployment.apps/volcano-controllers   1/1     1            1           129m
deployment.apps/volcano-scheduler     1/1     1            1           129m

NAME                                            DESIRED   CURRENT   READY   AGE
replicaset.apps/volcano-admission-7b864f5d49    1         1         1       129m
replicaset.apps/volcano-controllers-5d784c876   1         1         1       129m
replicaset.apps/volcano-scheduler-65fb9b4dd     1         1         1       129m

NAME                               COMPLETIONS   DURATION   AGE
job.batch/volcano-admission-init   1/1           8s         129m
```

## Scale GPU nodes to 2
```bash
az aks nodepool scale --resource-group $AKS_RG --cluster-name $AKS_NAME --name $NDMv4_POOL_NAME --node-count 2
```

## Create a kubernetes service account to view the output
```bash
kubectl create serviceaccount -n default mpi-worker-view
kubectl create rolebinding default-view --namespace default --serviceaccount default:mpi-worker-view --clusterrole view
```

## Create the NCCL job
Create the NCCL job file `job.yaml` with content below:
```bash
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: nccl-allreduce-job1
spec:
  minAvailable: 3
  schedulerName: volcano
  plugins:
    ssh: []
    svc: []
  tasks:
    - replicas: 1
      name: mpimaster
      policies:
        - event: TaskCompleted
          action: CompleteJob
      template:
        spec:
          initContainers:
            - command:
                - /bin/bash
                - -c
                - |
                  until [[ "$(kubectl get pod -l volcano.sh/job-name=nccl-allreduce-job1,volcano.sh/task-spec=mpiworker -o json | jq '.items | length')" != 0 ]]; do
                    echo "Waiting for MPI worker pods..."
                    sleep 3
                  done
                  echo "Waiting for MPI worker pods to be ready..."
                  kubectl wait pod -l volcano.sh/job-name=nccl-allreduce-job1,volcano.sh/task-spec=mpiworker --for=condition=Ready --timeout=600s
              image: mcr.microsoft.com/oss/kubernetes/kubectl:v1.26.3
              name: wait-for-workers
          serviceAccount: mpi-worker-view
          containers:
            - command:
                - /bin/bash
                - -c
                - |
                  MPI_HOST=$(cat /etc/volcano/mpiworker.host | tr "\n" ",")
                  mkdir -p /var/run/sshd; /usr/sbin/sshd
                  echo "HOSTS: $MPI_HOST"
                  mpirun --allow-run-as-root \
                  -np 16 -npernode 8 \
                  --bind-to numa --map-by ppr:8:node \
                  -hostfile /etc/volcano/mpiworker.host \
                  -x NCCL_DEBUG=info \
                  -x UCX_TLS=tcp \
                  -x NCCL_TOPO_FILE=/workspace/ndv4-topo.xml \
                  -x UCX_NET_DEVICES=eth0 \
                  -x CUDA_DEVICE_ORDER=PCI_BUS_ID \
                  -x NCCL_SOCKET_IFNAME=eth0 \
                  -mca coll_hcoll_enable 0 \
                  /workspace/nccl-tests/build/all_reduce_perf -b 8 -f 2 -g 1 -e 8G -c 1 \
                  | tee /home/re
              image: jzacr2.azurecr.io/pytorch_nccl_tests_2303:latest
              securityContext:
                capabilities:
                  add: ["IPC_LOCK"]
              name: mpimaster
              ports:
                - containerPort: 22
                  name: mpijob-port
              workingDir: /workspace
              resources:
                requests:
                  cpu: 1
          restartPolicy: OnFailure
    - replicas: 2
      name: mpiworker
      template:
        metadata:
          annotations:
            k8s.v1.cni.cncf.io/networks: hostdev-net,hostdev-net,hostdev-net,hostdev-net,hostdev-net,hostdev-net,hostdev-net,hostdev-net
        spec:
          containers:
            - command:
                - /bin/bash
                - -c
                - |
                  mkdir -p /var/run/sshd; /usr/sbin/sshd -D;
              image: jzacr2.azurecr.io/pytorch_nccl_tests_2303:latest
              securityContext:
                capabilities:
                  add: ["IPC_LOCK"]
              name: mpiworker
              ports:
                - containerPort: 22
                  name: mpijob-port
              workingDir: /workspace
              resources:
                requests:
                  cpu: 1
          restartPolicy: OnFailure
    - replicas: 2
      name: mpiworker
      template:
        metadata:
          annotations:
            k8s.v1.cni.cncf.io/networks: hostdev-net,hostdev-net,hostdev-net,hostdev-net,hostdev-net,hostdev-net,hostdev-net,hostdev-net
        spec:
          containers:
            - command:
                - /bin/bash
                - -c
                - |
                  mkdir -p /var/run/sshd; /usr/sbin/sshd -D;
              image: jzacr2.azurecr.io/pytorch_nccl_tests_2303:latest
              securityContext:
                capabilities:
                  add: ["IPC_LOCK"]
              name: mpiworker
              ports:
                - containerPort: 22
                  name: mpijob-port
              workingDir: /workspace
              resources:
                requests:
                  nvidia.com/gpu: 8
                  nvidia.com/mlnxnics: 8
                limits:
                  nvidia.com/gpu: 8
                  nvidia.com/mlnxnics: 8
              volumeMounts:
              - mountPath: /dev/shm
                name: shm
          restartPolicy: OnFailure
          terminationGracePeriodSeconds: 0
          volumes:
          - name: shm
            emptyDir:
              medium: Memory
              sizeLimit: 8Gi
---
```
Note: there are two occurances of `jzacr2.azurecr.io/pytorch_nccl_tests_2303:latest` in the above script, which is the NCCL container you pushed to your ACR. Edit it before proceeding. 

## Submit the NCCL job
```bash
$ kubectl apply -f job.yaml 
job.batch.volcano.sh/nccl-allreduce-job1 created
```

## Get the pod name
```bash
$ kubectl get pods
NAME                              READY   STATUS    RESTARTS   AGE
nccl-allreduce-job1-mpimaster-0   1/1     Running   0          16s
nccl-allreduce-job1-mpiworker-0   1/1     Running   0          16s
nccl-allreduce-job1-mpiworker-1   1/1     Running   0          16s
```

## Check the NCCL test output
```bash
$ kubectl logs -f nccl-allreduce-job1-mpimaster-0
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
nccl-allreduce-job1-mpiworker-1:57:214 [7] NCCL INFO comm 0x55c13a6640d0 rank 15 nranks 16 cudaDev 7 busId e00000 commId 0x46cbea2567b59372 - Init COMPLETE
nccl-allreduce-job1-mpiworker-1:51:174 [3] NCCL INFO comm 0x5586358b6c10 rank 11 nranks 16 cudaDev 3 busId 400000 commId 0x46cbea2567b59372 - Init COMPLETE
nccl-allreduce-job1-mpiworker-1:49:169 [1] NCCL INFO comm 0x5590048f9910 rank 9 nranks 16 cudaDev 1 busId 200000 commId 0x46cbea2567b59372 - Init COMPLETE
nccl-allreduce-job1-mpiworker-1:53:231 [5] NCCL INFO comm 0x564e5f765c40 rank 13 nranks 16 cudaDev 5 busId c00000 commId 0x46cbea2567b59372 - Init COMPLETE
nccl-allreduce-job1-mpiworker-1:54:194 [6] NCCL INFO comm 0x564950a5b020 rank 14 nranks 16 cudaDev 6 busId d00000 commId 0x46cbea2567b59372 - Init COMPLETE
nccl-allreduce-job1-mpiworker-1:50:212 [2] NCCL INFO comm 0x555b01ca9170 rank 10 nranks 16 cudaDev 2 busId 300000 commId 0x46cbea2567b59372 - Init COMPLETE
nccl-allreduce-job1-mpiworker-1:48:168 [0] NCCL INFO comm 0x55a22905c240 rank 8 nranks 16 cudaDev 0 busId 100000 commId 0x46cbea2567b59372 - Init COMPLETE
nccl-allreduce-job1-mpiworker-1:52:197 [4] NCCL INFO comm 0x55567f894360 rank 12 nranks 16 cudaDev 4 busId b00000 commId 0x46cbea2567b59372 - Init COMPLETE
           8             2     float     sum      -1    37.30    0.00    0.00      0    34.44    0.00    0.00      0
          16             4     float     sum      -1    36.03    0.00    0.00      0    33.94    0.00    0.00      0
          32             8     float     sum      -1    36.50    0.00    0.00      0    33.57    0.00    0.00      0
          64            16     float     sum      -1    36.33    0.00    0.00      0    33.99    0.00    0.00      0
         128            32     float     sum      -1    37.62    0.00    0.01      0    34.42    0.00    0.01      0
         256            64     float     sum      -1    38.28    0.01    0.01      0    34.77    0.01    0.01      0
         512           128     float     sum      -1    38.20    0.01    0.03      0    35.15    0.01    0.03      0
        1024           256     float     sum      -1    40.92    0.03    0.05      0    37.37    0.03    0.05      0
        2048           512     float     sum      -1    42.87    0.05    0.09      0    39.49    0.05    0.10      0
        4096          1024     float     sum      -1    41.82    0.10    0.18      0    40.85    0.10    0.19      0
        8192          2048     float     sum      -1    46.31    0.18    0.33      0    42.78    0.19    0.36      0
       16384          4096     float     sum      -1    58.10    0.28    0.53      0    55.03    0.30    0.56      0
       32768          8192     float     sum      -1    58.73    0.56    1.05      0    56.11    0.58    1.09      0
       65536         16384     float     sum      -1    60.01    1.09    2.05      0    59.40    1.10    2.07      0
      131072         32768     float     sum      -1    63.71    2.06    3.86      0    63.33    2.07    3.88      0
      262144         65536     float     sum      -1    68.25    3.84    7.20      0    68.67    3.82    7.16      0
      524288        131072     float     sum      -1    80.23    6.54   12.25      0    79.70    6.58   12.33      0
     1048576        262144     float     sum      -1    96.39   10.88   20.40      0    96.73   10.84   20.33      0
     2097152        524288     float     sum      -1    128.6   16.31   30.59      0    127.8   16.41   30.77      0
     4194304       1048576     float     sum      -1    148.1   28.32   53.11      0    146.5   28.62   53.67      0
     8388608       2097152     float     sum      -1    211.1   39.74   74.51      0    207.8   40.37   75.70      0
    16777216       4194304     float     sum      -1    333.4   50.32   94.35      0    330.8   50.72   95.10      0
    33554432       8388608     float     sum      -1    615.6   54.51  102.21      0    626.3   53.58  100.45      0
    67108864      16777216     float     sum      -1    932.6   71.96  134.92      0    929.6   72.19  135.36      0
   134217728      33554432     float     sum      -1   1672.7   80.24  150.45      0   1676.3   80.07  150.13      0
   268435456      67108864     float     sum      -1   3013.5   89.08  167.02      0   3004.6   89.34  167.52      0
   536870912     134217728     float     sum      -1   5702.0   94.15  176.54      0   5705.8   94.09  176.42      0
  1073741824     268435456     float     sum      -1    11063   97.05  181.98      0    11089   96.83  181.56      0
  2147483648     536870912     float     sum      -1    21637   99.25  186.10      0    21673   99.09  185.79      0
  4294967296    1073741824     float     sum      -1    42758  100.45  188.34      0    42779  100.40  188.25      0
  8589934592    2147483648     float     sum      -1    85129  100.90  189.20      0    85091  100.95  189.28      0
nccl-allreduce-job1-mpiworker-1:51:51 [3] NCCL INFO comm 0x5586358b6c10 rank 11 nranks 16 cudaDev 3 busId 400000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-0:51:51 [3] NCCL INFO comm 0x563b9a846840 rank 3 nranks 16 cudaDev 3 busId 400000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-1:57:57 [7] NCCL INFO comm 0x55c13a6640d0 rank 15 nranks 16 cudaDev 7 busId e00000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-1:53:53 [5] NCCL INFO comm 0x564e5f765c40 rank 13 nranks 16 cudaDev 5 busId c00000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-0:50:50 [2] NCCL INFO comm 0x55ce61480260 rank 2 nranks 16 cudaDev 2 busId 300000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-0:52:52 [4] NCCL INFO comm 0x5632e283bb30 rank 4 nranks 16 cudaDev 4 busId b00000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-0:48:48 [0] NCCL INFO comm 0x55d407b24020 rank 0 nranks 16 cudaDev 0 busId 100000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-1:50:50 [2] NCCL INFO comm 0x555b01ca9170 rank 10 nranks 16 cudaDev 2 busId 300000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-0:55:55 [6] NCCL INFO comm 0x55dc04852d60 rank 6 nranks 16 cudaDev 6 busId d00000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-0:49:49 [1] NCCL INFO comm 0x555ead805480 rank 1 nranks 16 cudaDev 1 busId 200000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-1:48:48 [0] NCCL INFO comm 0x55a22905c240 rank 8 nranks 16 cudaDev 0 busId 100000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-0:56:56 [7] NCCL INFO comm 0x556f8d65b050 rank 7 nranks 16 cudaDev 7 busId e00000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-1:49:49 [1] NCCL INFO comm 0x5590048f9910 rank 9 nranks 16 cudaDev 1 busId 200000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-1:54:54 [6] NCCL INFO comm 0x564950a5b020 rank 14 nranks 16 cudaDev 6 busId d00000 - Destroy COMPLETE
nccl-allreduce-job1-mpiworker-0:53:53 [5] NCCL INFO comm 0x556afbcbdc10 rank 5 nranks 16 cudaDev 5 busId c00000 - Destroy COMPLETE
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 57.347
#
nccl-allreduce-job1-mpiworker-1:52:52 [4] NCCL INFO comm 0x55567f894360 rank 12 nranks 16 cudaDev 4 busId b00000 - Destroy COMPLETE
```
If you see ~189 GBps output then you are done with this exercise. 
