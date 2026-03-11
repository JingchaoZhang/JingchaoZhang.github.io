#!/bin/bash
# Usage: bash launch_node.sh <num_nodes> <node_rank> <master_addr> <master_port> [sharding]
NUM_NODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
SHARDING=${5:-full}
GPUS_PER_NODE=8

echo "Launching node rank $NODE_RANK / $NUM_NODES (master=$MASTER_ADDR:$MASTER_PORT, sharding=$SHARDING)"

sudo docker run --rm \
    --gpus all --ipc=host --ulimit memlock=-1 \
    --net=host \
    --privileged \
    -v /home/azureuser/models:/models \
    -v /home/azureuser/finetune_bench.py:/bench.py \
    -v /home/azureuser/results:/results \
    -e SHARDING=$SHARDING \
    --name bench_node${NODE_RANK} \
    nvcr.io/nvidia/pytorch:24.12-py3 \
    bash -c "pip install transformers==4.47.1 -q && \
    NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN \
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        /bench.py"
