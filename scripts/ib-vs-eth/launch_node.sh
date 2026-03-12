#!/bin/bash
# Usage: bash launch_node.sh <num_nodes> <node_rank> <master_addr> <master_port> <ib_disable> <model_path> [seq_len] [batch_size] [steps]
#
# Key fix: NCCL_IB_DISABLE alone doesn't work with NCCL 2.28+ external RDMA plugin.
# When ib_disable=1, we set NCCL_NET=Socket to force TCP sockets (real Ethernet mode).

NUM_NODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
IB_DISABLE=${5:-0}
MODEL_PATH=${6:-/lustre/models/Qwen2.5-7B}
SEQ_LEN=${7:-2048}
BATCH_SIZE=${8:-1}
STEPS=${9:-20}
GPUS_PER_NODE=8

# Build NCCL network mode arguments
# NCCL_IB_DISABLE alone is ignored by the external RDMA network plugin in NCCL 2.28+.
# Must use NCCL_NET=Socket to force TCP sockets for true Ethernet-only mode.
if [ "$IB_DISABLE" = "1" ]; then
    NCCL_NET_ARGS="-e NCCL_NET=Socket -e NCCL_IB_DISABLE=1"
    NET_LABEL="Socket (Ethernet)"
else
    NCCL_NET_ARGS="-e NCCL_IB_DISABLE=0"
    NET_LABEL="RDMA (IB)"
fi

echo "Launching node rank $NODE_RANK / $NUM_NODES"
echo "  master=$MASTER_ADDR:$MASTER_PORT Network=$NET_LABEL"
echo "  model=$MODEL_PATH seq_len=$SEQ_LEN bs=$BATCH_SIZE steps=$STEPS"

sudo docker run --rm \
    --gpus all --ipc=host --ulimit memlock=-1 \
    --net=host \
    --privileged \
    -v /opt/microsoft:/opt/microsoft \
    -v /lustre:/lustre \
    $NCCL_NET_ARGS \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e NCCL_DEBUG=WARN \
    -e NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml \
    -e NCCL_IB_PCI_RELAXED_ORDERING=1 \
    --name bench_node${NODE_RANK} \
    nvcr.io/nvidia/pytorch:24.12-py3 \
    bash -c "pip install transformers==4.47.1 -q && \
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        /lustre/scripts/finetune_bench.py \
            --model_path $MODEL_PATH \
            --seq_len $SEQ_LEN \
            --batch_size $BATCH_SIZE \
            --steps $STEPS"
