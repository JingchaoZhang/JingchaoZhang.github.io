#!/bin/bash
# Usage: bash launch_node.sh <num_nodes> <node_rank> <master_addr> <master_port> <ib_disable> <model_path> [seq_len] [batch_size] [steps]
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

# Timeout: 15 minutes should be plenty (2-node completes in ~4 min)
TIMEOUT_SEC=900

echo "Launching node rank $NODE_RANK / $NUM_NODES"
echo "  master=$MASTER_ADDR:$MASTER_PORT IB_DISABLE=$IB_DISABLE"
echo "  model=$MODEL_PATH seq_len=$SEQ_LEN bs=$BATCH_SIZE steps=$STEPS"
echo "  timeout=${TIMEOUT_SEC}s"

# Use NCCL_DEBUG=INFO on first attempt to capture transport/topology details
NCCL_DEBUG_LEVEL=${NCCL_DEBUG_LEVEL:-WARN}

timeout $TIMEOUT_SEC sudo docker run --rm \
    --gpus all --ipc=host --ulimit memlock=-1 \
    --net=host \
    --privileged \
    -v /opt/microsoft:/opt/microsoft \
    -v /lustre:/lustre \
    -e NCCL_IB_DISABLE=$IB_DISABLE \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e NCCL_DEBUG=$NCCL_DEBUG_LEVEL \
    -e NCCL_TOPO_FILE=/opt/microsoft/ndv5/topo.xml \
    -e NCCL_IB_PCI_RELAXED_ORDERING=1 \
    -e NCCL_TIMEOUT=300 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --name bench_node${NODE_RANK} \
    nvcr.io/nvidia/pytorch:24.12-py3 \
    bash -c "pip install transformers==4.47.1 -q && \
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --rdzv_conf timeout=300 \
        /lustre/scripts/finetune_bench.py \
            --model_path $MODEL_PATH \
            --seq_len $SEQ_LEN \
            --batch_size $BATCH_SIZE \
            --steps $STEPS"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
    echo "ERROR: Timed out after ${TIMEOUT_SEC}s - likely NCCL hang"
    sudo docker kill bench_node${NODE_RANK} 2>/dev/null
    sudo docker rm bench_node${NODE_RANK} 2>/dev/null
fi
exit $EXIT_CODE
