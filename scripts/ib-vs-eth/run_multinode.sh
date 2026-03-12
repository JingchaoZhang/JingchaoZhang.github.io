#!/bin/bash
# Multi-node fine-tuning benchmark orchestrator for AMLFS cluster
# Usage: bash run_multinode.sh <num_nodes> <ib_disable> <model_path> [seq_len] [batch_size] [steps]

NUM_NODES=${1:?Usage: run_multinode.sh <num_nodes> <ib_disable> <model_path> [seq_len] [batch_size] [steps]}
IB_DISABLE=${2:-0}
MODEL_PATH=${3:-/lustre/models/Qwen2.5-7B}
SEQ_LEN=${4:-2048}
BATCH_SIZE=${5:-1}
STEPS=${6:-20}
MASTER_PORT=29500

# Read node list from hostfile
mapfile -t ALL_NODES < ~/hostfile_good
HEAD_NODE="${ALL_NODES[0]}"
MASTER_ADDR=$(getent ahosts "$HEAD_NODE" | grep STREAM | head -1 | awk '{print $1}')

# Take first N nodes
NODES=("${ALL_NODES[@]:0:$NUM_NODES}")
MODEL_NAME=$(basename "$MODEL_PATH")
IB_LABEL=$( [ "$IB_DISABLE" = "0" ] && echo "IB" || echo "ETH" )

echo "=== Fine-tuning Benchmark ==="
echo "Config: ${MODEL_NAME} | ${NUM_NODES} nodes | ${IB_LABEL}"
echo "Nodes: ${NODES[*]}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT} (${HEAD_NODE})"
echo "IB_DISABLE=${IB_DISABLE} | seq_len=${SEQ_LEN} | bs=${BATCH_SIZE} | steps=${STEPS}"
echo ""

LOGDIR="/lustre/results/${MODEL_NAME}_${NUM_NODES}nodes_${IB_LABEL}"
mkdir -p "$LOGDIR"

# AGGRESSIVE PRE-CLEANUP: kill containers, processes, free port on ALL participating nodes
echo "Pre-cleanup: killing stale containers and processes on all ${NUM_NODES} nodes..."
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    CLEANUP_CMD="sudo docker rm -f bench_node${i} 2>/dev/null; sudo pkill -f torchrun.*finetune_bench 2>/dev/null; sudo fuser -k ${MASTER_PORT}/tcp 2>/dev/null; true"
    if [ "$NODE" = "$(hostname)" ]; then
        eval "$CLEANUP_CMD"
    else
        ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$NODE" "$CLEANUP_CMD" 2>/dev/null
    fi
done
sleep 2
echo "Pre-cleanup done."

# Launch workers (non-head nodes) in background
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    RANK=$i
    if [ "$NODE" != "$(hostname)" ]; then
        echo "Launching worker on ${NODE} (rank ${RANK})..."
        ssh -o StrictHostKeyChecking=no "$NODE" \
            "nohup bash /lustre/scripts/launch_node.sh $NUM_NODES $RANK $MASTER_ADDR $MASTER_PORT $IB_DISABLE $MODEL_PATH $SEQ_LEN $BATCH_SIZE $STEPS > ${LOGDIR}/node${RANK}.log 2>&1 &"
        sleep 2
    fi
done

# Launch head node (rank 0) in foreground
echo "Launching head node $(hostname) (rank 0)..."
bash /lustre/scripts/launch_node.sh $NUM_NODES 0 $MASTER_ADDR $MASTER_PORT $IB_DISABLE $MODEL_PATH $SEQ_LEN $BATCH_SIZE $STEPS 2>&1 | tee "${LOGDIR}/node0.log"
HEAD_EXIT=$?

echo ""
echo "=== Head node finished (exit code: $HEAD_EXIT) ==="

# POST-CLEANUP: ensure containers are removed on all nodes
echo "Post-cleanup: removing containers on all nodes..."
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    CLEANUP_CMD="sudo docker rm -f bench_node${i} 2>/dev/null; sudo fuser -k ${MASTER_PORT}/tcp 2>/dev/null; true"
    if [ "$NODE" = "$(hostname)" ]; then
        eval "$CLEANUP_CMD"
    else
        ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$NODE" "$CLEANUP_CMD" 2>/dev/null
    fi
done
echo "Post-cleanup done."
exit $HEAD_EXIT
