#!/bin/bash
# Multi-node MoE fine-tuning benchmark orchestrator
# Usage: bash run_multinode_moe.sh <num_nodes> <ib_disable> <model_path> [seq_len] [batch_size] [steps]
# Example: bash run_multinode_moe.sh 2 0 /lustre/models/Mixtral-8x7B-v0.1 2048 1 20

set -e

NUM_NODES=${1:?Usage: run_multinode_moe.sh <num_nodes> <ib_disable> <model_path> [seq_len] [batch_size] [steps]}
IB_DISABLE=${2:-0}
MODEL_PATH=${3:-/lustre/models/Mixtral-8x7B-v0.1}
SEQ_LEN=${4:-2048}
BATCH_SIZE=${5:-1}
STEPS=${6:-20}
MASTER_PORT=29500

# Read node list from hostfile
mapfile -t ALL_NODES < ~/hostfile_good
HEAD_NODE="${ALL_NODES[0]}"
MASTER_ADDR=$(getent ahostsv4 "$HEAD_NODE" | awk 'NR==1{print $1}')

# Take first N nodes
NODES=("${ALL_NODES[@]:0:$NUM_NODES}")
MODEL_NAME=$(basename "$MODEL_PATH")
IB_LABEL=$( [ "$IB_DISABLE" = "0" ] && echo "IB" || echo "ETH" )

echo "=== MoE Fine-tuning Benchmark ==="
echo "Config: ${MODEL_NAME} | ${NUM_NODES} nodes | ${IB_LABEL}"
echo "Nodes: ${NODES[*]}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT} (${HEAD_NODE})"
echo "IB_DISABLE=${IB_DISABLE} | seq_len=${SEQ_LEN} | bs=${BATCH_SIZE} | steps=${STEPS}"
echo ""

LOGDIR="/lustre/results/${MODEL_NAME}_${NUM_NODES}nodes_${IB_LABEL}"
mkdir -p "$LOGDIR"

# Clean up any stale containers on ALL nodes first
echo "Cleaning stale containers on all nodes..."
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    if [ "$NODE" = "$(hostname)" ]; then
        sudo docker rm -f bench_node${i} 2>/dev/null || true
    else
        ssh "$NODE" "sudo docker rm -f bench_node${i}" 2>/dev/null || true
    fi
done
sleep 2

# Launch workers (non-head nodes) in background
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    RANK=$i
    if [ "$NODE" != "$(hostname)" ]; then
        echo "Launching worker on ${NODE} (rank ${RANK})..."
        ssh "$NODE" "nohup bash /lustre/scripts/launch_node_moe.sh $NUM_NODES $RANK $MASTER_ADDR $MASTER_PORT $IB_DISABLE $MODEL_PATH $SEQ_LEN $BATCH_SIZE $STEPS > ${LOGDIR}/node${RANK}.log 2>&1 &"
        sleep 2
    fi
done

# Launch head node (rank 0) in foreground
echo "Launching head node $(hostname) (rank 0)..."
bash /lustre/scripts/launch_node_moe.sh $NUM_NODES 0 $MASTER_ADDR $MASTER_PORT $IB_DISABLE $MODEL_PATH $SEQ_LEN $BATCH_SIZE $STEPS 2>&1 | tee "${LOGDIR}/node0.log"

echo ""
echo "=== Head node finished ==="

# Collect worker logs
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    if [ "$NODE" != "$(hostname)" ]; then
        echo ""
        echo "=== ${NODE} (rank ${i}) last 10 lines ==="
        ssh "$NODE" "tail -10 ${LOGDIR}/node${i}.log" 2>/dev/null || echo "(no log)"
    fi
done

# Cleanup containers
echo ""
echo "Cleaning up containers..."
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    if [ "$NODE" = "$(hostname)" ]; then
        sudo docker rm -f bench_node${i} 2>/dev/null || true
    else
        ssh "$NODE" "sudo docker rm -f bench_node${i}" 2>/dev/null || true
    fi
done
echo "Done. Logs in ${LOGDIR}/"
