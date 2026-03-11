#!/bin/bash
# Multi-node fine-tuning benchmark orchestrator
# Usage: bash run_multinode.sh <num_nodes> <node_list> [sharding]
# Example: bash run_multinode.sh 2 "vmssE6JHE7,vmssAYZGM2" hybrid
#          bash run_multinode.sh 3 "vmssE6JHE7,vmssAYZGM2,vmssCZIUQ2" full

NUM_NODES=${1:-2}
NODE_LIST=${2:-"vmssE6JHE7,vmssAYZGM2"}
SHARDING=${3:-full}
MASTER_ADDR="10.0.0.4"
MASTER_PORT=29500

IFS=',' read -ra NODES <<< "$NODE_LIST"

echo "=== Multi-node benchmark ==="
echo "Nodes: ${NUM_NODES} (${NODE_LIST})"
echo "Total GPUs: $((NUM_NODES * 8))"
echo "Sharding: ${SHARDING}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo ""

# Launch workers first (background via SSH)
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    RANK=$i
    if [ "$NODE" != "vmssE6JHE7" ]; then
        echo "Launching worker on ${NODE} (rank ${RANK})..."
        ssh "$NODE" "nohup bash ~/launch_node.sh $NUM_NODES $RANK $MASTER_ADDR $MASTER_PORT $SHARDING > ~/bench_node${RANK}.log 2>&1 &"
        sleep 2
    fi
done

# Launch head node (foreground, rank 0)
echo "Launching head node (rank 0)..."
bash ~/launch_node.sh $NUM_NODES 0 $MASTER_ADDR $MASTER_PORT $SHARDING 2>&1 | tee ~/bench_node0.log

echo ""
echo "=== Head node finished ==="

# Show worker logs
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    if [ "$NODE" != "vmssE6JHE7" ]; then
        echo ""
        echo "=== ${NODE} (rank ${i}) logs (last 20 lines) ==="
        ssh "$NODE" "tail -20 ~/bench_node${i}.log" 2>/dev/null
    fi
done

# Cleanup
echo ""
echo "Cleaning up containers..."
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    if [ "$NODE" == "vmssE6JHE7" ]; then
        sudo docker rm -f bench_node${i} 2>/dev/null
    else
        ssh "$NODE" "sudo docker rm -f bench_node${i}" 2>/dev/null
    fi
done
