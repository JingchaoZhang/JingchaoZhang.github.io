#!/bin/bash
# MoE IB vs Ethernet sweep
# Mixtral-8x7B: 1, 2, 4, 8 nodes × IB/ETH = 8 experiments
# (Skips 1-node if it doesn't fit — handled by timeout/fail)
# Usage: bash sweep_moe.sh

MODEL="Mixtral-8x7B-v0.1"
MODEL_PATH="/lustre/models/${MODEL}"
NODE_COUNTS=(1 2 4 8)
IB_MODES=(0 1)   # 0=IB enabled, 1=Ethernet
SEQ_LEN=2048
BATCH_SIZE=1      # Conservative for MoE memory
STEPS=20

RESULTS_FILE="/lustre/results/sweep_moe_summary.txt"
mkdir -p /lustre/results

echo "=== MoE IB vs Ethernet Sweep ===" | tee "$RESULTS_FILE"
echo "Started at $(date)" | tee -a "$RESULTS_FILE"
echo "Model: ${MODEL} (~46.7B total, ~12.9B active)" | tee -a "$RESULTS_FILE"
echo "Node counts: ${NODE_COUNTS[*]}" | tee -a "$RESULTS_FILE"
echo "IB modes: 0=IB, 1=Ethernet" | tee -a "$RESULTS_FILE"
echo "seq_len=${SEQ_LEN} batch_size=${BATCH_SIZE} steps=${STEPS}" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH" | tee -a "$RESULTS_FILE"
    echo "Run: huggingface-cli download mistralai/Mixtral-8x7B-v0.1 --local-dir $MODEL_PATH" | tee -a "$RESULTS_FILE"
    exit 1
fi

TOTAL=$((${#NODE_COUNTS[@]} * ${#IB_MODES[@]}))
RUN=0

for NODES in "${NODE_COUNTS[@]}"; do
    for IB_DISABLE in "${IB_MODES[@]}"; do
        RUN=$((RUN + 1))
        IB_LABEL=$( [ "$IB_DISABLE" = "0" ] && echo "IB" || echo "ETH" )

        echo "========================================" | tee -a "$RESULTS_FILE"
        echo "[$RUN/$TOTAL] ${MODEL} | ${NODES} nodes | ${IB_LABEL}" | tee -a "$RESULTS_FILE"
        echo "========================================" | tee -a "$RESULTS_FILE"

        # Clean up all containers before each run
        mapfile -t ALL_NODES < ~/hostfile_good
        for i in $(seq 0 $((NODES - 1))); do
            NODE="${ALL_NODES[$i]}"
            if [ "$NODE" = "$(hostname)" ]; then
                sudo docker rm -f bench_node${i} 2>/dev/null || true
            else
                ssh "$NODE" "sudo docker rm -f bench_node${i}" 2>/dev/null || true
            fi
        done
        sleep 5

        # Run benchmark (allow failures — some configs may OOM)
        if bash /lustre/scripts/run_multinode_moe.sh \
            "$NODES" "$IB_DISABLE" "$MODEL_PATH" "$SEQ_LEN" "$BATCH_SIZE" "$STEPS" \
            2>&1 | tee -a "$RESULTS_FILE"; then
            echo "Run completed successfully." | tee -a "$RESULTS_FILE"
        else
            echo "Run FAILED (exit code $?)." | tee -a "$RESULTS_FILE"
        fi

        # Extract key result
        LOGDIR="/lustre/results/${MODEL}_${NODES}nodes_${IB_LABEL}"
        if [ -f "${LOGDIR}/node0.log" ]; then
            echo "" | tee -a "$RESULTS_FILE"
            echo "--- Result ---" | tee -a "$RESULTS_FILE"
            grep -E "Tokens/sec:|Time/step:" "${LOGDIR}/node0.log" | tee -a "$RESULTS_FILE" || true
        fi
        echo "" | tee -a "$RESULTS_FILE"

        sleep 10
    done
done

echo "========================================" | tee -a "$RESULTS_FILE"
echo "Sweep finished at $(date)" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Summary table
echo "=== MOE SWEEP SUMMARY ===" | tee -a "$RESULTS_FILE"
printf "%-25s %-6s %-5s %-15s %-15s\n" "Model" "Nodes" "Mode" "Tokens/sec" "ms/step" | tee -a "$RESULTS_FILE"
printf "%-25s %-6s %-5s %-15s %-15s\n" "-----" "-----" "----" "----------" "-------" | tee -a "$RESULTS_FILE"

for NODES in "${NODE_COUNTS[@]}"; do
    for IB_DISABLE in "${IB_MODES[@]}"; do
        IB_LABEL=$( [ "$IB_DISABLE" = "0" ] && echo "IB" || echo "ETH" )
        LOGDIR="/lustre/results/${MODEL}_${NODES}nodes_${IB_LABEL}"
        if [ -f "${LOGDIR}/node0.log" ]; then
            TPS=$(grep "^Tokens/sec:" "${LOGDIR}/node0.log" | awk '{print $2}')
            MS=$(grep "^Time/step:" "${LOGDIR}/node0.log" | awk '{print $2}')
            printf "%-25s %-6s %-5s %-15s %-15s\n" "$MODEL" "$NODES" "$IB_LABEL" "${TPS:-FAIL}" "${MS:-FAIL}" | tee -a "$RESULTS_FILE"
        else
            printf "%-25s %-6s %-5s %-15s %-15s\n" "$MODEL" "$NODES" "$IB_LABEL" "NO_LOG" "NO_LOG" | tee -a "$RESULTS_FILE"
        fi
    done
done
echo "" | tee -a "$RESULTS_FILE"
echo "Full logs: /lustre/results/" | tee -a "$RESULTS_FILE"
