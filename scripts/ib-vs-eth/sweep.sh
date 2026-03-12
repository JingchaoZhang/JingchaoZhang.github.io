#!/bin/bash
# Full IB vs Ethernet fine-tuning sweep
# Runs: 2 models x 4 node counts x 2 IB modes = 16 experiments
# Usage: bash sweep.sh
#
# Prerequisites:
#   - /lustre/models/Qwen2.5-7B and /lustre/models/Qwen2.5-72B downloaded
#   - /lustre/scripts/ has finetune_bench.py, launch_node.sh, run_multinode.sh
#   - ~/hostfile_good has >= 8 healthy nodes
#   - PyTorch container pulled on all nodes

set -e

MODELS=("Qwen2.5-7B" "Qwen2.5-72B")
NODE_COUNTS=(1 2 4 8)
IB_MODES=(0 1)   # 0=IB enabled, 1=IB disabled (Ethernet)
SEQ_LEN=2048
STEPS=20

# Adjust batch size per model (72B needs smaller BS to fit in memory)
get_batch_size() {
    local model=$1
    local nodes=$2
    if [[ "$model" == *"72B"* ]]; then
        echo 1
    else
        echo 2
    fi
}

RESULTS_FILE="/lustre/results/sweep_summary.txt"
mkdir -p /lustre/results

echo "=== IB vs Ethernet Fine-tuning Sweep ===" | tee "$RESULTS_FILE"
echo "Started at $(date)" | tee -a "$RESULTS_FILE"
echo "Models: ${MODELS[*]}" | tee -a "$RESULTS_FILE"
echo "Node counts: ${NODE_COUNTS[*]}" | tee -a "$RESULTS_FILE"
echo "IB modes: 0=IB, 1=Ethernet" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

TOTAL=$((${#MODELS[@]} * ${#NODE_COUNTS[@]} * ${#IB_MODES[@]}))
RUN=0

for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="/lustre/models/${MODEL}"
    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Model not found at $MODEL_PATH" | tee -a "$RESULTS_FILE"
        continue
    fi

    for NODES in "${NODE_COUNTS[@]}"; do
        BS=$(get_batch_size "$MODEL" "$NODES")

        for IB_DISABLE in "${IB_MODES[@]}"; do
            RUN=$((RUN + 1))
            IB_LABEL=$( [ "$IB_DISABLE" = "0" ] && echo "IB" || echo "ETH" )

            echo "========================================" | tee -a "$RESULTS_FILE"
            echo "[$RUN/$TOTAL] ${MODEL} | ${NODES} nodes | ${IB_LABEL}" | tee -a "$RESULTS_FILE"
            echo "========================================" | tee -a "$RESULTS_FILE"

            # Run the benchmark
            bash /lustre/scripts/run_multinode.sh \
                "$NODES" "$IB_DISABLE" "$MODEL_PATH" "$SEQ_LEN" "$BS" "$STEPS" \
                2>&1 | tee -a "$RESULTS_FILE"

            # Extract key result
            LOGDIR="/lustre/results/${MODEL}_${NODES}nodes_${IB_LABEL}"
            if [ -f "${LOGDIR}/node0.log" ]; then
                echo "" | tee -a "$RESULTS_FILE"
                echo "--- Result ---" | tee -a "$RESULTS_FILE"
                grep -E "Tokens/sec:|Time/step:" "${LOGDIR}/node0.log" | tee -a "$RESULTS_FILE"
            fi
            echo "" | tee -a "$RESULTS_FILE"

            # Brief pause between runs
            sleep 10
        done
    done
done

echo "========================================" | tee -a "$RESULTS_FILE"
echo "Sweep finished at $(date)" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Print summary table
echo "=== SUMMARY TABLE ===" | tee -a "$RESULTS_FILE"
printf "%-15s %-6s %-5s %-15s %-15s\n" "Model" "Nodes" "Mode" "Tokens/sec" "ms/step" | tee -a "$RESULTS_FILE"
printf "%-15s %-6s %-5s %-15s %-15s\n" "-----" "-----" "----" "----------" "-------" | tee -a "$RESULTS_FILE"

for MODEL in "${MODELS[@]}"; do
    for NODES in "${NODE_COUNTS[@]}"; do
        for IB_DISABLE in "${IB_MODES[@]}"; do
            IB_LABEL=$( [ "$IB_DISABLE" = "0" ] && echo "IB" || echo "ETH" )
            LOGDIR="/lustre/results/${MODEL}_${NODES}nodes_${IB_LABEL}"
            if [ -f "${LOGDIR}/node0.log" ]; then
                TPS=$(grep "^Tokens/sec:" "${LOGDIR}/node0.log" | awk '{print $2}')
                MS=$(grep "^Time/step:" "${LOGDIR}/node0.log" | awk '{print $2}')
                printf "%-15s %-6s %-5s %-15s %-15s\n" "$MODEL" "$NODES" "$IB_LABEL" "${TPS:-FAIL}" "${MS:-FAIL}" | tee -a "$RESULTS_FILE"
            else
                printf "%-15s %-6s %-5s %-15s %-15s\n" "$MODEL" "$NODES" "$IB_LABEL" "NO_LOG" "NO_LOG" | tee -a "$RESULTS_FILE"
            fi
        done
    done
done

echo "" | tee -a "$RESULTS_FILE"
echo "Full logs: /lustre/results/" | tee -a "$RESULTS_FILE"
