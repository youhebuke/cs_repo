#!/bin/bash
# ============================================================
# DeepSeekV4 Distributed Training Launch Script
# ============================================================

set -e

export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export ASCEND_LAUNCH_BLOCKING=1
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Cluster Configuration ---
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# --- Training Arguments ---
CONFIG_FILE=${CONFIG_FILE:-"config.yaml"}

echo "============================================================"
echo "DeepSeekV4 Training"
echo "  NNODES: $NNODES"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  CONFIG: $CONFIG_FILE"
echo "============================================================"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config "$CONFIG_FILE"
