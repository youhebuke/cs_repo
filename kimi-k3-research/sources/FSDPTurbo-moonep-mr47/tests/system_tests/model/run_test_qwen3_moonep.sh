#!/bin/bash
# ============================================================
# Run Qwen3 MoE system test with FSDPTurbo + MoonEP
# ============================================================
# Usage (from FSDPTurbo repo root):
#   bash tests/system_tests/model/run_test_qwen3_moonep.sh
#
# Prerequisites:
#   - PyTorch >= 2.9, moonep==0.0.1, pip install -e .
#   - 4+ GPUs on one NVLink/NVSwitch node (MoonEP requires intra-node EP)
#   - MODEL_PATH / DATASET_PARQUET_PATH point to local Qwen3 MoE + wikitext

# Diagnose native SIGSEGV:
#   export MOONEP_DEBUG_SYNC=1 CUDA_LAUNCH_BLOCKING=1 PYTHONFAULTHANDLER=1
# If prefetch still faults after the Event.wait fix:
#   export MOONEP_ASYNC_FINISH=0 MOONEP_ENABLE_PDL=0

export CUDA_VISIBLE_DEVICES=4,5,6,7
export FULLY_SHARD_PARALLEL_SIZE=4
export EXPERT_PARALLEL_SIZE=4

export MODEL_PATH=/home/w00949074/models/Qwen3-30B-A3B
export DATASET_PARQUET_PATH=/home/w00949074/datasets/wikitext-2-raw-v1

NUM_NODES=1
NUM_GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=29500

echo "Starting Qwen3 MoE + MoonEP system test..."
echo "Configuration:"
echo "  FSDP2 (fully_shard_parallel_size=${FULLY_SHARD_PARALLEL_SIZE})"
echo "  MoonEP EP (expert_parallel_size=${EXPERT_PARALLEL_SIZE})"
echo "  dispatcher=moonep (see test_moonep.py)"
echo ""

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    tests/system_tests/model/test_moonep.py
TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "MoonEP system test completed successfully."
else
    echo "MoonEP system test failed (exit code=${TEST_RESULT})."
fi
exit $TEST_RESULT
