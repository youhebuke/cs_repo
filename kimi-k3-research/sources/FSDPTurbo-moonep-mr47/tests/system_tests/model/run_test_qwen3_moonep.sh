#!/bin/bash
# ============================================================
# GPU functional check: Qwen3 MoE + FSDPTurbo + MoonEP
# ============================================================
# The two memory changes target NPU. This script only exists so the
# same tree can train on H20. Export MOONEP_ASYNC_FINISH=0 on GPU.
# NPU should keep async_finish=1 (test_moonep.py default).

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export FSDP_TURBO_ROOT="${FSDP_TURBO_ROOT:-$ROOT}"
export PYTHONPATH="${FSDP_TURBO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export MOONEP_ASYNC_FINISH="${MOONEP_ASYNC_FINISH:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export FULLY_SHARD_PARALLEL_SIZE="${FULLY_SHARD_PARALLEL_SIZE:-4}"
export EXPERT_PARALLEL_SIZE="${EXPERT_PARALLEL_SIZE:-4}"

export MODEL_PATH="${MODEL_PATH:-/home/w00949074/models/Qwen3-30B-A3B}"
export DATASET_PARQUET_PATH="${DATASET_PARQUET_PATH:-/home/w00949074/datasets/wikitext-2-raw-v1}"

NUM_NODES=1
NUM_GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=29500

echo "Starting Qwen3 MoE + MoonEP system test..."
echo "  FSDP_TURBO_ROOT=${FSDP_TURBO_ROOT}"
echo "  PYTHONPATH=${PYTHONPATH}"
echo "  MOONEP_ASYNC_FINISH=${MOONEP_ASYNC_FINISH}"
echo "  dispatcher=moonep"
echo ""

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    "${FSDP_TURBO_ROOT}/tests/system_tests/model/test_moonep.py"
TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "MoonEP system test completed successfully."
else
    echo "MoonEP system test failed (exit code=${TEST_RESULT})."
fi
exit $TEST_RESULT
