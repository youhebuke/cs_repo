#!/bin/bash
# ============================================================
# GPU functional check: Qwen3 MoE + FSDPTurbo + MoonEP
# ============================================================
# NPU memory (OPT-1 / OPT-2) is the real goal. This script is only
# the H20 path that has already been verified to train.
#
# PYTHONPATH must point at this tree. A different editable install
# will silently ignore these patches.
#
# GPU:  MOONEP_ASYNC_FINISH=0  (required on H20)
# NPU:  unset or MOONEP_ASYNC_FINISH=1 (original default)

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export FSDP_TURBO_ROOT="${FSDP_TURBO_ROOT:-$ROOT}"
export PYTHONPATH="${FSDP_TURBO_ROOT}:${PYTHONPATH:-}"

# H20: Buffer.dispatch(async_finish=True) SIGSEGVs. NPU should not use this script's default.
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
