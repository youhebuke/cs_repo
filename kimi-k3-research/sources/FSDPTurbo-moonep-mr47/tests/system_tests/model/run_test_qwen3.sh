#!/bin/bash
# ============================================================
# Run Qwen3 complete system tests with FSDPTurbo
# ============================================================

export MODEL_PATH=/home/dataset/Qwen3-30B-A3B
export DATASET_PATH=/home/dataset/wikitext
export DATASET_PARQUET_PATH=/home/dataset/wikitext/wikitext-2-raw-v1

NUM_NODES=1
NUM_GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29500

echo "Starting Qwen3 complete test suite..."
echo "This test will run the following configuration.:"
echo "  FSDP2 + EP + TP + Recompute + Mixed Precision + Hook Module "
echo ""

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    tests/system_tests/model/test_qwen3.py
TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "All Qwen3 tests completed."
else
    echo "Qwen3 tests failed."
fi
exit $TEST_RESULT
