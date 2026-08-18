#!/bin/bash
set -x

WORKSPACE=${1:-${WORKSPACE}}
pr_id=$2
TARGET_BRANCH=${3:-"main"}

# 检查是否在docker容器中
if [ ! -f "/.dockerenv" ]; then
    echo "Not in a docker container."
    exit 1
fi

# 运行环境准备
function setup_running_env() {
    # 安装FSDPTurbo
    echo "Installing FSDPTurbo..."
    cd "${WORKSPACE}"/CODE
    pip install -e . -v

    echo "Install Complete..."
    pip list
}

# 运行环境清理
function clean_running_env () {
    echo "Cleaning up environment..."
    pip uninstall -y fsdp-turbo
    echo "Finish cleaning"
}

function run_testcase () {
    cd "${WORKSPACE}"/CODE

    for test_script in ./tests/system_tests/model/run_test*.sh; do
        if [ -f "$test_script" ]; then
            echo "bash -e $test_script"
            bash -e $test_script
            if [ $? -ne 0 ]; then
                echo "Test failed: $test_script"
                return 1
            fi
        fi
    done

    cd ..
}

function main() {

    # 准备环境
    echo "Setting up environment..."
    setup_running_env

    # 执行用例
    echo "Running tests..."
    run_testcase
    local test_result=$?

    # 清理环境
    echo "Cleaning environment..."
    clean_running_env

    exit $test_result
}

main
