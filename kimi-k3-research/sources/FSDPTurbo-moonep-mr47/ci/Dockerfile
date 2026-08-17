# ============================================
# FSDPTurbo Runtime Docker Image
# Provides CANN + PyTorch + torch_npu environment
# FSDPTurbo source code is mounted from host at runtime
# NPU Type: Configurable (910b, a3, etc.)
# Supports: x86_64 and aarch64
# Supports: Ubuntu and openEuler CANN base images
# ============================================

ARG OS=openeuler24.03
ARG BASE_IMAGE_VERSION=9.0.0
ARG NPU_TYPE=910b
ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=""

FROM ${BASE_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${BASE_IMAGE_VERSION}-${NPU_TYPE}-${OS}-py${PYTHON_VERSION}}

USER root
SHELL ["/bin/bash", "-c"]

ARG OS_FAMILY=ubuntu
ARG TORCH_VERSION=2.10
ARG TORCH_NPU_VERSION=2.10

# Prepare required system dependencies
COPY configure_repo.sh /tmp/configure_repo.sh
RUN chmod +x /tmp/configure_repo.sh && \
    bash /tmp/configure_repo.sh && \
    rm -f /tmp/configure_repo.sh

RUN ARCH=$(uname -m) && \
    echo "Detected CPU architecture: ${ARCH}" && \
    if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then \
        echo "ERROR: Unsupported architecture: ${ARCH}"; \
        exit 1; \
    fi

RUN echo "Installing system dependencies for ${OS_FAMILY}..." && \
    if [ "$OS_FAMILY" = "ubuntu" ]; then \
        apt-get install -y --no-install-recommends \
            build-essential \
            cmake \
            curl \
            gcc \
            g++ \
            git \
            jq \
            libnuma-dev \
            vim \
            wget && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*; \
    elif [ "$OS_FAMILY" = "openeuler" ]; then \
        yum install -y \
            cmake \
            curl \
            findutils \
            gcc \
            gcc-c++ \
            git \
            jq \
            make \
            numactl-devel \
            tar \
            vim \
            wget \
            which && \
        yum clean all && \
        rm -rf /var/cache/yum; \
    else \
        echo "ERROR: Unsupported OS family: ${OS_FAMILY}"; \
        exit 1; \
    fi && \
    rm -rf /tmp/* /var/tmp/* && \
    pip install --upgrade pip packaging setuptools==80.10.2 && \
    pip cache purge

# Install PyTorch, torch_npu, and FSDPTurbo runtime dependencies
RUN ARCH=$(uname -m) && \
    pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple/ && \
    pip config set global.trusted-host repo.huaweicloud.com && \
    if [ "$ARCH" = "x86_64" ]; then \
        pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"; \
    fi && \
    DEVLIB_DIR=$(find /usr/local/Ascend -type d -path "*/devlib/linux/${ARCH}" | head -n 1) && \
    if [ -n "$DEVLIB_DIR" ]; then \
        export LD_LIBRARY_PATH="${DEVLIB_DIR}:${LD_LIBRARY_PATH}"; \
    fi && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then \
        source /usr/local/Ascend/nnal/atb/set_env.sh; \
    fi && \
    pip install torch==${TORCH_VERSION} torch_npu==${TORCH_NPU_VERSION} && \
    pip install pyyaml>=6.0 transformers==5.0.0 datasets>=2.14.0 && \
    rm -rf /tmp/* /var/tmp/* && \
    pip cache purge

# Default workspace — mount your FSDPTurbo project at /workspace/FSDPTurbo
RUN mkdir -p /workspace
WORKDIR /workspace

# Environment setup script for CANN toolkit
RUN echo '#!/bin/bash' > /etc/profile.d/ascend.sh && \
    echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null' >> /etc/profile.d/ascend.sh && \
    echo '[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh' >> /etc/profile.d/ascend.sh && \
    chmod +x /etc/profile.d/ascend.sh

# Show install results
RUN pip list

# Setting Default Commands
CMD ["/bin/bash"]
