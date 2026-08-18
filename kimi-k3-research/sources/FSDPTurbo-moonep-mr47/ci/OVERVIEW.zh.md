# FSDPTurbo Runtime Docker 镜像概述

## 快速参考

| 项目 | 说明 |
| ------ | ------ |
| 镜像名称 | `fsdpturbo-ci` |
| 源码仓库 | [https://gitcode.com/Ascend/FSDPTurbo](https://gitcode.com/Ascend/FSDPTurbo) |
| Dockerfile 路径 | `docker/Dockerfile` |
| 默认场景 | FSDPTurbo 运行时环境（CANN + PyTorch + torch_npu） |
| 基础镜像 | 可配置 CANN 镜像，默认 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-openeuler24.03-py3.11` |
| 默认工作目录 | `/workspace` |

## 设计理念

本镜像是一个**运行时环境镜像**，只提供 CANN、PyTorch、torch_npu 和 FSDPTurbo 的 Python 依赖。**FSDPTurbo 源代码本身不在镜像中**，而是通过 `-v` 挂载宿主机目录的方式引入。这样你可以：

- 在宿主机上开发和修改 FSDPTurbo 代码
- 在容器内直接运行和调试
- 不需要每次改代码都重新构建镜像

## 镜像 Tag 关键字段描述

推荐 Tag 模板：

`{芯片信息}-{操作系统}-{Python标签}-{架构类型}`

示例：

- `910b-ubuntu22.04-py3.11-x86_64`
- `a3-openeuler24.03-py3.11-aarch64`

## 构建参数

| 参数 | 说明 | 默认值 |
| ------ | ------ | ------ |
| `-t, --npu-type` | NPU 类型：`a3` 或 `910b` | `910b` |
| `-o, --os` | 操作系统：`openeuler24.03` 或 `ubuntu22.04` | `openeuler24.03` |
| `--base-image-version` | CANN 基础镜像版本 | `9.0.0` |
| `--base-image` | 完整 CANN 基础镜像名，优先级高于 `--base-image-version`；会原样传入 | 空 |
| `--python-version` | CANN 基础镜像中的 Python 标签 | `3.11` |
| `--torch-version` | PyTorch 版本 | `2.10` |
| `--torch-npu-version` | TorchNPU 版本 | `2.10` |

## 快速开始

### 构建镜像

```bash
cd docker
bash build.sh
```

### 启动容器

复制下方启动命令前，请将 `{path-to-fsdpturbo}` 替换为宿主机上 FSDPTurbo 源码路径，`{path-to-data}`、`{path-to-weights}` 替换为数据和模型权重路径。也可以直接挂载整个用户目录 `-v /home/{user}:/home/{user}`，使容器内外路径完全一致。

```bash
docker run -it -d \
  --name fsdpturbo-ci \
  --pid=host \
  --network host \
  --ipc=host \
  --cgroupns host \
  --security-opt seccomp=unconfined \
  --cap-add=CAP_SYS_RESOURCE \
  --cap-add=CAP_SYS_ADMIN \
  --cap-add=CAP_MKNOD \
  --cap-add=CAP_SYS_PTRACE \
  --cap-add=CAP_IPC_LOCK \
  -e ASCEND_VISIBLE_DEVICES=0-7 \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --security-opt label=disable \
  --shm-size=32G \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v {path-to-fsdpturbo}:/workspace/FSDPTurbo \
  -v {path-to-data}:/data \
  -v {path-to-weights}:/weights \
  fsdpturbo-ci:910b-openeuler24.03-py3.11-aarch64 \
  /bin/bash
```

### 进入容器并安装 FSDPTurbo

```bash
docker exec -it fsdpturbo-ci /bin/bash

# 容器内执行：安装 FSDPTurbo（editable 模式，改代码无需重装）
pip install -e /workspace/FSDPTurbo

# 验证
python -c "import fsdp_turbo; print('FSDPTurbo OK')"
```

## 兼容性说明

- 当前版本采用统一 Dockerfile + 构建脚本结构，支持可配置的 CANN 基础镜像选择。
- 默认基础镜像使用 CANN 9.0.0、910b、openEuler 24.03、Python 3.11。
- 可以通过 `docker/build.sh` 切换 Ubuntu 22.04、a3 或其他 CANN 基础镜像版本。
- 镜像预装 PyTorch、TorchNPU 以及 FSDPTurbo 的 Python 依赖（transformers、datasets、pyyaml）。
- FSDPTurbo 源码需从宿主机挂载，容器内 `pip install -e` 安装后可随时修改代码。

## 许可证

FSDPTurbo 基于 Apache License 2.0 许可证发布。详见 [LICENSE](https://gitcode.com/Ascend/FSDPTurbo/blob/master/LICENSE) 文件。

与所有 Docker 镜像一样，这些镜像可能还包含受其他许可证约束的其他软件（例如基础发行版中的 Bash，以及所包含主要软件的任何直接或间接依赖项）。
