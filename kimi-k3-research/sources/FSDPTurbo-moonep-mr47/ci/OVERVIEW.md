# FSDPTurbo Runtime Docker Overview

## Quick Reference

| Item | Description |
| ------ | ------ |
| Image Name | `fsdpturbo-ci` |
| Repository | [https://gitcode.com/Ascend/FSDPTurbo](https://gitcode.com/Ascend/FSDPTurbo) |
| Dockerfile Path | `docker/Dockerfile` |
| Default Scenario | FSDPTurbo runtime environment (CANN + PyTorch + torch_npu) |
| Base Image | Configurable CANN image, default `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-openeuler24.03-py3.11` |
| Default Working Directory | `/workspace` |

## Design Philosophy

This is a **runtime environment image** — it provides CANN, PyTorch, torch_npu, and FSDPTurbo's Python dependencies only. **FSDPTurbo source code is NOT in the image**; it is mounted from the host via `-v` at container startup. This allows you to:

- Develop and modify FSDPTurbo code on the host
- Run and debug directly inside the container
- Avoid rebuilding the image for every code change

## Key Fields in the Image Tag

Recommended tag template:

`{chip_info}-{os}-{python_tag}-{arch}`

Examples:

- `910b-ubuntu22.04-py3.11-x86_64`
- `a3-openeuler24.03-py3.11-aarch64`

## Build Options

| Option | Description | Default |
| ------ | ------ | ------ |
| `-t, --npu-type` | NPU type: `a3` or `910b` | `910b` |
| `-o, --os` | Operating system: `openeuler24.03` or `ubuntu22.04` | `openeuler24.03` |
| `--base-image-version` | CANN base image version | `9.0.0` |
| `--base-image` | Full CANN base image name, higher priority than `--base-image-version`; passed through unchanged | empty |
| `--python-version` | Python tag in the CANN base image | `3.11` |
| `--torch-version` | PyTorch version | `2.10` |
| `--torch-npu-version` | TorchNPU version | `2.10` |

## Quick Start

### Build

```bash
cd docker
bash build.sh
```

### Start Container

Before running, replace `{path-to-fsdpturbo}` with the host path to FSDPTurbo source, and `{path-to-data}` / `{path-to-weights}` with actual data/weights paths. You can also mount the entire home directory with `-v /home/{user}:/home/{user}` to keep all paths identical between host and container.

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

### Enter Container and Install FSDPTurbo

```bash
docker exec -it fsdpturbo-ci /bin/bash

# Inside container: install FSDPTurbo in editable mode (code changes take effect immediately)
pip install -e /workspace/FSDPTurbo

# Verify
python -c "import fsdp_turbo; print('FSDPTurbo OK')"
```

## Compatibility Notes

- This image uses a unified Dockerfile and build script for configurable CANN base image selection.
- The default base image uses CANN 9.0.0, 910b, openEuler 24.03, and Python 3.11.
- You can switch to Ubuntu 22.04, a3, or a different CANN base image version through `docker/build.sh`.
- The image pre-installs PyTorch, TorchNPU, and FSDPTurbo's Python dependencies (transformers, datasets, pyyaml).
- FSDPTurbo source is mounted from the host; after `pip install -e` in the container, code changes take effect immediately.

## License

FSDPTurbo is released under the Apache License 2.0. See the [LICENSE](https://gitcode.com/Ascend/FSDPTurbo/blob/master/LICENSE) file for details.

Like all Docker images, these images may also contain other software under other licenses, such as Bash from the base distribution and any direct or indirect dependencies of the included main software.
