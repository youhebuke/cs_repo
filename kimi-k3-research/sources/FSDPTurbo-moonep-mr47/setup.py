# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fsdp-turbo",
    version="0.1.0",
    author="Huawei Technologies Co., Ltd.",
    description="A high-performance and easy-to-use distributed training acceleration library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitcode.com/Ascend/FSDPTurbo",
    packages=find_packages(include=["fsdp_turbo", "fsdp_turbo.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "pyyaml>=6.0",
        "transformers>=5.0.0",
        "datasets>=2.14.0",
    ],
    extras_require={
        "npu": [
            "torch_npu",
        ],
        "moonep": [
            "torch>=2.9.0,<2.12.0",
            "moonep==0.0.1",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache-2.0",
)
