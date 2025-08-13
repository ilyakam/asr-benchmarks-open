# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/New_York

WORKDIR /src

COPY requirements/requirements.txt /src/requirements/requirements.txt

# Install system dependencies and Python packages
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update || true && \
    apt-get install -y --no-install-recommends curl ffmpeg git nano software-properties-common tree wget && \
    add-apt-repository ppa:deadsnakes/ppa || true && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update || true && \
    apt-get install -y --no-install-recommends cuda-minimal-build-12-6 && \
    apt-get install -y --no-install-recommends python3.10 python3-dev && \
    ln -s `which python3.10` /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --timeout 300 --root-user-action=ignore -r requirements/requirements.txt && \
    # Clean up:
    python -m pip cache purge && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    find / -type f -name "*.pyc" -delete && \
    find / -type d -name "__pycache__" -exec rm -rf {} + && \
    rm -rf /usr/share/doc/* && \
    rm -rf /usr/share/man/* && \
    rm -rf /tmp/* && \
    rm -rf /var/tmp/* && \
    rm -rf /var/cache/apt/* && \
    find / -type d -name ".git" -exec rm -rf {} + && \
    find / -name "*.a" -or -name "*.la" -or -name "*.pdb" -or -name "*.md" -delete

ARG LIBRARY=base

COPY . /src

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$LIBRARY" != "base" ]; then \
        extra_reqs="" && \
        if [ "$LIBRARY" = "all" ]; then \
            extra_reqs="-r requirements/requirements_ctranslate2.txt -r requirements/requirements_nemo.txt"; \
        elif [ "$LIBRARY" = "ctranslate2" ]; then \
            extra_reqs="-r requirements/requirements_ctranslate2.txt"; \
        elif [ "$LIBRARY" = "nemo" ]; then \
            extra_reqs="-r requirements/requirements_nemo.txt"; \
        fi && \
        pip install --timeout 300 --root-user-action=ignore $extra_reqs && \
        python -m pip cache purge && \
        find / -type f -name "*.pyc" -delete && \
        find / -type d -name "__pycache__" -exec rm -rf {} +; \
    fi

# Tell the dynamic link loader where to find the NVIDIA CUDA cuDNN libraries:
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/:/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH"

ENV PYTHONPATH="/src:$PYTHONPATH"

WORKDIR /src
