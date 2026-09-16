FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ARG TORCH_VERSION=2.11.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

RUN python -m pip install --break-system-packages --upgrade pip setuptools wheel \
    && python -m pip install --break-system-packages \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torch==${TORCH_VERSION}"

COPY . /workspace

RUN python -m pip install --break-system-packages \
        numpy==2.4.3 \
        pandas==3.0.1 \
    && python -m pip install --break-system-packages -e ".[dev]"

CMD ["python", "-m", "pytest", "-q"]
