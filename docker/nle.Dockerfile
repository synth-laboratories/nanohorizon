ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -q \
    "fastapi>=0.115.0" \
    "gymnasium>=1.0.0" \
    "httpx>=0.28.1" \
    "imageio>=2.37.0" \
    "imageio-ffmpeg>=0.6.0" \
    "nle>=1.1.0" \
    "numpy>=2.0.0" \
    "pillow>=11.3.0" \
    "pyyaml>=6.0.2" \
    "uvicorn>=0.32.0"

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
ENV PYTHONPATH=/workspace/src

CMD ["python3", "-m", "nanohorizon.nle_core.http_shim"]
