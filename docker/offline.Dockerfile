ARG BASE_IMAGE=ghcr.io/synth-laboratories/nanohorizon-base:latest
FROM ${BASE_IMAGE}

RUN python3 -m pip install -q \
    "httpx>=0.28.1" \
    "pyyaml>=6.0.2" \
    "accelerate>=1.10.0" \
    "datasets>=4.1.0" \
    "peft>=0.17.0" \
    "transformers>=4.57.0" \
    "trl>=0.21.0" \
    "vllm>=0.10.0"
