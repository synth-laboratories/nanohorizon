ARG BASE_IMAGE=ghcr.io/synth-laboratories/nanohorizon-base:latest
FROM ${BASE_IMAGE}

RUN python3 -m pip install -q \
    "httpx>=0.28.1" \
    "pyyaml>=6.0.2"
