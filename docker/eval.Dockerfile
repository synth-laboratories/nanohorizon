ARG BASE_IMAGE=ghcr.io/synth-laboratories/nanohorizon-base:latest
FROM ${BASE_IMAGE}

RUN python3 -m pip install -q \
    "pyyaml>=6.0.2" \
    "peft>=0.17.0" \
    "transformers>=4.57.0"
