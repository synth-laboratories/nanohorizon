ARG BASE_IMAGE=ghcr.io/synth-laboratories/nanohorizon-base:latest
FROM ${BASE_IMAGE}

ENV NANOHORIZON_VENV_ROOT=/opt/nanohorizon/venvs

RUN mkdir -p "${NANOHORIZON_VENV_ROOT}" \
    && python3 -m venv --system-site-packages "${NANOHORIZON_VENV_ROOT}/teacher" \
    && "${NANOHORIZON_VENV_ROOT}/teacher/bin/python" -m pip install --upgrade pip \
    && "${NANOHORIZON_VENV_ROOT}/teacher/bin/python" -m pip install \
        "httpx>=0.28.1" \
        "pyyaml>=6.0.2" \
        "vllm>=0.10.0" \
    && python3 -m venv --system-site-packages "${NANOHORIZON_VENV_ROOT}/training" \
    && "${NANOHORIZON_VENV_ROOT}/training/bin/python" -m pip install --upgrade pip \
    && "${NANOHORIZON_VENV_ROOT}/training/bin/python" -m pip install \
        "httpx>=0.28.1" \
        "pyyaml>=6.0.2" \
        "accelerate>=1.10.0" \
        "datasets>=4.1.0" \
        "peft>=0.15.0" \
        "trl>=0.28.0" \
    && "${NANOHORIZON_VENV_ROOT}/training/bin/python" -m pip install --upgrade \
        "transformers @ git+https://github.com/huggingface/transformers.git@main"
