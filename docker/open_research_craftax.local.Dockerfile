ARG BASE_IMAGE=synth-local-open-research-craftax:latest
FROM ${BASE_IMAGE}

WORKDIR /work

COPY pyproject.toml README.md ./
COPY src ./src
COPY workspace ./workspace

ENV PYTHONPATH=/work/src \
    JAX_DISABLE_JIT=true \
    JAX_PLATFORMS=cpu \
    JAX_PLATFORM_NAME=cpu \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    XLA_FLAGS=--xla_force_host_platform_device_count=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.15
