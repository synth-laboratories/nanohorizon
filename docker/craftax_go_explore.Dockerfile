FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Pre-install craftax + jax + httpx so they're cached in the image
RUN uv pip install --system \
    "craftax>=1.5" \
    "jax>=0.5" \
    "httpx>=0.28" \
    "numpy"

# Pre-warm JAX compilation cache by importing craftax
RUN python3 -c "from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv; print('craftax imported')"

WORKDIR /home/daytona/workspace
