#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACK="${1:-}"

if [[ -z "$TRACK" ]]; then
  echo "usage: $0 {base|offline|rlvr|prompt_opt|eval|nle} [tag]" >&2
  exit 1
fi

TAG="${2:-latest}"
REGISTRY="${NANOHORIZON_IMAGE_REGISTRY:-ghcr.io/synth-laboratories}"
BASE_IMAGE="${NANOHORIZON_BASE_IMAGE:-${REGISTRY}/nanohorizon-base:${TAG}}"
PLATFORM="${NANOHORIZON_DOCKER_PLATFORM:-linux/amd64}"
USE_BUILDX="${NANOHORIZON_USE_BUILDX:-1}"

case "$TRACK" in
  base)
    IMAGE_NAME="nanohorizon-base"
    DOCKERFILE="$ROOT/docker/base.Dockerfile"
    ;;
  offline)
    IMAGE_NAME="nanohorizon-offline"
    DOCKERFILE="$ROOT/docker/offline.Dockerfile"
    ;;
  rlvr)
    IMAGE_NAME="nanohorizon-rlvr"
    DOCKERFILE="$ROOT/docker/rlvr.Dockerfile"
    ;;
  prompt_opt)
    IMAGE_NAME="nanohorizon-prompt-opt"
    DOCKERFILE="$ROOT/docker/prompt_opt.Dockerfile"
    ;;
  eval)
    IMAGE_NAME="nanohorizon-eval"
    DOCKERFILE="$ROOT/docker/eval.Dockerfile"
    ;;
  nle)
    IMAGE_NAME="nanohorizon-nle"
    DOCKERFILE="$ROOT/docker/nle.Dockerfile"
    ;;
  *)
    echo "unknown track: $TRACK" >&2
    exit 1
    ;;
esac

FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "Building $FULL_TAG"
BUILD_ARGS=()
if [[ "$TRACK" != "base" && "$TRACK" != "nle" ]]; then
  BUILD_ARGS+=(--build-arg "BASE_IMAGE=$BASE_IMAGE")
fi
if [[ "$USE_BUILDX" == "1" ]]; then
  PUSH_FLAG=()
  if [[ "${NANOHORIZON_DOCKER_PUSH:-0}" == "1" ]]; then
    PUSH_FLAG+=(--push)
  else
    PUSH_FLAG+=(--load)
  fi
  CMD=(docker buildx build --platform "$PLATFORM")
  if [[ "${#BUILD_ARGS[@]}" -gt 0 ]]; then
    CMD+=("${BUILD_ARGS[@]}")
  fi
  CMD+=(-f "$DOCKERFILE" -t "$FULL_TAG")
  CMD+=("${PUSH_FLAG[@]}")
  CMD+=("$ROOT")
  "${CMD[@]}"
else
  CMD=(docker build)
  if [[ "${#BUILD_ARGS[@]}" -gt 0 ]]; then
    CMD+=("${BUILD_ARGS[@]}")
  fi
  CMD+=(-f "$DOCKERFILE" -t "$FULL_TAG" "$ROOT")
  "${CMD[@]}"
  if [[ "${NANOHORIZON_DOCKER_PUSH:-0}" == "1" ]]; then
    echo "Pushing $FULL_TAG"
    docker push "$FULL_TAG"
  fi
fi
